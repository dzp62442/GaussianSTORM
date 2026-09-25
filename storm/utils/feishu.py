"""Best-effort OmniScene notifications through SVF-GS's external send_feishu."""

from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from importlib import import_module
import io
import json
import logging
from pathlib import Path
import sys

LOGGER = logging.getLogger("STORM")


def load_sender(module_paths):
    # Match SVF-GS's local/cloud search roots without copying webhook credentials.
    for directory in module_paths:
        path = str(Path(directory).expanduser().resolve())
        if Path(path).is_dir() and path not in sys.path:
            sys.path.append(path)
    return import_module("auto_monitor.send_feishu").send_feishu


def _duration(seconds):
    if seconds is None:
        return "待估算"
    hours, seconds = divmod(max(0, int(seconds)), 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:d}h {minutes:02d}m {seconds:02d}s"


def _parameters(counts):
    return (f"模型参数：可训练 {counts['trainable']:,}，冻结 {counts['frozen']:,}，"
            f"总计 {counts['total']:,}")


def _score(value, decimals):
    if value is None:
        return "未定义"
    if isinstance(value, str):
        return {"Infinity": "+∞", "-Infinity": "-∞"}.get(value, value)
    return f"{value:.{decimals}f}"


class FeishuNotifier:
    def __init__(self, args, log_dir):
        self.args = args
        self.log_dir = Path(log_dir)
        self.sender = None

    def _send(self, event, step, title, lines):
        if not self.args.enable_feishu:
            return
        body = "\n".join(lines)
        record = {"time": datetime.now(timezone.utc).isoformat(), "event": event,
                  "completed_steps": step, "title": title, "body": body, "sent": False}
        try:
            # The external module prints request errors that can contain the webhook URL.
            # Keep only its success result; never copy credentials into experiment logs.
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                if self.sender is None:
                    self.sender = load_sender(self.args.feishu_module_paths)
                record["sent"] = self.sender(title, body) is True
        except Exception as exc:
            record["error_type"] = type(exc).__name__
        if record["sent"]:
            LOGGER.info("Feishu notification sent: %s (step %d)", event, step)
        else:
            LOGGER.warning("Feishu notification failed: %s (step %d; %s). Training continues; "
                           "check auto_monitor dependencies and ~/.feishu_env.",
                           event, step, record.get("error_type", "send_feishu returned failure"))
        try:
            with (self.log_dir / "feishu_notifications.jsonl").open("a") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            LOGGER.warning("Cannot save Feishu notification status (%s); continuing.", type(exc).__name__)

    def _context(self, step):
        height, width = self.args.input_size
        return [f"实验：{self.args.project}/{self.args.exp_name}",
                f"模型：{self.args.model}；分辨率：{height}×{width}",
                f"工作目录：{self.log_dir}",
                f"已完成迭代：{step}/{self.args.num_iterations}"]

    def training_started(self, step, parameters, resume=None):
        if not self.args.enable_feishu:
            return
        title = "GaussianSTORM OmniScene 训练恢复" if resume else "GaussianSTORM OmniScene 训练启动"
        interval = self.args.val_every_n_iters * self.args.test_every_n_validations
        lines = self._context(step) + [
            f"初始化：{resume or '从头训练'}", _parameters(parameters),
            f"batch size：{self.args.batch_size}；学习率峰值：{self.args.lr:g}；"
            f"调度：{self.args.lr_sched}；warmup：{self.args.warmup_iters} 步",
            f"验证间隔：{self.args.val_every_n_iters} 步；mini 测试间隔：{interval} 步；训练结束必做 mini 测试",
            f"W&B：{self.args.wandb_mode if self.args.enable_wandb else '关闭'}",
        ]
        self._send("training_started", step, title, lines)

    def mini_completed(self, summary, output_dir, step, parameters, *, final=False,
                       training_log=None, elapsed=None, eta=None):
        if (not self.args.enable_feishu or summary.get("split") != "mini"
                or not summary.get("complete")):
            return
        title = "GaussianSTORM OmniScene 最终 mini 测试完成" if final else "GaussianSTORM OmniScene mini 测试完成"
        if summary.get("limited"):
            title += "（调试截断）"
        if summary.get("metrics_defined") is False:
            title += "（含未定义指标）"
        lines = self._context(step) + [
            f"mini 覆盖：{summary['completed_count']}/{summary['expected_count']} bin；"
            f"未截断集合：{summary['uncapped_count']} bin",
        ]
        for group in ("all_18", "novel_12"):
            scores = summary["groups"][group]
            lines.append(f"{group}：PSNR {_score(scores['psnr'], 3)}，SSIM {_score(scores['ssim'], 4)}，"
                         f"LPIPS {_score(scores['lpips'], 4)}，PCC {_score(scores['pcc'], 4)}")
        if summary.get("metrics_defined") is False:
            lines.append("未定义项计数：" + json.dumps(summary["undefined_metric_counts"], ensure_ascii=False))
        lines.append(_parameters(parameters))
        timing_path = Path(output_dir) / "reconstruction_time.json"
        try:
            timing = json.loads(timing_path.read_text())
            if timing.get("mean") is not None:
                label = "独占 GPU 计时" if summary.get("valid_gpu_measurement") else "仅调试参考，非有效独占 GPU 计时"
                lines.append(f"完整高斯重建：均值 {timing['mean']:.2f} ms/bin；"
                             f"预热后 {timing['count']} bin（{label}）")
            else:
                lines.append("完整高斯重建耗时：预热后无样本")
        except (OSError, ValueError, KeyError, TypeError):
            lines.append("完整高斯重建耗时：汇总不可用")
        if training_log:
            values = ", ".join(f"{key}={value:.6g}" for key, value in training_log.items()
                               if "loss" in key or key in {"lr", "grad_norm"})
            lines.append(f"最近训练日志（第 {training_log['completed_steps']} 步）：{values}")
        if elapsed is not None:
            lines.append(f"本次运行耗时：{_duration(elapsed)}；剩余训练时间估计：{_duration(eta)}（含评估均摊）")
        lines += [f"Checkpoint：{summary.get('checkpoint', '未记录')}", f"评估输出：{output_dir}"]
        self._send("mini_completed", step, title, lines)
