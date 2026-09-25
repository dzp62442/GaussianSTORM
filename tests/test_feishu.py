"""Notification protocol tests; all senders are mocked, with no network or GPU."""

from argparse import Namespace
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from main_storm import get_args_parser
from storm.omniscene_config import ROOT, parse_args
from storm.omniscene_runner import training_signature
from storm.utils.feishu import FeishuNotifier, load_sender


def config(*overrides, resolution="112x200"):
    return parse_args(get_args_parser(), ["--config", str(ROOT / f"configs/experiment/omniscene_{resolution}.yaml"),
                                         *overrides])


PARAMETERS = {"trainable": 100, "frozen": 1, "total": 101}
SUMMARY = {"split": "mini", "complete": True, "completed_count": 2, "expected_count": 2,
           "uncapped_count": 2048, "limited": True, "valid_gpu_measurement": False,
           "checkpoint": "checkpoints/ckpt_final.pth",
           "groups": {"all_18": dict(psnr=21., ssim=.81, lpips=.21, pcc=.71),
                      "novel_12": dict(psnr=19., ssim=.79, lpips=.23, pcc=.69)}}


class FeishuTests(unittest.TestCase):
    def test_defaults_overrides_and_resume_compatibility(self):
        for resolution in ("112x200", "224x400"):
            args = config(resolution=resolution)
            self.assertTrue(args.enable_feishu)
            self.assertEqual(args.feishu_module_paths[0], "~/Libraries")
            old_signature = vars(deepcopy(args))
            # Notification settings never invalidate existing pre-notification checkpoints.
            del old_signature["enable_feishu"], old_signature["feishu_module_paths"]
            self.assertEqual(training_signature(args), training_signature(Namespace(**old_signature)))
            disabled = deepcopy(args)
            disabled.enable_feishu = False
            disabled.feishu_module_paths = []
            self.assertEqual(training_signature(args), training_signature(disabled))
        self.assertFalse(config("--no-enable_feishu").enable_feishu)
        for override in ('enable_feishu="false"', "feishu_module_paths=wrong", "feishu_module_paths=[null]"):
            with self.assertRaises(ValueError):
                config("--set", override)

    def test_disabled_does_not_import_or_write(self):
        with tempfile.TemporaryDirectory() as directory, patch("storm.utils.feishu.load_sender") as loader:
            notifier = FeishuNotifier(config("--no-enable_feishu"), directory)
            notifier.training_started(0, PARAMETERS)
            notifier.mini_completed(SUMMARY, directory, 100001, PARAMETERS)
            loader.assert_not_called()
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_search_paths_are_added_only_when_loading(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(sys, "path", list(sys.path)), \
                patch("storm.utils.feishu.import_module") as importer:
            sender = importer.return_value.send_feishu
            self.assertIs(load_sender([directory, directory, directory + "/missing"]), sender)
            self.assertEqual(sys.path.count(directory), 1)
            self.assertNotIn(directory + "/missing", sys.path)
            importer.assert_called_once_with("auto_monitor.send_feishu")

    def test_start_and_mini_contents_and_local_status(self):
        with tempfile.TemporaryDirectory() as directory, patch("storm.utils.feishu.load_sender") as loader:
            sender = loader.return_value = Mock(return_value=True)
            args = config()
            notifier = FeishuNotifier(args, directory)
            notifier.training_started(0, PARAMETERS)
            self.assertIn("训练启动", sender.call_args.args[0])
            self.assertIn("W&B：offline", sender.call_args.args[1])
            notifier.training_started(10000, PARAMETERS, "resume.pth")
            self.assertIn("训练恢复", sender.call_args.args[0])
            self.assertIn("resume.pth", sender.call_args.args[1])
            (Path(directory) / "reconstruction_time.json").write_text(json.dumps({"mean": 12.3, "count": 1}))
            notifier.mini_completed(SUMMARY, directory, 100001, PARAMETERS, final=True,
                                    training_log={"completed_steps": 100001, "lr": 0., "rgb_loss": .2},
                                    elapsed=100, eta=0)
            title, body = sender.call_args.args
            self.assertIn("最终 mini 测试完成（调试截断）", title)
            for expected in ("all_18：PSNR 21.000", "novel_12：PSNR 19.000", "PCC 0.6900",
                             "2/2 bin", "2048 bin", "非有效独占 GPU 计时", "12.30 ms/bin",
                             "可训练 100，冻结 1，总计 101", "rgb_loss=0.2", "100001/100001", "0h 00m 00s"):
                self.assertIn(expected, body)
            loader.assert_called_once()
            records = [json.loads(line) for line in (Path(directory) / "feishu_notifications.jsonl").read_text().splitlines()]
            self.assertEqual([row["event"] for row in records], ["training_started", "training_started", "mini_completed"])
            self.assertTrue(all(row["sent"] for row in records))

    def test_skip_incomplete_and_other_splits(self):
        with tempfile.TemporaryDirectory() as directory, patch("storm.utils.feishu.load_sender") as loader:
            notifier = FeishuNotifier(config(), directory)
            for changed in ({"complete": False}, {"split": "val"}, {"split": "total"}):
                notifier.mini_completed({**SUMMARY, **changed}, directory, 1000, PARAMETERS)
            loader.assert_not_called()

    def test_failure_does_not_raise_or_leak_sender_errors(self):
        with tempfile.TemporaryDirectory() as directory, patch("storm.utils.feishu.load_sender") as loader:
            notifier = FeishuNotifier(config(), directory)
            loader.side_effect = ModuleNotFoundError("private detail")
            with self.assertLogs("STORM", level="WARNING") as log:
                notifier.training_started(0, PARAMETERS)
            self.assertNotIn("private detail", "\n".join(log.output))
            loader.side_effect = None
            def failure(*unused):
                print("private webhook URL")
                raise RuntimeError("private webhook URL")
            loader.return_value = failure
            captured = io.StringIO()
            with patch("sys.stdout", captured), self.assertLogs("STORM", level="WARNING") as log:
                notifier.training_started(1, PARAMETERS)
            self.assertNotIn("private webhook URL", captured.getvalue() + "\n".join(log.output))
            notifier.sender = Mock(return_value=False)
            with self.assertLogs("STORM", level="WARNING"):
                notifier.training_started(2, PARAMETERS)
            rows = [json.loads(line) for line in (Path(directory) / "feishu_notifications.jsonl").read_text().splitlines()]
            self.assertFalse(any(row["sent"] for row in rows))
            self.assertEqual(rows[0]["error_type"], "ModuleNotFoundError")
            self.assertEqual(rows[1]["error_type"], "RuntimeError")


if __name__ == "__main__":
    unittest.main()
