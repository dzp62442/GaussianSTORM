"""Iteration-based OmniScene training/evaluation using STORM's model and optimizer."""

import copy
from dataclasses import asdict, dataclass, field
import json
import logging
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from storm.dataset.omniscene_adapter import prepare_omniscene_batch
from storm.dataset.omniscene_dataset import OmniSceneDataset
from storm.dataset.samplers import InfiniteSampler
from storm.evaluation.omniscene import ImageMetrics, check_local_vgg, evaluate_omniscene
from storm.utils.experiment import code_revision, count_parameters, dependency_versions, file_sha256, model_weights_sha256, write_json
from storm.utils.feishu import FeishuNotifier

LOGGER = logging.getLogger("STORM")
MODEL_KEYS = ("gs_dim", "decoder_type", "num_cams", "num_motion_tokens", "use_sky_token",
              "use_affine_token", "use_latest_gsplat", "near", "far", "scale_offset",
              "opacity_offset", "max_scale", "tau", "projected_motion_dim", "disable_pos_embed",
              "sigmoid_rgb", "static_scene")


@dataclass
class TrainingState:
    completed_steps: int = 0
    validation_count: int = 0
    last_validation_step: int = 0
    tested_steps: list = field(default_factory=list)
    final_test_complete: bool = False

    def due(self, args):
        step = self.completed_steps
        validation = step > 0 and step % args.val_every_n_iters == 0 and self.last_validation_step < step
        periodic_test = step > 0 and step % (args.val_every_n_iters * args.test_every_n_validations) == 0
        final_test = step == args.num_iterations and not self.final_test_complete
        mini = (periodic_test and step not in self.tested_steps) or final_test
        return validation, mini


def build_model(args):
    from storm.models import STORM_models
    return STORM_models[args.model](img_size=args.input_size,
                                   grad_checkpointing=not args.disable_grad_checkpointing,
                                   **{key: getattr(args, key) for key in MODEL_KEYS})


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def training_signature(args):
    ignored = {"config", "overrides", "output_dir", "exp_name", "project", "entity", "device",
               "load_from", "resume_from", "auto_resume", "dry_run", "mode", "test_split",
               "gpu_memory_limit_gb", "check_limit", "num_workers",
               "enable_wandb", "enable_feishu", "feishu_module_paths",
               "lpips_weights", "log_dir", "ckpt_dir", "video_dir"}
    return {key: value for key, value in vars(args).items() if key not in ignored}


def save_checkpoint(path, model, optimizer, scaler, state, args, train_dataset, eval_manifest_sha256):
    payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
               "model_weights_sha256": model_weights_sha256(model),
               "loss_scaler": scaler.state_dict(), "state": asdict(state), "rng": rng_state(),
               "signature": training_signature(args), "config": vars(args),
               "train_manifest_sha256": train_dataset.manifest_sha256,
               "eval_manifest_sha256": eval_manifest_sha256,
               "sampler": {"seed": args.seed, "consumed": state.completed_steps},
               "latest_step": state.completed_steps - 1}
    temporary = Path(str(path) + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def select_resume(args, checkpoints):
    if args.resume_from:
        return Path(args.resume_from)
    if args.auto_resume:
        final = checkpoints / "ckpt_final.pth"
        if final.is_file():
            return final
        choices = sorted(checkpoints.glob("ckpt_step_*.pth"))
        if choices:
            return choices[-1]
    return None


def cleanup_checkpoints(directory, keep):
    if keep == 0:
        return
    for path in sorted(directory.glob("ckpt_step_*.pth"))[:-keep]:
        path.unlink()


def _device(args):
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("STORM rasterization requires CUDA; use --dry_run or --mode check-data for CPU checks")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; use --dry_run / --mode check-data for CPU checks")
        device = torch.device("cuda", device.index if device.index is not None else 0)
        torch.cuda.set_device(device)
        if args.gpu_memory_limit_gb is not None:
            free, total = torch.cuda.mem_get_info(device)
            limit = int(args.gpu_memory_limit_gb * 1024 ** 3)
            if limit <= 0 or limit > free - 1024 ** 3:
                raise RuntimeError("Requested GPU cap leaves less than 1 GiB of current free VRAM")
            torch.cuda.set_per_process_memory_fraction(limit / total, device)
    return device


def run(args):
    if args.dry_run:
        print(yaml.safe_dump(vars(args), allow_unicode=True, sort_keys=True))
        return
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Reviewed protocol uses one GPU and global batch size 1")
    if args.mode == "check-data":
        reports = [OmniSceneDataset(args, split).check_assets(args.check_limit) for split in ("train", "total")]
        print(json.dumps(reports, ensure_ascii=False, indent=2))
        if any(report["errors"] for report in reports):
            raise RuntimeError("OmniScene asset check failed")
        return
    log_dir = Path(args.output_dir) / args.project / args.exp_name
    checkpoints = log_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    from storm.utils.logging import setup_logging
    setup_logging(output=str(log_dir), level=logging.INFO)
    provenance = {**code_revision(), "command": sys.argv, "dependencies": dependency_versions()}
    if provenance["dependencies"]["gsplat_commit"] != "2b0de894232d21e8963179a7bbbd315f27c52c9c":
        raise RuntimeError("Install the README-pinned gsplat commit before OmniScene experiments")
    device = _device(args)
    check_local_vgg()
    if args.enable_perceptual_loss and args.mode == "train":
        from storm.utils.lpips import MD5_MAP, md5_hash
        if not Path(args.lpips_weights).is_file() or md5_hash(args.lpips_weights) != MD5_MAP["vgg_lpips"]:
            raise FileNotFoundError(f"Existing STORM LPIPS weights required: {args.lpips_weights}")
    from storm.utils.misc import fix_random_seeds
    fix_random_seeds(args.seed)
    model = build_model(args).to(device)
    params = count_parameters(model)
    LOGGER.info("OmniScene parameters: %s", params)
    if args.mode == "test":
        if not args.load_from:
            raise ValueError("--mode test requires --load_from")
        checkpoint = torch.load(args.load_from, map_location="cpu")
        for key in (*MODEL_KEYS, "model", "input_size"):
            if checkpoint.get("config", {}).get(key) != getattr(args, key):
                raise ValueError(f"Test model configuration differs from checkpoint: {key}")
        model.load_state_dict(checkpoint["model"], strict=True)
        state = checkpoint.get("state", {})
        step = state.get("completed_steps", checkpoint.get("latest_step", -1) + 1)
        provenance.update(checkpoint=args.load_from, checkpoint_sha256=file_sha256(args.load_from), completed_steps=step)
        dataset = OmniSceneDataset(args, args.test_split)
        output = log_dir / "eval" / args.test_split / f"step_{step:06d}"
        if args.max_eval_bins:
            output = output / f"limited_{args.max_eval_bins}"
        write_json(output / "config.json", vars(args))
        summary = evaluate_omniscene(model, dataset, args, output, provenance, save_visuals=True)
        FeishuNotifier(args, log_dir).mini_completed(summary, output, step, params)
        return summary
    return train(model, args, device, log_dir, checkpoints, provenance)


def train(model, args, device, log_dir, checkpoints, provenance):
    from timm.optim.optim_factory import param_groups_weight_decay
    from storm.utils.logging import WandbLogger
    from storm.utils.losses import compute_loss
    from storm.utils.lpips_loss import RGBLpipsLoss
    from storm.utils.misc import NativeScalerWithGradNormCount, adjust_learning_rate

    train_dataset = OmniSceneDataset(args, "train")
    validation = OmniSceneDataset(args, "val")
    mini = OmniSceneDataset(args, "mini")
    optimizer = torch.optim.AdamW(param_groups_weight_decay(model, args.weight_decay), lr=args.lr, betas=(0.9, 0.95))
    scaler = NativeScalerWithGradNormCount(enabled=device.type == "cuda")
    state = TrainingState()
    resume = select_resume(args, checkpoints)
    if not resume and any(checkpoints.glob("*.pth")):
        raise FileExistsError(f"Checkpoints already exist in {checkpoints}; use --resume_from or a new exp_name")
    pending_rng = None
    if resume:
        payload = torch.load(resume, map_location="cpu")
        if payload["signature"] != training_signature(args):
            changed = [key for key in payload["signature"] if payload["signature"][key] != training_signature(args).get(key)]
            raise ValueError(f"Resume config differs: {changed}")
        if payload["train_manifest_sha256"] != train_dataset.manifest_sha256:
            raise ValueError("Training manifest changed since checkpoint")
        if payload.get("eval_manifest_sha256") != validation.manifest_sha256:
            raise ValueError("Evaluation manifest changed or was not recorded in checkpoint")
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        scaler.load_state_dict(payload["loss_scaler"])
        state = TrainingState(**payload["state"])
        if payload["sampler"] != {"seed": args.seed, "consumed": state.completed_steps}:
            raise ValueError("Checkpoint sampler cursor is inconsistent")
        pending_rng = payload["rng"]
        del payload
    if not 0 <= state.completed_steps <= args.num_iterations:
        raise ValueError("Invalid checkpoint step count")
    (log_dir / "resolved_config.yaml").write_text(yaml.safe_dump(vars(args), allow_unicode=True, sort_keys=True))
    write_json(log_dir / "provenance.json", provenance)
    parameters = count_parameters(model)
    write_json(log_dir / "parameters.json", parameters)
    for dataset in (train_dataset, validation, mini):
        write_json(log_dir / f"{dataset.split}_bins.json", dataset.metadata())
    loss_module = RGBLpipsLoss(perceptual_weight=args.perceptual_weight,
                               enable_perceptual_loss=args.enable_perceptual_loss,
                               weights_path=args.lpips_weights, offline=True,
                               perceptual_chunk_size=1,
                               checkpoint_perceptual=not args.disable_grad_checkpointing).to(device).eval()
    metric = ImageMetrics(device)
    logger = None
    if args.enable_wandb:
        # Apply the resolved config automatically, including to W&B subprocesses.
        os.environ["WANDB_MODE"] = args.wandb_mode
        log_args = copy.copy(args)
        log_args.log_dir = str(log_dir)
        # Offline runs are local segments; do not ask the server to resume a run.
        logger = WandbLogger(log_args, resume=False)
    if pending_rng is not None:
        restore_rng(pending_rng)
    notifier = FeishuNotifier(args, log_dir)
    start_step = state.completed_steps
    started_at = time.perf_counter()
    latest_train_log = None

    def checkpoint_path():
        return checkpoints / ("ckpt_final.pth" if state.completed_steps == args.num_iterations
                              else f"ckpt_step_{state.completed_steps:06d}.pth")

    def save():
        save_checkpoint(checkpoint_path(), model, optimizer, scaler, state, args, train_dataset,
                        validation.manifest_sha256)
        cleanup_checkpoints(checkpoints, args.keep_n_ckpts)
        write_json(log_dir / "training_state.json", asdict(state))

    def evaluate_pending():
        val_due, mini_due = state.due(args)
        for do_eval, dataset in ((val_due, validation), (mini_due, mini)):
            if not do_eval:
                continue
            info = {**provenance, "completed_steps": state.completed_steps,
                    "checkpoint": str(checkpoint_path()) if checkpoint_path().exists() else "in-memory weights"}
            output = log_dir / "eval" / dataset.split / f"step_{state.completed_steps:06d}"
            if args.max_eval_bins:
                output = output / f"limited_{args.max_eval_bins}"
            summary = evaluate_omniscene(model, dataset, args, output, info, metric,
                                         save_visuals=bool(args.vis_every_n_iters and
                                                           state.completed_steps % args.vis_every_n_iters == 0))
            if dataset.split == "val":
                state.validation_count += 1
                state.last_validation_step = state.completed_steps
            else:
                if state.completed_steps not in state.tested_steps:
                    state.tested_steps.append(state.completed_steps)
                state.final_test_complete = state.completed_steps == args.num_iterations
            if logger:
                logger.set_step(state.completed_steps)
                logger.update({f"{dataset.split}/{group}/{key}": value for group, scores in summary["groups"].items()
                               for key, value in scores.items()})
            write_json(log_dir / "training_state.json", asdict(state))
            if dataset.split == "mini":
                elapsed = time.perf_counter() - started_at
                updates = state.completed_steps - start_step
                remaining = args.num_iterations - state.completed_steps
                eta = elapsed / updates * remaining if updates else (0 if remaining == 0 else None)
                notifier.mini_completed(summary, output, state.completed_steps, parameters,
                                        final=state.final_test_complete, training_log=latest_train_log,
                                        elapsed=elapsed, eta=eta)

    try:
        if state.completed_steps < args.num_iterations or not state.final_test_complete:
            notifier.training_started(state.completed_steps, parameters, resume)
        # A checkpoint can precede its evaluations; complete them before another update.
        evaluate_pending()
        sampler = InfiniteSampler(len(train_dataset), shuffle=True, seed=args.seed, start=0, step=1,
                                  advance=state.completed_steps)
        loader = DataLoader(train_dataset, sampler=sampler, batch_size=args.batch_size,
                            num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
                            drop_last=True, generator=torch.Generator().manual_seed(args.seed))
        iterator = iter(loader) if state.completed_steps < args.num_iterations else None
        with (log_dir / "training_metrics.jsonl").open("a") as metrics_file:
            while state.completed_steps < args.num_iterations:
                batch = next(iterator)
                model.train()
                loss_module.set_perceptual_loss(args.enable_perceptual_loss and
                                                state.completed_steps >= args.perceptual_loss_start_iter)
                adjust_learning_rate(optimizer, state.completed_steps, args)
                inputs, cameras, target, _ = prepare_omniscene_batch(batch, device, args.timespan)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.precision == "bf16"):
                    output = model({**inputs, **cameras, "render_chunk_size": args.eval_render_chunk_size})
                    losses = compute_loss(output, target, args, loss_module)
                    loss = sum(value for key, value in losses.items() if "loss" in key)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite training loss: {batch['scene']}")
                scale_before = scaler.state_dict().get("scale", 1.0)
                norm = scaler(loss, optimizer, parameters=model.parameters(), clip_grad=args.grad_clip)
                if scaler.state_dict().get("scale", 1.0) < scale_before:
                    raise FloatingPointError("GradScaler skipped an update; stopping without counting it")
                state.completed_steps += 1
                row = {"completed_steps": state.completed_steps, "bin": batch["scene"][0],
                       "lr": optimizer.param_groups[0]["lr"], "grad_norm": float(norm),
                       **{key: float(value.detach()) for key, value in losses.items()}}
                if device.type == "cuda":
                    row["process_peak_allocated_gib"] = torch.cuda.max_memory_allocated(device) / 1024 ** 3
                    row["process_peak_reserved_gib"] = torch.cuda.max_memory_reserved(device) / 1024 ** 3
                metrics_file.write(json.dumps(row, allow_nan=False) + "\n")
                metrics_file.flush()
                latest_train_log = row
                if state.completed_steps % args.log_every_n_iters == 0:
                    LOGGER.info("Training %d/%d %s", state.completed_steps, args.num_iterations, row)
                    if logger:
                        logger.set_step(state.completed_steps)
                        logger.update({key: value for key, value in row.items() if key != "bin"})
                # Release the training graph before validation/mini evaluation.
                del output, losses, loss, inputs, cameras, target, batch
                optimizer.zero_grad(set_to_none=True)
                save_due = state.completed_steps % args.ckpt_every_n_iters == 0 or state.completed_steps == args.num_iterations
                if save_due:
                    save()  # Persist pending evaluations for crash recovery.
                evaluate_pending()
                if save_due:
                    save()
        if not state.final_test_complete:
            raise RuntimeError("Final mini test was not completed")
        save()
        LOGGER.info("Training complete: %s", asdict(state))
        return asdict(state)
    finally:
        if logger:
            logger.finish()
