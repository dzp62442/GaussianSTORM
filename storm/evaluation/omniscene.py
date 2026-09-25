"""Full-image OmniScene all_18 / novel_12 metrics, matching depthsplat/SVF-GS."""

import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from storm.dataset.omniscene_adapter import prepare_omniscene_batch
from storm.utils.experiment import count_parameters, dependency_versions, gpu_activity, model_weights_sha256, summarize_times, timed_reconstruction, write_json

GROUPS = {"all_18": slice(0, 18), "novel_12": slice(0, 12)}
LOGGER = logging.getLogger("STORM")


def check_local_vgg():
    path = Path(torch.hub.get_dir()) / "checkpoints" / "vgg16-397923af.pth"
    if not path.is_file():
        raise FileNotFoundError(f"Offline evaluation requires existing VGG16 weights: {path}")


def compute_pcc(reference, prediction):
    if reference.shape != prediction.shape:
        raise ValueError("PCC depth shapes differ")
    x, y = reference.reshape(-1).double(), prediction.reshape(-1).double()
    if x.numel() < 2 or not torch.isfinite(x).all() or not torch.isfinite(y).all():
        return x.new_tensor(float("nan"))
    x, y = x - x.mean(), y - y.mean()
    denominator = x.norm() * y.norm()
    if denominator <= 0 or not torch.isfinite(denominator):
        return x.new_tensor(float("nan"))
    return (x.dot(y) / denominator).clamp(-1, 1)


class ImageMetrics:
    def __init__(self, device):
        check_local_vgg()
        from lpips import LPIPS
        self.lpips = LPIPS(net="vgg").to(device).eval().requires_grad_(False)

    @torch.no_grad()
    def __call__(self, reference, prediction):
        from skimage.metrics import structural_similarity
        reference, prediction = reference.float().clamp(0, 1), prediction.float().clamp(0, 1)
        if reference.shape != prediction.shape:
            raise ValueError("Rendered and reference RGB shapes differ")
        mse = (reference - prediction).square().mean(dim=(1, 2, 3))
        psnr = -10 * mse.log10()
        ssim = [structural_similarity(gt, pred, win_size=11, gaussian_weights=True,
                                      channel_axis=0, data_range=1.0)
                for gt, pred in zip(reference.cpu().numpy(), prediction.cpu().numpy())]
        # One view at a time prevents the metric network from dominating VRAM.
        with torch.autocast(device_type=prediction.device.type, enabled=False):
            lpips = torch.cat([self.lpips(gt[None], pred[None], normalize=True).flatten()
                               for gt, pred in zip(reference, prediction)])
        return {"psnr": psnr, "ssim": torch.as_tensor(ssim, device=prediction.device), "lpips": lpips}


def json_metric(value):
    """Strict JSON values without fabricating scores for undefined metrics."""
    if np.isnan(value):
        return None
    if np.isposinf(value):
        return "Infinity"
    if np.isneginf(value):
        return "-Infinity"
    return float(value)


def group_records(token, image_metrics, reference_depth, accumulated_depth):
    if reference_depth.shape[0] != 18 or accumulated_depth.shape != reference_depth.shape:
        raise ValueError("PCC requires all eighteen target depths in protocol order")
    rows = []
    for group, indices in GROUPS.items():
        row = {"bin": token, "view_group": group, "views": indices.stop,
               **{name: values[indices].mean().item() for name, values in image_metrics.items()},
               "pcc": compute_pcc(reference_depth[indices], accumulated_depth[indices]).item()}
        notes = {}
        for key in ("psnr", "ssim", "lpips", "pcc"):
            row[key] = json_metric(row[key])
            if row[key] is None:
                notes[key] = "undefined"
            elif isinstance(row[key], str):
                notes[key] = row[key]
        if row["pcc"] is None:
            reference, prediction = reference_depth[indices], accumulated_depth[indices]
            if not torch.isfinite(reference).all():
                notes["pcc"] = "nonfinite_relative_depth"
            elif not torch.isfinite(prediction).all():
                notes["pcc"] = "nonfinite_rendered_depth"
            else:
                notes["pcc"] = "zero_variance_or_undefined_correlation"
        if notes:
            row["metric_notes"] = notes
        rows.append(row)
    return rows


def aggregate_records(records, expected_tokens):
    summaries = {}
    for group in GROUPS:
        selected = [row for row in records if row["view_group"] == group]
        tokens = [row["bin"] for row in selected]
        if tokens != expected_tokens:
            raise ValueError(f"Incomplete or out-of-order {group} coverage")
        summaries[group] = {}
        for key in ("psnr", "ssim", "lpips", "pcc"):
            values = [row[key] for row in selected]
            # Keep full-set semantics: one undefined value makes this aggregate
            # undefined, instead of silently reporting a subset's mean.
            with np.errstate(invalid="ignore"):
                summaries[group][key] = (None if any(value is None for value in values)
                                         else json_metric(np.mean([float(value) for value in values])))
    return summaries


def save_preview(path, gt, prediction):
    def grid(images):
        images = (images.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
        return np.concatenate([np.concatenate(images[i:i + 6], axis=1) for i in (0, 6, 12)], axis=0)
    Image.fromarray(np.concatenate([grid(gt), grid(prediction)], axis=0)).save(path)


@torch.inference_mode()
def evaluate_omniscene(model, dataset, args, output_dir, provenance, image_metric=None, save_visuals=False):
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    training = model.training
    model.eval()
    records, timings, errors = [], [], []
    metadata = {**provenance, **{key: value for key, value in dataset.metadata().items() if key != "bins"},
                "model_weights_sha256": model_weights_sha256(model),
                "reference": "da2", "depth_mode": "accumulated_z", "requested_resolution": list(args.input_size),
                "actual_resolution": list(args.input_size), "dynamic_mask_scope_train": "novel_12",
                "evaluation_mask": "none", "view_order": "camera-major novel pairs then six inputs",
                "render_chunk_size": args.eval_render_chunk_size, "radius_clip": args.eval_radius_clip,
                "precision": args.precision, "torch": torch.__version__, "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                "dependencies": dependency_versions(),
                "gpu_activity_start": gpu_activity(device), "valid_gpu_measurement": False, "complete": False}
    write_json(directory / "selected_bins.json", dataset.metadata())
    write_json(directory / "summary.json", metadata)
    write_json(directory / "parameters.json", count_parameters(model))
    loader = DataLoader(dataset, batch_size=args.test_batch_size, shuffle=False,
                        num_workers=args.num_workers, persistent_workers=False,
                        generator=torch.Generator().manual_seed(args.seed), pin_memory=False)
    try:
        metric = image_metric or ImageMetrics(device)
        with (directory / "records.jsonl").open("w") as handle:
            for index, batch in enumerate(loader):
                token = batch["scene"][0]
                inputs, cameras, _, targets = prepare_omniscene_batch(batch, device, args.timespan)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=args.precision == "bf16"):
                    gaussians, elapsed = timed_reconstruction(model, inputs)
                    rendered = model.render_static(gaussians, cameras, depth_mode=args.pcc_depth_mode,
                                                   chunk_size=args.eval_render_chunk_size,
                                                   radius_clip=args.eval_radius_clip)
                rgb = (rendered["rendered_image"][0, 0].permute(0, 3, 1, 2).float() * 0.5 + 0.5).clamp(0, 1)
                depth = rendered["accumulated_depth"][0, 0].float()
                rows = group_records(token, metric(targets["rgb"][0], rgb), targets["rel_depth"][0], depth)
                for row in rows:
                    handle.write(json.dumps(row, allow_nan=False) + "\n")
                handle.flush()
                records.extend(rows)
                timings.append({"bin": token, "milliseconds": elapsed,
                                "gaussians": gaussians["means"].numel() // 3})
                if save_visuals and index < args.num_vis_samples:
                    save_preview(directory / f"{index:03d}_{token}.png", targets["rgb"][0], rgb)
                if (index + 1) % args.log_every_n_iters == 0:
                    LOGGER.info("%s evaluation %d/%d", dataset.split, index + 1, len(dataset))
                del gaussians, rendered, inputs, cameras, targets, rgb, depth
        groups = aggregate_records(records, dataset.bin_tokens)
        issues = [{"bin": row["bin"], "view_group": row["view_group"], "metrics": row["metric_notes"]}
                  for row in records if row.get("metric_notes")]
        undefined = {group: {key: sum(row[key] is None for row in records if row["view_group"] == group)
                             for key in ("psnr", "ssim", "lpips", "pcc")} for group in GROUPS}
        metadata.update(complete=True, completed_count=len(records) // 2, error_count=0, groups=groups,
                        metrics_defined=all(value is not None for scores in groups.values() for value in scores.values()),
                        undefined_metric_counts=undefined, metric_issue_count=len(issues))
        write_json(directory / "metric_issues.json", issues)
        if issues:
            LOGGER.warning("%s evaluation completed with %d metric notes; see %s. Training can continue.",
                           dataset.split, len(issues), directory / "metric_issues.json")
        for group in GROUPS:
            suffix = "" if group == "all_18" else "_novel_12"
            selected = [row for row in records if row["view_group"] == group]
            write_json(directory / f"scores_pcc_all{suffix}.json", [row["pcc"] for row in selected])
            write_json(directory / f"scores_all_avg{suffix}.json", groups[group])
        return metadata
    except Exception as exc:
        errors.append({"error": str(exc), "completed_bins": len(records) // 2})
        metadata.update(error_count=len(errors), errors=errors, completed_count=len(records) // 2)
        raise
    finally:
        metadata["gpu_activity_end"] = gpu_activity(device)
        metadata["valid_gpu_measurement"] = (
            device.type == "cuda" and len(timings) > args.timing_warmup_samples
            and metadata["gpu_activity_start"]["other_compute_pids"] == []
            and metadata["gpu_activity_end"]["other_compute_pids"] == [])
        if device.type == "cuda":
            metadata["process_peak_allocated_gib"] = torch.cuda.max_memory_allocated(device) / 1024 ** 3
            metadata["process_peak_reserved_gib"] = torch.cuda.max_memory_reserved(device) / 1024 ** 3
        write_json(directory / "summary.json", metadata)
        write_json(directory / "reconstruction_time.json", summarize_times(timings, args.timing_warmup_samples))
        model.train(training)
