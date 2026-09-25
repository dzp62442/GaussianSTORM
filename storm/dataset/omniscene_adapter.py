"""Separate network inputs, target cameras and supervision before the forward pass."""

import torch


def prepare_omniscene_batch(batch, device, timespan=2.0):
    context, target = batch["context"], batch["target"]
    image = context["image"].to(device)
    b, v, c, h, w = image.shape
    if b != 1 or v != 6 or c != 3 or target["image"].shape[1] != 18:
        raise ValueError("Expected batch=1, 6 RGB inputs and 18 targets")
    inputs = {
        # Normalize inside reconstruct_static so normalization is timed too.
        "context_image": image[:, None],
        "context_intrinsics": context["intrinsics"].to(device)[:, None],
        "context_camtoworlds": context["extrinsics"].to(device)[:, None],
        "context_time": torch.zeros(b, 1, device=device),
    }
    cameras = {"target_intrinsics": target["intrinsics"].to(device)[:, None],
               "target_camtoworlds": target["extrinsics"].to(device)[:, None],
               "target_camera_ids": target["camera_ids"].to(device)[:, None],
               "context_time": inputs["context_time"], "target_time": torch.zeros(b, 1, device=device),
               "timespan": timespan, "height": h, "width": w}
    supervision = {"target_image": target["image"].to(device)[:, None] * 2 - 1,
                   "target_depth": target["depth"].to(device)[:, None],
                   "target_valid_mask": target["masks"].to(device)[:, None]}
    metrics = {"rgb": target["image"].to(device)}
    if "rel_depth" in target:
        metrics["rel_depth"] = target["rel_depth"].to(device)
    return inputs, cameras, supervision, metrics
