"""Local experiment provenance, unique parameter counts and synchronized timing."""

import hashlib
from importlib.metadata import distribution
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dependency_versions():
    versions = {name: distribution(name).version for name in
                ("torch", "torchvision", "gsplat", "lpips", "scikit-image", "pillow")}
    direct = distribution("gsplat").read_text("direct_url.json")
    versions["gsplat_commit"] = json.loads(direct).get("vcs_info", {}).get("commit_id") if direct else None
    return versions


def code_revision():
    root = Path(__file__).resolve().parents[2]
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root, text=True).strip()
    # Include tracked edits and untracked implementation files, since review runs
    # legitimately precede a commit. Do not include runtime artifacts.
    paths = sorted(set(git("ls-files", "-z").split("\0")) | set(
        git("ls-files", "--others", "--exclude-standard", "-z").split("\0")))
    digest = hashlib.sha256()
    for name in paths:
        path = root / name
        if path.is_file():
            digest.update(name.encode())
            digest.update(path.read_bytes())
    return {"git_sha": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain")),
            "source_sha256": digest.hexdigest()}


def count_parameters(model):
    unique = {id(p): p for p in model.parameters()}.values()
    counts = {"trainable": 0, "frozen": 0}
    for parameter in unique:
        counts["trainable" if parameter.requires_grad else "frozen"] += parameter.numel()
    counts["total"] = counts["trainable"] + counts["frozen"]
    counts["excludes"] = ["loss networks", "metric networks", "buffers", "optimizer state"]
    return counts


def model_weights_sha256(model):
    """Stable identity across checkpoint metadata/RNG updates and in-memory evals."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(f"{name}:{tensor.dtype}:{tuple(tensor.shape)}".encode())
        data = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
        digest.update(data.tobytes())
    return digest.hexdigest()


def gpu_activity(device):
    """Record shared-GPU contention outside the measured forward region."""
    if device.type != "cuda":
        return {"other_compute_pids": None}
    try:
        import pynvml
        pynvml.nvmlInit()
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        index = device.index if device.index is not None else torch.cuda.current_device()
        identity = visible.split(",")[index] if visible else str(index)
        handle = (pynvml.nvmlDeviceGetHandleByIndex(int(identity)) if identity.isdigit()
                  else pynvml.nvmlDeviceGetHandleByUUID(identity))
        pids = sorted({process.pid for process in pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                       if process.pid != os.getpid()})
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return {"other_compute_pids": pids, "device_used_gib": memory.used / 1024 ** 3,
                "device_free_gib": memory.free / 1024 ** 3}
    except Exception as exc:
        return {"other_compute_pids": None, "activity_error": str(exc)}


def timed_reconstruction(model, inputs):
    device = inputs["context_image"].device
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    gaussians = model.reconstruct_static(inputs)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return gaussians, (time.perf_counter() - start) * 1000


def summarize_times(records, warmup):
    values = [record["milliseconds"] for record in records[warmup:]]
    return {"unit": "ms/bin", "warmup_excluded": min(warmup, len(records)),
            "count": len(values), "mean": float(np.mean(values)) if values else None,
            "median": float(np.median(values)) if values else None,
            "p90": float(np.percentile(values, 90)) if values else None,
            "records": records,
            "boundary": "GPU inputs -> normalized RGB/rays/encoder/activated Gaussians/motion/sky token/affine",
            "excludes": ["data loading", "host-to-device copy", "target rendering", "metrics"]}
