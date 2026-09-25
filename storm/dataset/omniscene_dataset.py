"""OmniScene static 6 -> 18 protocol, ported from depthsplat and SVF-GS.

Only existing RGB, camera metadata, Metric3D, dynamic masks and evaluation DA2
are opened. In particular the LIDAR_TOP record is never accessed.
"""

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from storm.omniscene_config import CAMERAS

CAMERA_IDS = [i for i in range(6) for _ in range(2)] + list(range(6))


def da2_to_relative_depth(disp):
    # Preserve the reference loaders' formula, including undefined results for
    # constant maps. Evaluation records an undefined PCC without rejecting a bin.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        ratio = min(disp.max() / (disp.min() + 0.001), 50.0)
        depth = 1.0 / np.maximum(disp, disp.max() / ratio)
        return (depth - depth.min()) / (depth.max() - depth.min())


def resize_depth(array, size):
    array = array.astype(np.float32)
    if tuple(array.shape) != tuple(size):
        array = np.asarray(Image.fromarray(array).resize((size[1], size[0]), Image.Resampling.BILINEAR))
    return array.copy()


class OmniSceneDataset(Dataset):
    def __init__(self, args, split):
        if split not in {"train", "val", "mini", "total"}:
            raise ValueError(split)
        self.args, self.split = args, split
        self.root = Path(args.data_root)
        self.size = tuple(args.input_size)
        self.load_rel_depth = split != "train"
        self.manifest_path = self.root / args.data_version / (
            args.train_manifest if split == "train" else args.eval_manifest)
        raw = self.manifest_path.read_bytes()
        self.manifest_sha256 = hashlib.sha256(raw).hexdigest()
        bins = json.loads(raw)["bins"]
        if split in {"val", "mini"}:
            rule = getattr(args, f"{split}_selection")
            bins = bins[rule.get("offset", 0):rule.get("stop"):rule.get("stride", 1)]
            bins = bins[:rule.get("limit")]
        self.full_count = len(bins)
        if split != "train" and args.max_eval_bins:
            bins = bins[:args.max_eval_bins]
        if not bins:
            raise ValueError(f"Empty bin list: {self.manifest_path}")
        self.bin_tokens = bins
        self.selected_sha256 = hashlib.sha256(json.dumps(bins).encode()).hexdigest()

    def __len__(self):
        return len(self.bin_tokens)

    def metadata(self):
        return {"split": self.split, "manifest": str(self.manifest_path),
                "manifest_sha256": self.manifest_sha256, "selected_sha256": self.selected_sha256,
                "bins": self.bin_tokens, "expected_count": len(self),
                "uncapped_count": self.full_count, "limited": len(self) != self.full_count}

    def view_records(self, token):
        path = self.root / self.args.data_version / "bin_infos_3.2m" / f"{token}.pkl"
        with path.open("rb") as handle:
            sensors = pickle.load(handle)["sensor_info"]
        for cam in CAMERAS:
            if len(sensors[cam]) < 3:
                raise ValueError(f"{token}/{cam}: fewer than three camera records")
        return [sensors[cam][idx] for cam in CAMERAS for idx in (1, 2)] + [
            sensors[cam][0] for cam in CAMERAS]

    def asset_paths(self, record):
        # Metadata prefixes differ across machines. Resolve the samples/sweeps
        # component, not a global substring replacement on the data root.
        parts = Path(record["data_path"]).parts
        matches = [i for i, part in enumerate(parts) if part in {"samples", "sweeps"}]
        if len(matches) != 1:
            raise ValueError(f"Unexpected RGB path: {record['data_path']}")
        idx = matches[0]
        folder, tail = parts[idx], Path(*parts[idx + 1:])
        stem = tail.with_suffix("")
        return {
            "image": self.root / f"{folder}_small" / tail,
            "intrinsics": self.root / f"{folder}_param_small" / tail.with_suffix(".json"),
            "depth": self.root / f"{folder}_dptm_small" / f"{stem}_dpt.npy",
            "rel_depth": self.root / f"{folder}_dpt_small" / tail.with_suffix(".npy"),
            "mask": self.root / f"{folder}_mask_small" / tail.with_suffix(".png"),
        }

    def _load_view(self, record, novel):
        paths = self.asset_paths(record)
        with paths["intrinsics"].open() as handle:
            intrinsics = np.asarray(json.load(handle)["camera_intrinsic"], dtype=np.float64)
        with Image.open(paths["image"]) as source:
            image = source.convert("RGB")
            width, height = image.size
            # Match depthsplat PIL resize default (RGB bicubic).
            if (height, width) != self.size:
                image = image.resize((self.size[1], self.size[0]), Image.Resampling.BICUBIC)
            rgb = np.array(image, copy=True)
        intrinsics[0] *= self.size[1] / width
        intrinsics[1] *= self.size[0] / height
        c2w = np.asarray(record["sensor2lidar_transform"], dtype=np.float32)
        if intrinsics.shape != (3, 3) or c2w.shape != (4, 4):
            raise ValueError("Invalid camera matrix shape")
        depth = resize_depth(np.load(paths["depth"], allow_pickle=False), self.size)
        mask = np.ones(self.size, dtype=bool)
        if novel:
            with Image.open(paths["mask"]) as source:
                source = source.convert("L")
                if source.size != (self.size[1], self.size[0]):
                    source = source.resize((self.size[1], self.size[0]), Image.Resampling.BILINEAR)
                mask = (np.asarray(source, dtype=np.float32) / 255).astype(bool)
        # A novel view can be fully masked. Keep it in the 18-view sample;
        # supervision excludes its pixels while the six input targets stay valid.
        out = {"image": torch.from_numpy(rgb).permute(2, 0, 1).float() / 255,
               "intrinsics": torch.tensor(intrinsics, dtype=torch.float32),
               "extrinsics": torch.from_numpy(c2w.copy()), "depth": torch.from_numpy(depth),
               "masks": torch.from_numpy(mask.copy())}
        if self.load_rel_depth:
            disp = resize_depth(np.load(paths["rel_depth"], allow_pickle=False), self.size)
            out["rel_depth"] = torch.from_numpy(da2_to_relative_depth(disp))
        return out

    def __getitem__(self, index):
        token = self.bin_tokens[index]
        try:
            views = [self._load_view(record, i < 12) for i, record in enumerate(self.view_records(token))]
        except Exception as exc:
            raise RuntimeError(f"OmniScene {token}: {exc}") from exc
        target = {key: torch.stack([view[key] for view in views]) for key in views[0]}
        target["camera_ids"] = torch.tensor(CAMERA_IDS, dtype=torch.long)
        context = {key: target[key][12:] for key in ("image", "intrinsics", "extrinsics")}
        return {"context": context, "target": target, "scene": token}

    def check_assets(self, limit=0):
        """Read-only coverage check; never generate assets or touch forbidden files."""
        errors = []
        tokens = self.bin_tokens[:limit] if limit else self.bin_tokens
        for token in tokens:
            try:
                for index, record in enumerate(self.view_records(token)):
                    paths = self.asset_paths(record)
                    required = ["image", "intrinsics", "depth"]
                    if index < 12:
                        required.append("mask")
                    if self.load_rel_depth:
                        required.append("rel_depth")
                    for key in required:
                        if not paths[key].is_file():
                            errors.append({"bin": token, "view": index, "kind": key, "path": str(paths[key])})
            except Exception as exc:
                errors.append({"bin": token, "error": str(exc)})
        return {"split": self.split, "checked_bins": len(tokens), "expected_bins": len(self),
                "complete_scan": len(tokens) == len(self), "errors": errors}
