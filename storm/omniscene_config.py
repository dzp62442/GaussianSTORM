"""Reviewed OmniScene configuration; legacy argparse defaults remain unchanged."""

import argparse
import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CAMERAS = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", "CAM_BACK",
           "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
EXTRA_DEFAULTS = {
    "mode": "train", "test_split": "total", "dry_run": False, "check_limit": 0,
    "max_eval_bins": 0, "gpu_memory_limit_gb": None,
    "data_version": "interp_12Hz_trainval", "train_manifest": "bins_train_3.2m.json",
    "eval_manifest": "bins_val_3.2m.json", "camera_order": CAMERAS,
    "context_indices": [0], "novel_indices": [1, 2], "append_context_to_targets": True,
    "static_scene": True, "use_dynamic_mask": True, "dynamic_mask_scope": "novel_12",
    "load_rel_depth_train": False, "load_rel_depth_eval": True,
    "relative_depth_source": "da2", "val_selection": {"stop": 30000, "stride": 3000, "limit": 10},
    "mini_selection": {"offset": 0, "stride": 14, "limit": 2048}, "test_selection": "all",
    "num_cams": 6, "near": 0.2, "far": 400.0, "scale_offset": -2.3,
    "opacity_offset": -2.0, "max_scale": 0.5, "tau": 0.5, "projected_motion_dim": 32,
    "disable_pos_embed": False, "sigmoid_rgb": False, "test_batch_size": 1,
    "val_every_n_iters": 1000, "test_every_n_validations": 10,
    "test_at_training_end": True, "save_final_checkpoint": True,
    "eval_view_groups": ["all_18", "novel_12"], "pcc_depth_mode": "accumulated_z",
    "train_depth_mode": "expected_z", "eval_radius_clip": 0.0,
    "eval_render_chunk_size": 6, "timing_warmup_samples": 5,
    "precision": "bf16", "wandb_mode": "offline",
    "enable_feishu": True,
    "feishu_module_paths": ["~/Libraries", "/vepfs-mlp2/c20250502/haoce/dzp"],
    "lpips_weights": str(Path.home() / ".cache/torch/hub/checkpoints/vgg.pth"),
}


def load_config(path, stack=()):
    path = Path(path).resolve()
    if path in stack:
        raise ValueError(f"Circular config inheritance: {stack + (path,)}")
    with path.open() as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a mapping in {path}")
    result = {}
    # Dataset first; the experiment (including its parent) overrides dataset defaults.
    for key in ("dataset_config", "extends"):
        ref = raw.pop(key, None)
        if ref is not None:
            result.update(load_config(path.parent / ref, stack + (path,)))
    result.update(raw)
    return result


def parse_args(parser, argv=None):
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--config")
    config_path = probe.parse_known_args(argv)[0].config
    parser.add_argument("--config")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("-h", "--help", action="help")
    for action in list(parser._actions):
        if isinstance(action, argparse._StoreTrueAction):
            parser.add_argument(f"--no-{action.dest}", dest=action.dest, action="store_false",
                                default=argparse.SUPPRESS)
    for key, default in EXTRA_DEFAULTS.items():
        if isinstance(default, bool):
            parser.add_argument(f"--{key}", action=argparse.BooleanOptionalAction, default=default)
        else:
            kind = yaml.safe_load if isinstance(default, (list, dict)) else type(default)
            if default is None:
                kind = float
            parser.add_argument(f"--{key}", default=copy.deepcopy(default), type=kind)
    allowed = {a.dest for a in parser._actions}
    if config_path:
        values = load_config(config_path)
        unknown = values.keys() - allowed
        if unknown:
            parser.error(f"Unknown config keys: {sorted(unknown)}")
        parser.set_defaults(**values)
    args = parser.parse_args(argv)
    for item in args.overrides:
        key, sep, value = item.partition("=")
        if not sep or key not in allowed or key in {"config", "overrides"}:
            parser.error(f"Invalid override: {item}")
        setattr(args, key, yaml.safe_load(value))
    if args.dataset == "omniscene":
        validate_config(args)
        for key in ("data_root", "output_dir", "load_from", "resume_from", "lpips_weights"):
            value = getattr(args, key)
            if value:
                setattr(args, key, str((ROOT / Path(value).expanduser()).resolve()))
    return args


def validate_config(args):
    fixed = {
        "model": "STORM-B/8", "decoder_type": "dummy", "gs_dim": 3,
        "camera_order": CAMERAS, "context_indices": [0], "novel_indices": [1, 2],
        "append_context_to_targets": True, "num_cams": 6, "num_max_cameras": 6,
        "num_context_timesteps": 1, "num_target_timesteps": 1,
        "static_scene": True, "load_flow": False, "load_ground": False,
        "skip_sky_mask": True, "load_depth": True, "use_dynamic_mask": True,
        "dynamic_mask_scope": "novel_12", "load_rel_depth_train": False,
        "load_rel_depth_eval": True, "relative_depth_source": "da2",
        "test_selection": "all", "subset_ratio": 1.0, "batch_size": 1,
        "eval_batch_size": 1, "test_batch_size": 1, "use_sky_token": True,
        "use_affine_token": True, "num_motion_tokens": 16, "use_latest_gsplat": False,
        "enable_sky_depth_loss": False, "enable_sky_opacity_loss": False,
        "enable_depth_loss": True, "enable_flow_reg_loss": True,
        "pcc_depth_mode": "accumulated_z", "train_depth_mode": "expected_z",
        "eval_view_groups": ["all_18", "novel_12"], "wandb_mode": "offline",
        "save_final_checkpoint": True, "test_at_training_end": True,
    }
    for key, expected in fixed.items():
        if getattr(args, key) != expected:
            raise ValueError(f"OmniScene protocol requires {key}={expected!r}")
    if list(args.input_size) not in ([112, 200], [224, 400]):
        raise ValueError("OmniScene input_size must be [112,200] or [224,400]")
    if args.mode not in {"train", "test", "check-data"} or args.test_split not in {"mini", "total"}:
        raise ValueError("Invalid mode or test_split")
    for key in ("num_iterations", "val_every_n_iters", "test_every_n_validations",
                "ckpt_every_n_iters", "log_every_n_iters", "eval_render_chunk_size"):
        value = getattr(args, key)
        if type(value) is not int or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("num_workers", "timing_warmup_samples", "check_limit", "max_eval_bins",
                "vis_every_n_iters", "num_vis_samples", "keep_n_ckpts"):
        if type(getattr(args, key)) is not int or getattr(args, key) < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    for key in ("val_selection", "mini_selection"):
        selection = getattr(args, key)
        if not isinstance(selection, dict) or selection.keys() - {"offset", "stop", "stride", "limit"}:
            raise ValueError(f"Invalid {key}")
        for name, value in selection.items():
            if type(value) is not int or value < (1 if name in {"stride", "limit", "stop"} else 0):
                raise ValueError(f"Invalid {key}.{name}")
    if args.timespan <= 0 or args.near <= 0 or args.far <= args.near:
        raise ValueError("Invalid timespan or depth bounds")
    if args.lr is None or args.lr <= 0 or args.lr_sched != "cosine":
        raise ValueError("Use an absolute positive lr and cosine schedule")
    if args.precision not in {"bf16", "fp32"}:
        raise ValueError("precision must be bf16 or fp32")
    if type(args.enable_feishu) is not bool:
        raise ValueError("enable_feishu must be a boolean")
    if (not isinstance(args.feishu_module_paths, list)
            or any(not isinstance(path, str) or not path.strip() for path in args.feishu_module_paths)):
        raise ValueError("feishu_module_paths must be a list of nonempty paths")
    if type(args.auto_resume) is not bool:
        raise ValueError("auto_resume must be a boolean")
    # Automatic training resume must not conflict with standalone test weights.
    if args.load_from and args.resume_from:
        raise ValueError("load_from and training resume are mutually exclusive")
    if args.resume_from and args.mode != "train":
        raise ValueError("resume_from is only supported in train mode")
    if args.mode == "train" and args.load_from:
        raise ValueError("OmniScene trains from scratch; use resume_from to resume")
    if args.start_iteration:
        raise ValueError("Resume the complete checkpoint; do not set start_iteration manually")
    if args.evaluate or args.visualization_only:
        raise ValueError("Use --mode test for OmniScene, not legacy evaluation flags")
