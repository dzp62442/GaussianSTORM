"""Explicit one-bin CUDA check; never run by unittest discovery.

python -m tests.check_omniscene_cuda --config configs/experiment/omniscene_112x200.yaml \
    --mode test --load_from work_dirs/.../checkpoints/ckpt_final.pth --gpu_memory_limit_gb 8
"""

import json

import torch

from main_storm import get_args_parser
from storm.dataset.omniscene_adapter import prepare_omniscene_batch
from storm.dataset.omniscene_dataset import OmniSceneDataset
from storm.omniscene_config import parse_args
from storm.omniscene_runner import _device, build_model


def main():
    args = parse_args(get_args_parser())
    device = _device(args)
    torch.set_num_threads(2)
    model = build_model(args).to(device).eval()
    if args.load_from:
        payload = torch.load(args.load_from, map_location="cpu")
        model.load_state_dict(payload["model"])
        del payload
    dataset = OmniSceneDataset(args, "total")
    sample = torch.utils.data.default_collate([dataset[0]])
    inputs, cameras, _, _ = prepare_omniscene_batch(sample, device)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.precision == "bf16"):
        gs = model.reconstruct_static(inputs)
        expected = model.render_static(gs, cameras, depth_mode="expected_z", chunk_size=6)
        accumulated = model.render_static(gs, cameras, depth_mode="accumulated_z", chunk_size=6)
        for key in ("rendered_image", "rendered_depth", "accumulated_depth", "rendered_alpha"):
            # BF16 sky linear kernels may select different GEMMs for different view batches.
            torch.testing.assert_close(expected[key], accumulated[key], rtol=2e-2 if key == "rendered_image" else 1e-5,
                                       atol=1e-2 if key == "rendered_image" else 2e-5)
        chunked = model.render_static(gs, cameras, depth_mode="accumulated_z", chunk_size=1)
        chunk_differences = {key: {"max_abs": float((accumulated[key] - chunked[key]).abs().max()),
                                 "mean_abs": float((accumulated[key] - chunked[key]).abs().mean()),
                                 "changed": int(((accumulated[key] - chunked[key]).abs() > 2e-5).sum())}
                             for key in ("rendered_image", "rendered_depth", "accumulated_depth", "rendered_alpha")}
        for key in ("rendered_depth", "accumulated_depth", "rendered_alpha"):
            torch.testing.assert_close(accumulated[key], chunked[key], rtol=1e-5, atol=2e-5)
        full_inverse = torch.linalg.inv(cameras["target_camtoworlds"][:, 0].float())
        inverse_by_chunk = torch.cat([torch.linalg.inv(cameras["target_camtoworlds"][:, 0, i:i+1].float())
                                      for i in range(18)], dim=1)
        original_means = gs["means"].clone()
        gs["forward_flow"] += 10000
        static = model.render_static(gs, cameras, chunk_size=6)
        torch.testing.assert_close(original_means, gs["means"], rtol=0, atol=0)
        torch.testing.assert_close(expected["rendered_image"], static["rendered_image"], rtol=0, atol=0)
        # The native renderer expects target v == physical camera count. Compare
        # one novel target per camera, preserving native camera/affine order.
        indices = [0, 2, 4, 6, 8, 10]
        subset = {key: (value[:, :, indices] if key in {
            "target_intrinsics", "target_camtoworlds", "target_camera_ids"} else value)
            for key, value in cameras.items()}
        modern = model.render_static(gs, subset, chunk_size=6)
        model.static_scene = False
        native = model({**inputs, **subset, "context_image": inputs["context_image"] * 2 - 1})["render_results"]
        for key in ("rendered_image", "rendered_depth", "rendered_alpha"):
            torch.testing.assert_close(modern[key], native[key], rtol=1e-5, atol=2e-5)
    print(json.dumps({"resolution": args.input_size, "bin": sample["scene"][0],
                      "ed_d": "passed", "chunk_differences": chunk_differences,
                      "inverse_chunk_max_diff": float((full_inverse-inverse_by_chunk).abs().max()),
                      "velocity_invariance": "passed",
                      "native_static_equivalence": "passed",
                      "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024 ** 3}, indent=2))


if __name__ == "__main__":
    main()
