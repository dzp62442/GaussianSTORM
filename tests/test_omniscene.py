"""CPU protocol tests; no CUDA initialization, downloads or external datasets."""

import contextlib
import io
import json
from pathlib import Path
import pickle
import random
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
import torch

from main_storm import get_args_parser
from storm.dataset.omniscene_adapter import prepare_omniscene_batch
from storm.dataset.omniscene_dataset import CAMERA_IDS, OmniSceneDataset, da2_to_relative_depth
from storm.evaluation.omniscene import aggregate_records, compute_pcc, evaluate_omniscene, group_records
from storm.omniscene_config import CAMERAS, ROOT, load_config, parse_args
from storm.omniscene_runner import TrainingState, select_resume
from storm.utils.experiment import summarize_times
from storm.utils.losses import compute_depth_loss, compute_loss
from storm.utils.lpips_loss import RGBLpipsLoss

torch.set_num_threads(2)


def config(*overrides):
    return parse_args(get_args_parser(), ["--config", str(ROOT / "configs/experiment/omniscene_112x200.yaml"), *overrides])


class ConfigurationTests(unittest.TestCase):
    def test_precedence_and_resolutions(self):
        low = config("--lr", "0.0003", "--no-enable_wandb", "--num_workers", "0")
        high = parse_args(get_args_parser(), ["--config", str(ROOT / "configs/experiment/omniscene_224x400.yaml")])
        self.assertEqual(low.lr, 0.0003)
        self.assertFalse(low.enable_wandb)
        self.assertEqual(low.num_workers, 0)
        self.assertEqual(high.input_size, [224, 400])
        self.assertEqual(high.num_iterations, 100001)
        self.assertEqual(high.num_cams, 6)
        self.assertTrue(low.auto_resume)
        self.assertTrue(high.auto_resume)
        self.assertFalse(config("--no-auto_resume").auto_resume)

    def test_auto_resume_does_not_block_standalone_test(self):
        args = config("--mode", "test", "--load_from", "checkpoint.pth")
        self.assertTrue(args.auto_resume)
        self.assertEqual(Path(args.load_from).name, "checkpoint.pth")
        self.assertIsNone(select_resume(args, ROOT / "work_dirs"))
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            config("--mode", "test", "--load_from", "checkpoint.pth", "--resume_from", "checkpoint.pth")
        with self.assertRaisesRegex(ValueError, "only supported in train"):
            config("--mode", "test", "--resume_from", "checkpoint.pth")
        with self.assertRaisesRegex(ValueError, "boolean"):
            config("--set", 'auto_resume="false"')

    def test_auto_resume_selects_only_current_experiment_and_complete_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            current = root / "low" / "checkpoints"
            other = root / "high" / "checkpoints"
            current.mkdir(parents=True)
            other.mkdir(parents=True)
            (other / "ckpt_final.pth").touch()
            (current / "ckpt_step_020000.pth.tmp").touch()
            (current / "ckpt_final.pth.tmp").touch()
            args = config()
            self.assertIsNone(select_resume(args, current))
            older = current / "ckpt_step_005000.pth"
            latest = current / "ckpt_step_010000.pth"
            for path in (latest, older):
                path.touch()
            self.assertEqual(select_resume(args, current), latest)
            args.auto_resume = False
            self.assertIsNone(select_resume(args, current))
            args.auto_resume = True
            final = current / "ckpt_final.pth"
            final.touch()
            self.assertEqual(select_resume(args, current), final)
            args.resume_from = str(older)
            self.assertEqual(select_resume(args, current), older)

    def test_reject_protocol_changes(self):
        for override in ["load_rel_depth_train=true", "static_scene=false", "num_cams=3", "batch_size=2",
                         "dynamic_mask_scope=all_18", "use_latest_gsplat=true"]:
            with self.assertRaises(ValueError):
                config("--set", override)
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            config("--set", "typo=true")

    def test_cycle_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cycle.yaml"
            path.write_text("extends: cycle.yaml\n")
            with self.assertRaises(ValueError):
                load_config(path)

    def test_step_schedule_and_final_recovery(self):
        args, state, validations, tests = config(), TrainingState(), [], []
        for step in range(1, args.num_iterations + 1):
            state.completed_steps = step
            val, mini = state.due(args)
            if val:
                validations.append(step)
                state.last_validation_step = step
                state.validation_count += 1
            if mini:
                tests.append(step)
                state.tested_steps.append(step)
                state.final_test_complete = step == args.num_iterations
        self.assertEqual(validations, list(range(1000, 100001, 1000)))
        self.assertEqual(tests, list(range(10000, 100001, 10000)) + [100001])
        self.assertEqual(state.due(args), (False, False))
        interrupted = TrainingState(completed_steps=100001, validation_count=100, last_validation_step=100000)
        self.assertEqual(interrupted.due(args), (False, True))
        checkpoint_before_eval = TrainingState(completed_steps=10000, validation_count=9, last_validation_step=9000)
        self.assertEqual(checkpoint_before_eval.due(args), (True, True))

    def test_sampler_cursor_replays_consumed_not_prefetched(self):
        from storm.dataset.samplers import InfiniteSampler
        from itertools import islice
        original = list(islice(InfiniteSampler(13, shuffle=True, seed=9, start=0, step=1), 30))
        resumed = list(islice(InfiniteSampler(13, shuffle=True, seed=9, start=0, step=1, advance=17), 13))
        self.assertEqual(resumed, original[17:])


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.args = config("--data_root", str(self.root), "--num_workers", "0")
        folder = self.root / self.args.data_version
        (folder / "bin_infos_3.2m").mkdir(parents=True)
        for split in ("train", "val"):
            (folder / f"bins_{split}_3.2m.json").write_text(json.dumps({"bins": ["fixture"]}))
        info = {"sensor_info": {}}
        height, width = 224, 400
        for camera_index, camera in enumerate(CAMERAS):
            records = []
            for index in range(3):
                stem = f"{camera}/{index}"
                pose = np.eye(4, dtype=np.float32)
                pose[0, 3] = camera_index + index / 10
                records.append({"data_path": f"/old/root/samples/{stem}.jpg", "timestamp": 123 + index,
                                "sensor2lidar_transform": pose})
                paths = {"image": self.root / "samples_small" / f"{stem}.jpg",
                         "param": self.root / "samples_param_small" / f"{stem}.json",
                         "metric": self.root / "samples_dptm_small" / f"{stem}_dpt.npy",
                         "relative": self.root / "samples_dpt_small" / f"{stem}.npy",
                         "mask": self.root / "samples_mask_small" / f"{stem}.png"}
                for path in paths.values():
                    path.parent.mkdir(parents=True, exist_ok=True)
                rgb = np.full((height, width, 3), 20 + 10 * camera_index + index, dtype=np.uint8)
                Image.fromarray(rgb).save(paths["image"])
                paths["param"].write_text(json.dumps({"camera_intrinsic": [[300, 0, 200], [0, 320, 112], [0, 0, 1]]}))
                np.save(paths["metric"], np.full((height, width), 12 + index, dtype=np.float16))
                np.save(paths["relative"], np.broadcast_to(np.linspace(1, 30, width), (height, width)).astype(np.float16))
                if index:  # No input-view mask, confidence or LiDAR assets exist.
                    mask = np.full((height, width), 255, dtype=np.uint8)
                    mask[:, :40] = 0
                    Image.fromarray(mask).save(paths["mask"])
            info["sensor_info"][camera] = records
        self.bin_path = folder / "bin_infos_3.2m/fixture.pkl"
        self.bin_path.write_bytes(pickle.dumps(info))

    def tearDown(self):
        self.temp.cleanup()

    def test_views_units_intrinsics_and_no_extra_assets(self):
        dataset = OmniSceneDataset(self.args, "train")
        example = dataset[0]
        self.assertEqual(example["context"]["image"].shape, (6, 3, 112, 200))
        self.assertEqual(example["target"]["image"].shape, (18, 3, 112, 200))
        self.assertEqual(example["target"]["camera_ids"].tolist(), CAMERA_IDS)
        self.assertTrue(torch.equal(example["target"]["image"][12:], example["context"]["image"]))
        self.assertTrue(example["target"]["masks"][12:].all())
        self.assertFalse(example["target"]["masks"][:12, :, 0].any())
        torch.testing.assert_close(example["context"]["intrinsics"][0], torch.tensor([[150., 0, 100], [0, 160, 56], [0, 0, 1]]))
        self.assertEqual(float(example["target"]["depth"][0, 0, 0]), 13.)
        self.assertNotIn("rel_depth", example["target"])
        self.assertFalse(dataset.check_assets()["errors"])
        # Time metadata cannot affect camera selection, poses or supervision.
        info = pickle.loads(self.bin_path.read_bytes())
        for records in info["sensor_info"].values():
            for record in records:
                record["timestamp"] *= 12345
        self.bin_path.write_bytes(pickle.dumps(info))
        again = dataset[0]
        for key in example["target"]:
            torch.testing.assert_close(again["target"][key], example["target"][key])

    def test_adapter_separates_truth_and_eval_depth(self):
        example = OmniSceneDataset(self.args, "total")[0]
        batch = torch.utils.data.default_collate([example])
        inputs, cameras, supervision, metrics = prepare_omniscene_batch(batch, "cpu")
        self.assertEqual(set(inputs), {"context_image", "context_intrinsics", "context_camtoworlds", "context_time"})
        self.assertEqual(cameras["target_camera_ids"].shape, (1, 1, 18))
        self.assertEqual(supervision["target_depth"].shape, (1, 1, 18, 112, 200))
        self.assertEqual(metrics["rel_depth"].shape, (1, 18, 112, 200))
        self.assertEqual(float(metrics["rel_depth"].min()), 0)
        self.assertEqual(float(metrics["rel_depth"].max()), 1)
        self.assertEqual(float(inputs["context_time"].sum()), 0)

    def test_fully_masked_novel_views_are_kept_but_missing_files_still_fail(self):
        for camera in CAMERAS:
            for index in (1, 2):
                path = self.root / "samples_mask_small" / camera / f"{index}.png"
                Image.fromarray(np.zeros((224, 400), dtype=np.uint8)).save(path)
        dataset = OmniSceneDataset(self.args, "train")
        example = dataset[0]
        self.assertEqual(example["target"]["image"].shape[0], 18)
        self.assertFalse(example["target"]["masks"][:12].any())
        self.assertTrue(example["target"]["masks"][12:].all())
        self.assertTrue(example["target"]["image"][:12].any())
        self.assertFalse(dataset.check_assets()["errors"])
        # Do not silently replace absent masks with an all-valid or all-invalid mask.
        (self.root / "samples_mask_small" / CAMERAS[0] / "1.png").unlink()
        with self.assertRaises(RuntimeError) as raised:
            dataset[0]
        self.assertIsInstance(raised.exception.__cause__, FileNotFoundError)

    def test_depth_values_and_duplicate_manifest_do_not_reject_samples(self):
        for camera in CAMERAS:
            for index in (0, 1, 2):
                np.save(self.root / "samples_dptm_small" / camera / f"{index}_dpt.npy",
                        np.zeros((224, 400), dtype=np.float32))
                np.save(self.root / "samples_dpt_small" / camera / f"{index}.npy",
                        np.ones((224, 400), dtype=np.float32))
        manifest = self.root / self.args.data_version / self.args.eval_manifest
        manifest.write_text(json.dumps({"bins": ["fixture", "fixture"]}))
        # Preserve the supplied geometry, without a positive-focal/determinant threshold.
        info = pickle.loads(self.bin_path.read_bytes())
        info["sensor_info"][CAMERAS[0]][1]["sensor2lidar_transform"][2, 2] = 1e-8
        self.bin_path.write_bytes(pickle.dumps(info))
        param = self.root / "samples_param_small" / CAMERAS[0] / "1.json"
        values = json.loads(param.read_text())
        values["camera_intrinsic"][0][0] = -300
        param.write_text(json.dumps(values))
        dataset = OmniSceneDataset(self.args, "total")
        self.assertEqual(dataset.bin_tokens, ["fixture", "fixture"])
        example = dataset[0]
        self.assertEqual(float(example["target"]["depth"].sum()), 0)
        self.assertTrue(torch.isnan(example["target"]["rel_depth"]).all())
        self.assertLess(float(example["target"]["intrinsics"][0, 0, 0]), 0)


class LossAndMetricTests(unittest.TestCase):
    def test_perceptual_checkpoint_preserves_value_and_gradient(self):
        class Perceptual(torch.nn.Module):
            def forward(self, prediction, target):
                return (prediction.sin() - target.cos()).square().mean((1, 2, 3))
        plain = RGBLpipsLoss(enable_perceptual_loss=False)
        chunked = RGBLpipsLoss(enable_perceptual_loss=False, perceptual_chunk_size=1, checkpoint_perceptual=True)
        for module in (plain, chunked):
            module.perceptual_loss = Perceptual()
            module.set_perceptual_loss(True)
        gt = torch.rand(5, 8, 8, 3)
        first = torch.rand_like(gt).requires_grad_()
        second = first.detach().clone().requires_grad_()
        a, b = plain(first, gt)["perceptual_loss"], chunked(second, gt)["perceptual_loss"]
        a.backward()
        b.backward()
        torch.testing.assert_close(a, b)
        torch.testing.assert_close(first.grad, second.grad)

    def test_masked_losses_and_gradients(self):
        args = config()
        args.enable_perceptual_loss = False
        pred = torch.rand(1, 1, 18, 4, 5, 3, requires_grad=True)
        depth = torch.rand(1, 1, 18, 4, 5, requires_grad=True)
        mask = torch.ones(1, 1, 18, 4, 5, dtype=torch.bool)
        mask[:, :, :12, :, :2] = False
        mask[:, :, 0] = False
        target = {"target_image": torch.rand(1, 1, 18, 3, 4, 5),
                  "target_depth": torch.rand_like(depth) + 1, "target_valid_mask": mask}
        output = {"gs_params": {"forward_flow": torch.ones(1, 1, 6, 4, 5, 3)},
                  "render_results": {"rgb_key": "rgb", "depth_key": "depth", "flow_key": None,
                                     "decoder_depth_key": None, "rgb": pred, "depth": depth}}
        loss_fn = RGBLpipsLoss(enable_perceptual_loss=False)
        before = compute_loss(output, target, args, loss_fn)
        fallback = compute_loss(output, target, args, None)
        torch.testing.assert_close(before["rgb_loss"], fallback["rgb_loss"])
        sum(before.values()).backward()
        self.assertEqual(float(pred.grad[~mask].abs().sum()), 0)
        self.assertEqual(float(depth.grad[~mask].abs().sum()), 0)
        self.assertAlmostEqual(float(before["flow_reg_loss"]), 0.005)
        with torch.no_grad():
            pred[~mask] = 100
            depth[~mask] = 100
        after = compute_loss(output, target, args, loss_fn)
        for key in before:
            torch.testing.assert_close(before[key], after[key])

    def test_all_novel_views_masked_lpips_remains_finite_and_has_zero_masked_gradient(self):
        class SquaredDistance(torch.nn.Module):
            def forward(self, pred, gt):
                return (pred - gt).square().mean((1, 2, 3))

        loss_fn = RGBLpipsLoss(enable_perceptual_loss=False, perceptual_weight=.05,
                               perceptual_chunk_size=1, checkpoint_perceptual=True)
        loss_fn.perceptual_loss = SquaredDistance()
        loss_fn.set_perceptual_loss(True)
        rgb = torch.rand(18, 4, 5, 3, requires_grad=True)
        target = torch.rand_like(rgb)
        original_target = target.clone()
        mask = torch.ones(18, 4, 5, dtype=torch.bool)
        mask[:12] = False
        losses = loss_fn(rgb, target, mask)
        expected_rgb = (rgb[12:] - target[12:]).square().mean()
        torch.testing.assert_close(losses["rgb_loss"], expected_rgb)
        # LPIPS keeps the original mean over all 18 views; masked views contribute zero.
        torch.testing.assert_close(losses["perceptual_loss"], .05 * expected_rgb * (6 / 18))
        sum(losses.values()).backward()
        self.assertTrue(torch.isfinite(rgb.grad).all())
        self.assertEqual(float(rgb.grad[:12].abs().sum()), 0)
        self.assertGreater(float(rgb.grad[12:].abs().sum()), 0)
        torch.testing.assert_close(target, original_target)
        with torch.no_grad():
            rgb[:12] = 100
        again = loss_fn(rgb, target, mask)
        for key in losses:
            torch.testing.assert_close(losses[key], again[key])
        empty = loss_fn(rgb, target, torch.zeros_like(mask))
        for value in empty.values():
            self.assertEqual(float(value), 0)

    def test_native_all_valid_and_depth_guards(self):
        rgb, gt = torch.rand(2, 8, 8, 3), torch.rand(2, 8, 8, 3)
        loss_fn = RGBLpipsLoss(enable_perceptual_loss=False)
        torch.testing.assert_close(loss_fn(rgb, gt)["rgb_loss"], loss_fn(rgb, gt, torch.ones(2, 8, 8, dtype=torch.bool))["rgb_loss"])
        pred, depth = torch.rand(2, 8, 8), torch.rand(2, 8, 8) + 1
        torch.testing.assert_close(compute_depth_loss(pred, depth), (pred / depth.max() - depth / depth.max()).abs().mean())
        for truth in (torch.zeros_like(depth), torch.full_like(depth, float("nan")), depth):
            predicted = pred.detach().clone().requires_grad_()
            empty = compute_depth_loss(predicted, truth, valid_region=torch.zeros_like(truth, dtype=torch.bool))
            self.assertEqual(float(empty), 0)
            empty.backward()
            self.assertEqual(float(predicted.grad.abs().sum()), 0)

    def test_grouped_pearson_is_flattened_per_bin(self):
        torch.manual_seed(2)
        reference = torch.rand(18, 8, 9)
        prediction = reference * torch.arange(1, 19)[:, None, None] + torch.randn_like(reference) * .1
        image_metrics = {key: torch.arange(18).float() for key in ("psnr", "ssim", "lpips")}
        rows = group_records("bin", image_metrics, reference, prediction)
        for row, n in zip(rows, (18, 12)):
            expected = np.corrcoef(reference[:n].numpy().ravel(), prediction[:n].numpy().ravel())[0, 1]
            self.assertAlmostEqual(row["pcc"], expected, places=7)
        self.assertNotAlmostEqual(rows[0]["pcc"], rows[1]["pcc"], places=3)
        summary = aggregate_records(rows, ["bin"])
        self.assertEqual(summary["all_18"]["psnr"], 8.5)
        with self.assertRaises(ValueError):
            aggregate_records(rows + rows, ["bin"])
        self.assertTrue(torch.isnan(compute_pcc(reference, torch.ones_like(reference))))
        self.assertTrue(np.isnan(da2_to_relative_depth(np.ones((8, 9), dtype=np.float32))).all())

    def test_undefined_metrics_preserve_bins_and_do_not_average_a_subset(self):
        ref = torch.arange(18 * 4 * 5).float().reshape(18, 4, 5)
        metrics = {key: torch.ones(18) for key in ("psnr", "ssim", "lpips")}
        first = group_records("a", metrics, ref, torch.ones_like(ref))
        second = group_records("b", {**metrics, "psnr": torch.full((18,), float("inf"))}, ref, ref)
        self.assertIsNone(first[0]["pcc"])
        self.assertIn("pcc", first[0]["metric_notes"])
        self.assertEqual(second[0]["psnr"], "Infinity")
        summary = aggregate_records(first + second, ["a", "b"])
        self.assertIsNone(summary["all_18"]["pcc"])
        self.assertEqual(summary["novel_12"]["ssim"], 1.)
        self.assertEqual(summary["all_18"]["psnr"], "Infinity")
        json.dumps(summary, allow_nan=False)
        # Repeated samples explicitly present in the manifest retain their multiplicity.
        self.assertEqual(aggregate_records(second + second, ["b", "b"])["all_18"]["pcc"], 1.)

    def test_evaluation_finishes_and_writes_undefined_metric_diagnostics(self):
        args = config("--num_workers", "0", "--set", "timing_warmup_samples=0")
        shape = (18, 4, 5)
        reference = torch.arange(np.prod(shape)).float().reshape(shape)

        class Dataset(torch.utils.data.Dataset):
            split = "mini"
            bin_tokens = ["constant", "defined"]

            def __len__(self):
                return 2

            def __getitem__(self, index):
                return {"scene": self.bin_tokens[index], "index": index}

            def metadata(self):
                return dict(split=self.split, bins=self.bin_tokens, expected_count=2,
                            uncapped_count=2, limited=False)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(()))

            def reconstruct_static(self, inputs):
                return {"means": torch.zeros(1, 5, 3)}

            def render_static(self, gaussians, cameras, **kwargs):
                return {"rendered_image": torch.zeros(1, 1, *shape, 3),
                        "accumulated_depth": reference[None, None]}

        def prepare(batch, *unused):
            ref = torch.full_like(reference, float("nan")) if int(batch["index"][0]) == 0 else reference
            return ({"context_image": torch.zeros(1, 1, 6, 3, 4, 5)}, {}, {},
                    {"rgb": torch.zeros(1, 18, 3, 4, 5), "rel_depth": ref[None]})

        metric = lambda *unused: {key: torch.ones(18) for key in ("psnr", "ssim", "lpips")}
        with tempfile.TemporaryDirectory() as directory, \
                patch("storm.evaluation.omniscene.prepare_omniscene_batch", side_effect=prepare), \
                patch("storm.evaluation.omniscene.dependency_versions", return_value={}):
            model = Model().train()
            summary = evaluate_omniscene(model, Dataset(), args, directory, {}, metric)
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["completed_count"], 2)
            self.assertFalse(summary["metrics_defined"])
            self.assertEqual(summary["undefined_metric_counts"]["all_18"]["pcc"], 1)
            self.assertIsNone(summary["groups"]["all_18"]["pcc"])
            issues = json.loads((Path(directory) / "metric_issues.json").read_text())
            self.assertEqual(len(issues), 2)
            self.assertEqual(issues[0]["bin"], "constant")
            self.assertEqual(len((Path(directory) / "records.jsonl").read_text().splitlines()), 4)
            self.assertTrue(model.training)
            saved = json.loads((Path(directory) / "summary.json").read_text())
            self.assertEqual(saved["groups"], summary["groups"])

    def test_timer_warmup_does_not_remove_quality_rows(self):
        times = [{"bin": str(i), "milliseconds": float(i)} for i in range(8)]
        summary = summarize_times(times, 5)
        self.assertEqual(summary["count"], 3)
        self.assertEqual(summary["mean"], 6)
        self.assertEqual(len(summary["records"]), 8)


class StaticModelTests(unittest.TestCase):
    def test_static_render_affine_depth_and_velocity_independence(self):
        from storm.models.storm import STORM
        model = STORM(img_size=(16, 24), patch_size=8, embed_dim=32, depth=1, num_heads=4,
                      num_cams=6, num_motion_tokens=16, static_scene=True, grad_checkpointing=False).eval()
        pose = torch.eye(4).repeat(1, 1, 18, 1, 1)
        k = torch.tensor([[20., 0, 12], [0, 20, 8], [0, 0, 1]]).repeat(1, 1, 18, 1, 1)
        inputs = {"context_image": torch.rand(1, 1, 6, 3, 16, 24), "context_time": torch.zeros(1, 1),
                  "context_intrinsics": k[:, :, 12:], "context_camtoworlds": pose[:, :, 12:]}
        cameras = {"context_time": torch.zeros(1, 1), "target_time": torch.zeros(1, 1),
                   "target_intrinsics": k, "target_camtoworlds": pose,
                   "target_camera_ids": torch.tensor(CAMERA_IDS)[None, None], "height": 16, "width": 24}
        calls = []
        def raster(**kw):
            calls.append(kw["means"].clone())
            b, v = kw["viewmats"].shape[:2]
            alpha = torch.full((b, v, kw["height"], kw["width"], 1), .4)
            rgb = kw["colors"].mean(1)[:, None, None, None].expand(b, v, kw["height"], kw["width"], 3)
            depth = torch.full_like(alpha, 7.)
            if kw["render_mode"] == "RGB+D":
                depth = depth * alpha
            return torch.cat([rgb, depth], dim=-1), alpha, {}
        with torch.no_grad(), patch("storm.models.storm.rasterization", raster):
            gs = model.reconstruct_static(inputs)
            first = model.render_static(gs, cameras, chunk_size=5)
            gs["forward_flow"] += 1000
            second = model.render_static(gs, cameras, depth_mode="accumulated_z", chunk_size=6)
            for key in ("rendered_image", "rendered_depth", "accumulated_depth"):
                torch.testing.assert_close(first[key], second[key])
            for means in calls:
                torch.testing.assert_close(means, gs["means"].reshape(1, -1, 3))
            combined = model({**inputs, **cameras})
            torch.testing.assert_close(combined["render_results"]["rendered_image"], first["rendered_image"])
            with self.assertRaises(ValueError):
                model.reconstruct_static({**inputs, "target_depth": torch.ones(1)})
            with self.assertRaises(ValueError):
                model.render_static(gs, {**cameras, "target_time": torch.ones(1, 1)})


class TrainingRecoveryTests(unittest.TestCase):
    def test_interrupted_evaluation_resume_and_final_test_recovery(self):
        from storm.omniscene_runner import train

        class FixtureDataset(torch.utils.data.Dataset):
            def __init__(self, args, split):
                self.split = split
                self.manifest_sha256 = "fixture-manifest"
                self.bin_tokens = [str(i) for i in range(5)]

            def __len__(self):
                return 5

            def __getitem__(self, index):
                return {"feature": torch.tensor([float(index + 1)]), "scene": str(index)}

            def metadata(self):
                return {"split": self.split, "bins": self.bin_tokens}

        class ToyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(.3))

            def forward(self, inputs):
                noise = torch.rand_like(inputs["feature"]) + random.random() + np.random.random()
                return self.weight * inputs["feature"] * noise

        class NoPerceptual(torch.nn.Module):
            def __init__(self, **kwargs):
                super().__init__()

            def set_perceptual_loss(self, enabled):
                pass

        events = []
        fail_once = [True]

        def evaluate(model, dataset, args, output, info, metric, **kwargs):
            event = (dataset.split, info["completed_steps"])
            events.append(event)
            if event == ("mini", 2) and fail_once[0]:
                fail_once[0] = False
                raise RuntimeError("injected evaluation interruption")
            return {"split": dataset.split, "complete": True, "limited": False,
                    "completed_count": 5, "expected_count": 5, "uncapped_count": 5,
                    "groups": {group: dict(psnr=20., ssim=.8, lpips=.2, pcc=.7)
                               for group in ("all_18", "novel_12")}}

        with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
            sender = Mock(return_value=True)
            stack.enter_context(patch("storm.utils.feishu.load_sender", return_value=sender))
            stack.enter_context(patch("storm.omniscene_runner.OmniSceneDataset", FixtureDataset))
            stack.enter_context(patch("storm.omniscene_runner.ImageMetrics", return_value=object()))
            stack.enter_context(patch("storm.omniscene_runner.evaluate_omniscene", side_effect=evaluate))
            stack.enter_context(patch("storm.omniscene_runner.prepare_omniscene_batch",
                                      side_effect=lambda batch, *unused: ({"feature": batch["feature"]}, {}, {}, {})))
            stack.enter_context(patch("storm.utils.lpips_loss.RGBLpipsLoss", NoPerceptual))
            stack.enter_context(patch("storm.utils.losses.compute_loss",
                                      side_effect=lambda output, *unused: {"rgb_loss": (output - 2).square().mean()}))
            args = config("--num_iterations", "3", "--val_every_n_iters", "1", "--test_every_n_validations", "2",
                          "--ckpt_every_n_iters", "1", "--num_workers", "0", "--no-enable_wandb")
            root = Path(directory) / "resumed"
            checkpoints = root / "checkpoints"
            checkpoints.mkdir(parents=True)
            from storm.utils.misc import fix_random_seeds
            fix_random_seeds(args.seed)
            model = ToyModel()
            with self.assertRaisesRegex(RuntimeError, "injected"):
                train(model, args, torch.device("cpu"), root, checkpoints, {})
            # Failed mini tests and ordinary validation must not claim test completion.
            self.assertEqual(sender.call_count, 1)
            self.assertIn("训练启动", sender.call_args.args[0])
            resume_path = checkpoints / "ckpt_step_000002.pth"
            pending = torch.load(resume_path, map_location="cpu")
            self.assertEqual(pending["state"]["completed_steps"], 2)
            self.assertEqual(pending["state"]["validation_count"], 1)
            self.assertIsNone(args.resume_from)
            self.assertTrue(args.auto_resume)
            # Simulate a fresh process with unrelated RNG state; auto-resume must restore it.
            fix_random_seeds(99)
            resumed = ToyModel()
            result = train(resumed, args, torch.device("cpu"), root, checkpoints, {})
            self.assertEqual(result["completed_steps"], 3)
            self.assertEqual(result["validation_count"], 3)
            self.assertEqual(result["tested_steps"], [2, 3])
            self.assertTrue(result["final_test_complete"])
            self.assertEqual(events[-3:], [("mini", 2), ("val", 3), ("mini", 3)])
            self.assertEqual(sender.call_count, 4)
            self.assertIn("训练恢复", sender.call_args_list[1].args[0])
            self.assertIn("已完成迭代：2/3", sender.call_args_list[2].args[1])
            self.assertIn("最终 mini 测试完成", sender.call_args_list[3].args[0])
            provenance = json.loads((root / "provenance.json").read_text())
            self.assertEqual(provenance["resume_from"], str(resume_path))
            self.assertEqual(provenance["resume_completed_steps"], 2)

            # Compare resumed updates and sampler cursor against uninterrupted execution.
            args.resume_from = None
            fix_random_seeds(args.seed)
            reference = ToyModel()
            control = Path(directory) / "control"
            (control / "checkpoints").mkdir(parents=True)
            train(reference, args, torch.device("cpu"), control, control / "checkpoints", {})
            torch.testing.assert_close(resumed.weight, reference.weight, rtol=0, atol=0)

            # Recover a final checkpoint saved before its mandatory final test.
            final = checkpoints / "ckpt_final.pth"
            payload = torch.load(final, map_location="cpu")
            payload["state"]["final_test_complete"] = False
            payload["state"]["tested_steps"] = [2]
            torch.save(payload, final)
            # Auto-discovery must also find a final checkpoint awaiting its mini test.
            self.assertIsNone(args.resume_from)
            events.clear()
            train(ToyModel(), args, torch.device("cpu"), root, checkpoints, {})
            self.assertEqual(events, [("mini", 3)])
            final_payload = torch.load(final, map_location="cpu")
            self.assertTrue(final_payload["state"]["final_test_complete"])
            events.clear()
            sender.reset_mock()
            final_bytes = final.read_bytes()
            before_provenance = (root / "provenance.json").read_bytes()
            with patch("storm.omniscene_runner.ImageMetrics") as metrics:
                completed = train(ToyModel(), args, torch.device("cpu"), root, checkpoints, {})
                metrics.assert_not_called()
            self.assertTrue(completed["final_test_complete"])
            self.assertEqual(final.read_bytes(), final_bytes)
            self.assertEqual((root / "provenance.json").read_bytes(), before_provenance)
            self.assertEqual(events, [])
            sender.assert_not_called()
            args.auto_resume = False
            with self.assertRaises(FileExistsError):
                train(ToyModel(), args, torch.device("cpu"), root, checkpoints, {})
            args.auto_resume = True
            args.lr = .0003
            with self.assertRaisesRegex(ValueError, "Resume config differs"):
                train(ToyModel(), args, torch.device("cpu"), root, checkpoints, {})
            args.lr = .0004
            final_payload["eval_manifest_sha256"] = "a-different-evaluation-manifest"
            torch.save(final_payload, final)
            with self.assertRaisesRegex(ValueError, "Evaluation manifest"):
                train(ToyModel(), args, torch.device("cpu"), root, checkpoints, {})


if __name__ == "__main__":
    unittest.main()
