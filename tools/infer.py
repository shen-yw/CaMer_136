import copy
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from tqdm.auto import tqdm

from rvsd.configs.baseline import RVSDBaselineConfig
from rvsd.datasets import build_dataloader
from rvsd.metrics import SODA_MAE_THRESHOLD, SODAMetricAggregator, build_soda_metric_aggregator
from rvsd.models import build_vsd_model
from rvsd.tools.debug_diagnostics import build_debug_summary, save_debug_summary
from rvsd.tools.train import (
    _maybe_reset_legacy_batch_size_correction,
    get_autocast_context,
    load_checkpoint_if_available,
    resolve_device,
)


def _load_compatible_checkpoint(model: torch.nn.Module, checkpoint_path: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_model = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    incompatible = model.load_state_dict(checkpoint_model, strict=False)
    _maybe_reset_legacy_batch_size_correction(model, checkpoint, prefix="[checkpoint]")
    print(f"[checkpoint] Loaded compatible checkpoint from {checkpoint_path}")
    if incompatible.missing_keys:
        print(f"[checkpoint] missing keys: {len(incompatible.missing_keys)}")
    if incompatible.unexpected_keys:
        print(f"[checkpoint] unexpected keys: {len(incompatible.unexpected_keys)}")


def _save_mask(mask_tensor: torch.Tensor, output_path: Path, threshold: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    probs = torch.sigmoid(mask_tensor).squeeze(0).cpu().numpy()
    mask = (probs > threshold).astype(np.uint8) * 255
    Image.fromarray(mask).save(output_path)


def _save_overlay(mask_tensor: torch.Tensor, image_path: str, output_path: Path, threshold: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(image_path).convert("RGB")
    image_np = np.asarray(image).astype(np.float32)
    probs = torch.sigmoid(mask_tensor).squeeze(0).cpu().numpy()
    mask = probs > threshold
    overlay = image_np.copy()
    overlay[mask, 0] = 0.65 * overlay[mask, 0] + 0.35 * 255.0
    overlay[mask, 1] = 0.65 * overlay[mask, 1]
    overlay[mask, 2] = 0.65 * overlay[mask, 2]
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(output_path)


def _load_binary_mask(label_path: str) -> torch.Tensor:
    mask = Image.open(label_path).convert("L")
    array = (np.asarray(mask, dtype=np.float32) > 0).astype(np.float32)
    return torch.from_numpy(array).unsqueeze(0)


@torch.no_grad()
def summarize_predictions_metrics(
    aggregated: dict[str, dict[str, dict[str, object]]],
    output_dir: str | Path | None = None,
    filename: str = "infer_metrics.json",
    threshold: float | None = None,
    hard_threshold: float | None = None,
    mae_threshold: float | None = None,
    use_soft_mae: bool = False,
) -> dict[str, float]:
    metric_aggregator = SODAMetricAggregator(
        threshold=threshold,
        hard_threshold=hard_threshold,
        mae_threshold=SODA_MAE_THRESHOLD if mae_threshold is None else mae_threshold,
        use_soft_mae=use_soft_mae,
    )
    evaluation_items = [
        prediction
        for video_predictions in aggregated.values()
        for prediction in video_predictions.values()
    ]
    progress_bar = tqdm(
        evaluation_items,
        total=len(evaluation_items),
        desc="Metric",
        dynamic_ncols=True,
        leave=True,
    )
    for prediction in progress_bar:
        logit = prediction["avg_logit"]
        target = _load_binary_mask(prediction["label_path"]).to(logit.dtype)
        if tuple(target.shape[-2:]) != tuple(logit.shape[-2:]):
            target = F.interpolate(target.unsqueeze(0), size=logit.shape[-2:], mode="nearest").squeeze(0)
        metric_aggregator.update(logit, target)
    progress_bar.close()

    metrics = metric_aggregator.summarize()
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        metrics_path = output_path / filename
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"[infer] metrics saved to {metrics_path}")
    print("[infer] " + " ".join(f"{key}={value:.6f}" for key, value in metrics.items()))
    return metrics


def _expected_frame_counts(dataset) -> dict[tuple[str, str], int]:
    expected: dict[tuple[str, str], int] = {}
    samples = getattr(dataset, "samples", None)
    if samples is None:
        return expected
    for sample in samples:
        video_id = str(sample["video_id"])
        for pair in sample["frame_pairs"]:
            if bool(pair.get("is_padding", False)):
                continue
            key = (video_id, str(pair["stem"]))
            expected[key] = expected.get(key, 0) + 1
    return expected


def _finalize_pending_prediction(
    frame_store: dict[str, object],
    metric_aggregator: SODAMetricAggregator | None,
    config: RVSDBaselineConfig,
    predictions_dir: Path,
    visualization_dir: Path,
) -> torch.Tensor:
    avg_logit = frame_store["logit_sum"] / max(frame_store["count"], 1)
    frame_store["avg_logit"] = avg_logit
    if metric_aggregator is not None:
        target = _load_binary_mask(frame_store["label_path"]).to(avg_logit.dtype)
        if tuple(target.shape[-2:]) != tuple(avg_logit.shape[-2:]):
            target = F.interpolate(target.unsqueeze(0), size=avg_logit.shape[-2:], mode="nearest").squeeze(0)
        metric_aggregator.update(avg_logit, target)
    if config.run.save_predictions:
        _save_mask(
            mask_tensor=avg_logit,
            output_path=predictions_dir / str(frame_store["video_id"]) / f"{frame_store['frame_name']}.png",
            threshold=config.run.threshold,
        )
    if config.run.save_visualizations:
        _save_overlay(
            mask_tensor=avg_logit,
            image_path=str(frame_store["image_path"]),
            output_path=visualization_dir / str(frame_store["video_id"]) / f"{frame_store['frame_name']}.png",
            threshold=config.run.threshold,
        )
    return avg_logit


def _resolve_compile_block_indices(model: torch.nn.Module, config: RVSDBaselineConfig) -> tuple[int, ...]:
    explicit = tuple(int(index) for index in getattr(config.run, "compile_backbone_block_indices", ()) or ())
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        return explicit
    num_blocks = int(getattr(backbone, "n_blocks", 0))
    if explicit:
        return tuple(index for index in explicit if 0 <= index < num_blocks)
    if not bool(getattr(config.model.adaptive, "enable_token_merging", False)):
        return tuple()
    if str(getattr(config.model.adaptive, "runtime_mode", "")).lower() != "risk_v1":
        return tuple()
    try:
        _transient_indices, persistent_index = backbone._resolve_risk_v1_schedule()
        tail_stop_index = backbone._resolve_tail_stop_block_index(persistent_index)
    except Exception:
        persistent_index = int(getattr(config.model.adaptive, "merge_block_index", 0))
        tail_stop_index = int(getattr(config.model.adaptive, "tail_stop_block_index", -1))
        if tail_stop_index < 0:
            tail_stop_index = num_blocks - 1
    persistent_index = max(0, min(int(persistent_index), max(num_blocks - 1, 0)))
    tail_stop_index = max(persistent_index, min(int(tail_stop_index), max(num_blocks - 1, 0)))
    return tuple(range(persistent_index, tail_stop_index + 1))


@torch.no_grad()
def build_inference_model(config: RVSDBaselineConfig):
    device = resolve_device(config.run.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = bool(config.run.enable_cudnn_benchmark)
        allow_tf32 = bool(config.run.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
        torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")
    model = build_vsd_model(config).to(device)
    patch_size = getattr(model.backbone, "patch_size", None)
    image_size = tuple(int(dim) for dim in config.dataset.image_size)
    if (
        patch_size is not None
        and len(image_size) == 2
        and image_size[0] % int(patch_size) == 0
        and image_size[1] % int(patch_size) == 0
    ):
        patch_height = image_size[0] // int(patch_size)
        patch_width = image_size[1] // int(patch_size)
        layout = model.backbone._group_layout(patch_height, patch_width)
        model.backbone._group_layout_tensors(layout, device)
        model.backbone._base_rope(patch_height, patch_width, device)
    try:
        load_checkpoint_if_available(config, model=model)
    except RuntimeError as exc:
        runtime_mode = config.model.adaptive.runtime_mode.lower()
        explicit_checkpoint = config.run.checkpoint_path or config.optimizer.resume_checkpoint
        if runtime_mode == "risk_v1" and explicit_checkpoint:
            _load_compatible_checkpoint(model, explicit_checkpoint)
        else:
            raise exc
    model.eval()
    if bool(getattr(config.run, "compile_backbone_blocks", False)):
        backbone = getattr(model, "backbone", None)
        if backbone is not None and hasattr(backbone, "compile_inference_blocks"):
            block_indices = _resolve_compile_block_indices(model, config)
            try:
                compiled_indices = backbone.compile_inference_blocks(
                    block_indices,
                    mode=str(getattr(config.run, "compile_mode", "reduce-overhead")),
                    fullgraph=bool(getattr(config.run, "compile_fullgraph", False)),
                )
                print(
                    "[infer] local torch.compile backbone blocks: "
                    f"{compiled_indices if compiled_indices else '<none>'}"
                )
            except Exception as exc:
                print(f"[infer] local torch.compile backbone blocks unavailable, using eager blocks: {exc}")
    return model, device


class _LogitsOnlyInferenceWrapper(nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        return self.model(clip, return_outputs=False)


class _CudaGraphInferenceReplay:
    """Replay a fixed-shape inference graph while keeping data movement outside timing."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        use_amp: bool,
        warmup_replays: int = 3,
    ) -> None:
        self.model = model
        self.device = device
        self.use_amp = use_amp
        self.warmup_replays = max(int(warmup_replays), 0)
        self.graph: torch.cuda.CUDAGraph | None = None
        self.static_input: torch.Tensor | None = None
        self.static_output: torch.Tensor | None = None
        self.input_shape: tuple[int, ...] | None = None
        self.capture_error: str | None = None

    def is_ready_for(self, clip: torch.Tensor) -> bool:
        return self.graph is not None and self.input_shape == tuple(int(dim) for dim in clip.shape)

    def capture(self, clip: torch.Tensor) -> None:
        if self.device.type != "cuda":
            raise RuntimeError("CUDA Graph capture requires a CUDA device.")
        self.input_shape = tuple(int(dim) for dim in clip.shape)
        self.static_input = torch.empty_like(clip)
        self.static_input.copy_(clip, non_blocking=True)
        torch.cuda.synchronize(self.device)

        # Prime allocator and lazy kernels on a side stream so capture only records the steady-state graph.
        warmup_stream = torch.cuda.Stream(device=self.device)
        warmup_stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(warmup_stream):
            with torch.inference_mode():
                with get_autocast_context(self.device, self.use_amp):
                    for _ in range(self.warmup_replays):
                        self.model(self.static_input)
        torch.cuda.current_stream(self.device).wait_stream(warmup_stream)
        torch.cuda.synchronize(self.device)

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            with torch.inference_mode():
                with get_autocast_context(self.device, self.use_amp):
                    self.static_output = self.model(self.static_input)
        torch.cuda.synchronize(self.device)

    def copy_input(self, clip: torch.Tensor) -> None:
        if self.static_input is None:
            raise RuntimeError("CUDA Graph input buffer is not initialized.")
        self.static_input.copy_(clip, non_blocking=True)

    def replay(self) -> torch.Tensor | None:
        if self.graph is None:
            raise RuntimeError("CUDA Graph has not been captured.")
        self.graph.replay()
        return self.static_output


@torch.no_grad()
def capture_first_debug_snapshot(config: RVSDBaselineConfig, split: str = "test") -> dict[str, object] | None:
    model, device = build_inference_model(config)
    dataloader = build_dataloader(config, split=split, training=False, shuffle=False)
    batch = next(iter(dataloader), None)
    if batch is None:
        return None
    clip = batch["clip"].to(device, non_blocking=True)
    with get_autocast_context(device, config.optimizer.use_amp):
        _, outputs = model(clip, return_outputs=True)
    return outputs.get("debug")


@torch.no_grad()
def measure_model_speed_stats(config: RVSDBaselineConfig, split: str = "test") -> dict[str, object]:
    if (
        bool(getattr(config.run, "use_cuda_graph", False))
        and bool(getattr(config.run, "cuda_graph_disable_group_contract_extension", True))
        and bool(getattr(config.model.adaptive, "use_group_contract_extension", False))
    ):
        config = copy.deepcopy(config)
        config.model.adaptive.use_group_contract_extension = False
        print("[speed_real] CUDA Graph compatibility: disabled group contract extension for graph capture.")
    model, device = build_inference_model(config)
    print(
        "[speed_real] "
        f"inference backend use_amp={bool(config.optimizer.use_amp)} "
        f"allow_tf32={bool(config.run.allow_tf32)} "
        f"cudnn_benchmark={bool(config.run.enable_cudnn_benchmark)} "
        f"profile={bool(getattr(config.run, 'speed_profile_enabled', False))}"
    )
    speed_profile_enabled = bool(getattr(config.run, "speed_profile_enabled", False))
    if hasattr(model, "enable_speed_profile"):
        model.enable_speed_profile = speed_profile_enabled
    if hasattr(model, "backbone") and hasattr(model.backbone, "enable_speed_profile"):
        model.backbone.enable_speed_profile = speed_profile_enabled
    if hasattr(model, "backbone") and hasattr(model.backbone, "enable_speed_profile_detail"):
        model.backbone.enable_speed_profile_detail = speed_profile_enabled
    if hasattr(model, "backbone") and hasattr(model.backbone, "_group_contract_extension_enabled"):
        print(f"[speed_real] group contract extension enabled: {bool(model.backbone._group_contract_extension_enabled)}")
    eager_speed_model: torch.nn.Module = _LogitsOnlyInferenceWrapper(model)
    speed_model: torch.nn.Module = eager_speed_model
    compile_enabled = False
    compile_probe_pending = False
    cuda_graph_requested = (
        device.type == "cuda"
        and bool(getattr(config.run, "use_cuda_graph", False))
        and not speed_profile_enabled
    )
    cuda_graph_runner: _CudaGraphInferenceReplay | None = None
    cuda_graph_used = False
    cuda_graph_error = ""
    if bool(getattr(config.run, "use_cuda_graph", False)) and speed_profile_enabled:
        print("[speed_real] CUDA Graph disabled because speed_profile_enabled=true needs Python timing hooks.")
    if config.run.compile_inference and hasattr(torch, "compile"):
        try:
            speed_model = torch.compile(
                speed_model,
                mode=config.run.compile_mode,
                fullgraph=bool(config.run.compile_fullgraph),
                dynamic=False,
            )
            print(f"[speed_real] torch.compile enabled mode={config.run.compile_mode} fullgraph={config.run.compile_fullgraph}")
            compile_enabled = True
            compile_probe_pending = True
        except Exception as exc:
            print(f"[speed_real] torch.compile unavailable, falling back to eager: {exc}")
    if cuda_graph_requested:
        cuda_graph_runner = _CudaGraphInferenceReplay(
            speed_model,
            device=device,
            use_amp=bool(config.optimizer.use_amp),
            warmup_replays=int(getattr(config.run, "cuda_graph_warmup_replays", 3)),
        )
        print(
            "[speed_real] CUDA Graph requested "
            f"warmup_replays={int(getattr(config.run, 'cuda_graph_warmup_replays', 3))}"
        )
    dataloader = build_dataloader(config, split=split, training=False, shuffle=False)
    total_available_steps = max(len(dataloader), 1)
    warmup_steps = min(max(config.run.fps_warmup_steps, 0), max(total_available_steps - 1, 0))
    measure_steps = min(max(config.run.fps_measure_steps, 1), max(total_available_steps - warmup_steps, 1))

    measured_frames = 0
    measured_time_s = 0.0
    measured_iterations = 0
    measured_batch_shape: tuple[int, ...] | None = None
    frames_per_iter = 0
    profile_keys = [
        "pre_merge_ms",
        "merge_ms",
        "post_merge_ms",
        "decoder_head_ms",
        "group_pack_ms",
        "risk_score_ms",
        "gate_select_ms",
        "contract_merge_ms",
        "rope_rebuild_ms",
        "post_blocks_ms",
        "restore_collect_ms",
        "collect_normalize_ms",
        "restore_to_map_ms",
        "collect_cls_ms",
        "ratio_policy_ms",
        "grouped_sizes_ms",
        "merge_plan_ms",
        "contract_kernel_ms",
        "prefix_cat_ms",
        "attention_bias_ms",
        "size_bias_ms",
        "merge_block_forward_ms",
        "sparse_state_commit_ms",
        "sparse_decoder_input_ms",
        "restore_index_prep_ms",
        "dark_prior_ms",
        "post_size_bias_apply_ms",
        "post_norm1_ms",
        "post_qkv_ms",
        "post_rope_apply_ms",
        "post_attn_ms",
        "post_attn_proj_ms",
        "post_norm2_ms",
        "post_mlp_fc1_ms",
        "post_mlp_act_ms",
        "post_mlp_fc2_ms",
        "post_residual_ms",
        "post_layout_ms",
    ]
    stage_time_totals = {key: 0.0 for key in profile_keys}

    total_steps = min(total_available_steps, warmup_steps + measure_steps)
    progress_bar = tqdm(
        dataloader,
        total=total_steps,
        desc=f"FPS {split}",
        dynamic_ncols=True,
        leave=True,
    )

    for step_index, batch in enumerate(progress_bar):
        clip = batch["clip"].to(device, non_blocking=True)
        graph_ready_for_clip = False
        if cuda_graph_runner is not None:
            if not cuda_graph_runner.is_ready_for(clip):
                try:
                    cuda_graph_runner.capture(clip)
                    cuda_graph_used = True
                    print(f"[speed_real] CUDA Graph captured shape={tuple(int(dim) for dim in clip.shape)}")
                except Exception as exc:
                    cuda_graph_error = str(exc)
                    raise RuntimeError(
                        "CUDA Graph capture failed. Disable run.use_cuda_graph or set "
                        "run.cuda_graph_disable_group_contract_extension=true for the graph-compatible path."
                    ) from exc
            if cuda_graph_runner is not None and cuda_graph_runner.is_ready_for(clip):
                cuda_graph_runner.copy_input(clip)
                graph_ready_for_clip = True
        if device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record(torch.cuda.current_stream(device))
        else:
            start = perf_counter()
        with torch.inference_mode():
            with get_autocast_context(device, config.optimizer.use_amp):
                try:
                    if graph_ready_for_clip and cuda_graph_runner is not None:
                        cuda_graph_runner.replay()
                    else:
                        speed_model(clip)
                    if compile_probe_pending:
                        compile_probe_pending = False
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        continue
                except Exception as exc:
                    if not compile_enabled:
                        raise
                    print(f"[speed_real] torch.compile runtime fallback to eager: {exc}")
                    speed_model = eager_speed_model
                    compile_enabled = False
                    compile_probe_pending = False
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    continue
        if device.type == "cuda":
            end_event.record(torch.cuda.current_stream(device))
            torch.cuda.synchronize(device)
            elapsed_s = start_event.elapsed_time(end_event) / 1000.0
        else:
            elapsed_s = perf_counter() - start

        if step_index >= warmup_steps:
            measured_batch_shape = tuple(int(dim) for dim in clip.shape)
            frames_per_iter = int(clip.shape[0] * clip.shape[2])
            measured_frames += frames_per_iter
            measured_time_s += elapsed_s
            measured_iterations += 1
            model_stage_profile = getattr(model, "last_speed_profile", {})
            for key in stage_time_totals:
                stage_time_totals[key] += float(model_stage_profile.get(key, 0.0))
        current_fps = measured_frames / measured_time_s if measured_time_s > 0 else 0.0
        progress_bar.set_postfix(fps=f"{current_fps:.2f}")
        if step_index >= warmup_steps + measure_steps - 1:
            break
    progress_bar.close()

    if measured_time_s <= 0 or measured_iterations <= 0:
        return {
            "model_fps": 0.0,
            "mean_iter_ms": 0.0,
            "frames_per_iter": 0,
            "measured_iterations": 0,
            "measured_batch_shape": (),
            "warmup_steps": warmup_steps,
            "measure_steps": 0,
            "split": split,
            "cuda_graph_requested": cuda_graph_requested,
            "cuda_graph_used": cuda_graph_used,
            "cuda_graph_error": cuda_graph_error,
        }

    return {
        "model_fps": measured_frames / measured_time_s,
        "mean_iter_ms": (measured_time_s / measured_iterations) * 1000.0,
        "frames_per_iter": frames_per_iter,
        "measured_iterations": measured_iterations,
        "measured_batch_shape": measured_batch_shape or (),
        "warmup_steps": warmup_steps,
        "measure_steps": measured_iterations,
        "split": split,
        "cuda_graph_requested": cuda_graph_requested,
        "cuda_graph_used": cuda_graph_used,
        "cuda_graph_error": cuda_graph_error,
        **{f"avg_{key}": value / measured_iterations for key, value in stage_time_totals.items()},
    }


@torch.no_grad()
def measure_model_fps(config: RVSDBaselineConfig, split: str = "test") -> float:
    return float(measure_model_speed_stats(config, split=split)["model_fps"])


@torch.no_grad()
def run_inference(
    config: RVSDBaselineConfig,
    return_predictions: bool = False,
    split: str = "test",
    compute_metrics: bool | None = None,
):
    model, device = build_inference_model(config)
    if compute_metrics is None:
        compute_metrics = bool(config.run.compute_metrics)

    dataloader = build_dataloader(config, split=split, training=False, shuffle=False)
    aggregated: dict[str, dict[str, dict[str, object]]] = {}
    expected_counts = _expected_frame_counts(dataloader.dataset)
    fast_metric_mode = (
        compute_metrics
        and not return_predictions
        and not config.run.save_predictions
        and not config.run.save_visualizations
        and bool(expected_counts)
        and not config.dataset.dummy_mode
    )
    metric_aggregator = build_soda_metric_aggregator(config) if fast_metric_mode else None
    first_debug: dict[str, object] | None = None
    finalized_frames = 0

    progress_bar = tqdm(
        dataloader,
        total=len(dataloader),
        desc=f"Infer {split}",
        dynamic_ncols=True,
        leave=True,
    )

    for batch in progress_bar:
        clip = batch["clip"].to(device, non_blocking=True)
        collect_debug = first_debug is None
        with torch.inference_mode():
            with get_autocast_context(device, config.optimizer.use_amp):
                if collect_debug:
                    logits, outputs = model(clip, return_outputs=True)
                    first_debug = outputs["debug"]
                else:
                    logits = model(clip, return_outputs=False)
        progress_bar.set_postfix(
            videos=len(aggregated) if not fast_metric_mode else finalized_frames,
            batch=tuple(clip.shape),
        )

        for batch_index, meta in enumerate(batch["meta"]):
            video_id = meta["video_id"]
            video_store = aggregated.setdefault(video_id, {})
            for time_index, frame_name in enumerate(meta["frame_names"]):
                if meta["is_padding"][time_index]:
                    continue
                original_height, original_width = meta["original_sizes"][time_index]
                frame_logit = logits[batch_index, time_index]
                frame_logit = F.interpolate(
                    frame_logit.unsqueeze(0),
                    size=(original_height, original_width),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0).cpu()

                frame_store = video_store.setdefault(
                    frame_name,
                    {
                        "logit_sum": torch.zeros_like(frame_logit),
                        "count": 0,
                        "image_path": meta["image_paths"][time_index],
                        "label_path": meta["label_paths"][time_index],
                        "original_size": (original_height, original_width),
                        "video_id": video_id,
                        "frame_name": frame_name,
                    },
                )
                frame_store["logit_sum"] += frame_logit
                frame_store["count"] += 1
                if fast_metric_mode:
                    expected_count = expected_counts.get((str(video_id), str(frame_name)), 1)
                    if int(frame_store["count"]) >= expected_count:
                        _finalize_pending_prediction(
                            frame_store=frame_store,
                            metric_aggregator=metric_aggregator,
                            config=config,
                            predictions_dir=Path(config.run.output_dir) / "predictions",
                            visualization_dir=Path(config.run.output_dir) / "visualizations",
                        )
                        del video_store[frame_name]
                        finalized_frames += 1
            if fast_metric_mode and not video_store:
                aggregated.pop(video_id, None)
    progress_bar.close()

    if first_debug is not None:
        print(f"[infer] input shape: {first_debug['input_shape']}")
        print(f"[infer] backbone feature shapes: {first_debug['backbone_feature_shapes']}")
        print(f"[infer] fusion output shape: {first_debug['fusion_output_shape']}")
        print(f"[infer] head output shape: {first_debug['head_output_shape']}")
        print(f"[infer] final logits shape: {first_debug['final_logits_shape']}")
        if "decoder_variant" in first_debug:
            print(f"[infer] decoder variant: {first_debug['decoder_variant']}")
        if "sparse_decoder_dense_source_indices" in first_debug:
            print(f"[infer] sparse decoder dense source indices: {first_debug['sparse_decoder_dense_source_indices']}")
        if "sparse_decoder_dense_shapes" in first_debug:
            print(f"[infer] sparse decoder dense shapes: {first_debug['sparse_decoder_dense_shapes']}")
        if "sparse_decoder_sparse_tokens_shape" in first_debug:
            print(f"[infer] sparse decoder sparse tokens shape: {first_debug['sparse_decoder_sparse_tokens_shape']}")
        if "sparse_decoder_sparse_coords_shape" in first_debug:
            print(f"[infer] sparse decoder sparse coords shape: {first_debug['sparse_decoder_sparse_coords_shape']}")
        if "sparse_decoder_sparse_sizes_shape" in first_debug:
            print(f"[infer] sparse decoder sparse sizes shape: {first_debug['sparse_decoder_sparse_sizes_shape']}")
        if "sparse_decoder_patch_hw" in first_debug:
            print(f"[infer] sparse decoder patch hw: {first_debug['sparse_decoder_patch_hw']}")
        if "runtime_mode" in first_debug:
            print(f"[infer] runtime mode: {first_debug['runtime_mode']}")
        if "tail_stop_block_index" in first_debug:
            print(
                f"[infer] tail stop block index: {first_debug['tail_stop_block_index']} "
                f"(skipped tail blocks={first_debug.get('skipped_tail_blocks', 0)})"
            )
        if "group_contract_extension_enabled" in first_debug:
            print(f"[infer] group contract extension enabled: {first_debug['group_contract_extension_enabled']}")
        if "attention_bias_correction_enabled" in first_debug:
            print(f"[infer] attention bias correction enabled: {first_debug['attention_bias_correction_enabled']}")
        if "keep_ratios" in first_debug:
            print(f"[infer] keep ratios: {first_debug['keep_ratios']}")
            print(f"[infer] merge block indices: {first_debug['merge_block_indices']}")
        if "risk_hard_gate_ratio" in first_debug:
            print(f"[infer] risk hard gate ratio: {first_debug['risk_hard_gate_ratio']:.4f}")
        if "risk_dynamic_keep_enabled" in first_debug:
            print(f"[infer] risk dynamic keep enabled: {first_debug['risk_dynamic_keep_enabled']}")
        if "risk_dynamic_hard_gate_enabled" in first_debug:
            print(f"[infer] risk dynamic hard gate enabled: {first_debug['risk_dynamic_hard_gate_enabled']}")
        if "stage_merge_stats" in first_debug:
            for stage_stats in first_debug["stage_merge_stats"]:
                print(
                    f"[infer] stage {stage_stats['merge_block_index']} groups: "
                    f"merged_groups={stage_stats['merged_groups']} "
                    f"protected_groups={stage_stats['protected_groups']} "
                    f"total_groups={stage_stats['total_groups']} "
                    f"risk_hard_gate_ratio={stage_stats['risk_hard_gate_ratio']:.4f} "
                    f"applied_hard_gate_ratio={stage_stats.get('applied_hard_gate_ratio', 0.0):.4f} "
                    f"protected_ratio={stage_stats['protected_ratio']:.4f} "
                    f"hard_gate_active={stage_stats['hard_gate_active']} "
                    f"applied_keep_ratio={stage_stats.get('applied_keep_ratio', 0.0):.4f} "
                    f"risk_complexity={stage_stats.get('risk_complexity', -1.0):.4f}"
                )
        if "final_token_keep_ratio" in first_debug:
            keep_ratio = float(first_debug["final_token_keep_ratio"])
            saved_ratio = float(first_debug.get("final_token_saved_ratio", 1.0 - keep_ratio))
            print(
                f"[infer] final token keep ratio: {keep_ratio:.4f} "
                f"(saved {saved_ratio:.4f})"
            )
        if "final_interface_token_keep_ratio" in first_debug:
            print(f"[infer] final interface token keep ratio: {first_debug['final_interface_token_keep_ratio']:.4f}")
        debug_summary = build_debug_summary(
            first_debug,
            requested_keep_ratio=float(config.model.adaptive.fixed_keep_ratio),
            seed=int(config.run.seed),
            context="infer_first_batch",
            extra={
                "split": str(config.run.split_name),
                "output_dir": str(config.run.output_dir),
                "checkpoint_path": str(config.run.checkpoint_path),
            },
        )
        debug_summary_path = save_debug_summary(
            Path(config.run.output_dir) / "infer_debug_summary.json",
            debug_summary,
        )
        print(f"[infer] debug summary saved to {debug_summary_path}")

    predictions_dir = Path(config.run.output_dir) / "predictions"
    visualization_dir = Path(config.run.output_dir) / "visualizations"

    for video_id, frames in list(aggregated.items()):
        for frame_name, frame_store in list(frames.items()):
            avg_logit = _finalize_pending_prediction(
                frame_store=frame_store,
                metric_aggregator=metric_aggregator,
                config=config,
                predictions_dir=predictions_dir,
                visualization_dir=visualization_dir,
            )
            frame_store["avg_logit"] = avg_logit
        if fast_metric_mode and not return_predictions:
            aggregated.pop(video_id, None)

    if config.run.save_predictions:
        print(f"[infer] predictions saved to {predictions_dir}")
    if config.run.save_visualizations:
        print(f"[infer] visualizations saved to {visualization_dir}")

    metrics = None
    if compute_metrics:
        if fast_metric_mode and metric_aggregator is not None:
            metrics = metric_aggregator.summarize()
            output_path = Path(config.run.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            metrics_path = output_path / "infer_metrics.json"
            metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            print("[infer] fast metric mode active (single-pass exact aggregation)")
            print(f"[infer] metrics saved to {metrics_path}")
            print("[infer] " + " ".join(f"{key}={value:.6f}" for key, value in metrics.items()))
        else:
            metrics = summarize_predictions_metrics(
                aggregated,
                output_dir=config.run.output_dir,
                threshold=config.run.threshold,
                hard_threshold=float(getattr(config.run, "metric_hard_threshold", config.run.threshold)),
                mae_threshold=float(getattr(config.run, "metric_mae_threshold", 12.0 / 255.0)),
                use_soft_mae=bool(getattr(config.run, "metric_use_soft_mae", False)),
            )

    if return_predictions:
        return aggregated
    result = {"num_videos": len(aggregated)}
    if metrics is not None:
        result["metrics"] = metrics
    return result
