import copy
import json
from pathlib import Path

from rvsd.configs.baseline import RVSDBaselineConfig
from rvsd.tools.infer import measure_model_speed_stats


def _controlled_speed_config(config: RVSDBaselineConfig) -> RVSDBaselineConfig:
    speed_config = copy.deepcopy(config)
    speed_config.dataset.dummy_mode = True
    speed_config.dataset.max_videos = None
    speed_config.dataset.max_clips_per_video = None
    required_steps = max(speed_config.run.fps_warmup_steps, 0) + max(speed_config.run.fps_measure_steps, 1)
    min_samples = max(required_steps * max(speed_config.run.batch_size, 1), 1)
    speed_config.dataset.dummy_num_samples = max(speed_config.dataset.dummy_num_samples, min_samples)
    speed_config.run.save_predictions = False
    speed_config.run.save_visualizations = False
    return speed_config


def run_speed_test(config: RVSDBaselineConfig, split: str = "test") -> dict[str, float | int | str]:
    speed_config = _controlled_speed_config(config)
    speed_stats = measure_model_speed_stats(speed_config, split=split)

    metrics: dict[str, float | int | str] = {
        "measurement_type": "controlled_model_fps",
        "model_fps": float(speed_stats["model_fps"]),
        "mean_iter_ms": float(speed_stats["mean_iter_ms"]),
        "frames_per_iter": int(speed_stats["frames_per_iter"]),
        "measured_iterations": int(speed_stats["measured_iterations"]),
        "measured_batch_shape": str(tuple(speed_stats["measured_batch_shape"])),
        "batch_size": int(speed_config.run.batch_size),
        "clip_length": int(speed_config.dataset.clip_length),
        "image_height": int(speed_config.dataset.image_size[0]),
        "image_width": int(speed_config.dataset.image_size[1]),
        "fps_warmup_steps": int(speed_config.run.fps_warmup_steps),
        "fps_measure_steps": int(speed_config.run.fps_measure_steps),
        "dummy_num_samples": int(speed_config.dataset.dummy_num_samples),
    }

    output_dir = Path(speed_config.run.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "speed_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("[speed] controlled model FPS measurement")
    print(
        "[speed] "
        f"batch_size={speed_config.run.batch_size} "
        f"clip_length={speed_config.dataset.clip_length} "
        f"image_size={tuple(speed_config.dataset.image_size)} "
        f"warmup={speed_config.run.fps_warmup_steps} "
        f"measure={speed_config.run.fps_measure_steps} "
        f"frames_per_iter={metrics['frames_per_iter']} "
        f"mean_iter_ms={metrics['mean_iter_ms']:.3f}"
    )
    print(f"[speed] metrics saved to {metrics_path}")
    print("[speed] " + " ".join(f"{key}={value}" for key, value in metrics.items()))
    return metrics
