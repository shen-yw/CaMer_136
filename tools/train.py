import math
import os
from pathlib import Path
from datetime import timedelta
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from rvsd.configs.baseline import RVSDBaselineConfig
from rvsd.datasets import build_dataloader, build_shadow_free_dataloader
from rvsd.losses import build_segmentation_loss, build_shadow_free_aux_loss
from rvsd.metrics import build_soda_metric_aggregator, compute_shadow_metrics
from rvsd.models import build_vsd_model
from rvsd.tools.debug_diagnostics import build_debug_summary, save_debug_summary


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[2]


def configure_cuda_runtime() -> None:
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def init_distributed_mode(device_name: str) -> tuple[torch.device, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size > 1:
        backend = "gloo" if os.name == "nt" else ("nccl" if device_name == "cuda" and torch.cuda.is_available() else "gloo")
        if device_name == "cuda" and torch.cuda.is_available():
            configure_cuda_runtime()
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")
        master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
        master_port = int(os.environ.get("MASTER_PORT", "29541"))
        store = dist.TCPStore(
            host_name=master_addr,
            port=master_port,
            world_size=world_size,
            is_master=rank == 0,
            timeout=timedelta(minutes=30),
            use_libuv=False,
        )
        dist.init_process_group(
            backend=backend,
            store=store,
            rank=rank,
            world_size=world_size,
            timeout=timedelta(minutes=30),
        )
        return device, rank, world_size

    return resolve_device(device_name), rank, world_size


def barrier() -> None:
    if is_distributed():
        dist.barrier()


def reduce_tensor_mean(value: torch.Tensor) -> torch.Tensor:
    if not is_distributed():
        return value
    reduced = value.clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    reduced /= get_world_size()
    return reduced


def reduce_float(value: float, device: torch.device) -> float:
    tensor = torch.tensor(value, device=device, dtype=torch.float32)
    return float(reduce_tensor_mean(tensor).item())


def compute_best_metric_value(metrics: dict[str, float], metric_name: str) -> float:
    """Return a maximized checkpoint score for scalar or composite val metrics."""
    normalized_name = metric_name.lower()
    if normalized_name in {"mae", "ber", "shadow_ber", "s_ber", "non_shadow_ber", "n_ber", "loss"}:
        if normalized_name not in metrics:
            raise ValueError(f"Unsupported best metric: {metric_name}")
        return -float(metrics[normalized_name])
    if normalized_name in {"soda_balanced", "balanced_soda"}:
        return (
            0.55 * float(metrics["jaccard"])
            + 0.45 * float(metrics["fmeasure"])
            - 0.25 * float(metrics["mae"])
            - 0.010 * float(metrics["ber"])
        )
    if normalized_name in {"soda_sber", "sber_soda", "shadow_recall"}:
        return (
            0.50 * float(metrics["jaccard"])
            + 0.35 * float(metrics["fmeasure"])
            - 0.25 * float(metrics["mae"])
            - 0.010 * float(metrics["ber"])
            - 0.006 * float(metrics["s_ber"])
            - 0.001 * float(metrics["n_ber"])
        )
    if normalized_name in {"soda_iou", "iou_soda", "iou_nber", "soda_iou_nber"}:
        return (
            0.68 * float(metrics["jaccard"])
            + 0.22 * float(metrics["fmeasure"])
            - 0.40 * float(metrics["mae"])
            - 0.006 * float(metrics["ber"])
            - 0.002 * float(metrics["s_ber"])
            - 0.010 * float(metrics["n_ber"])
        )
    metric_value = metrics.get(normalized_name)
    if metric_value is None:
        raise ValueError(f"Unsupported best metric: {metric_name}")
    return float(metric_value)


def best_checkpoint_filename(metric_name: str) -> str:
    normalized_name = metric_name.lower()
    aliases = {
        "iou": "best_iou.pt",
        "jaccard": "best_iou.pt",
        "ber": "best_ber.pt",
        "shadow_ber": "best_sber.pt",
        "s_ber": "best_sber.pt",
        "soda_balanced": "best_soda_balanced.pt",
        "balanced_soda": "best_soda_balanced.pt",
        "soda_sber": "best_soda_sber.pt",
        "sber_soda": "best_soda_sber.pt",
        "soda_iou": "best_soda_iou.pt",
        "iou_soda": "best_soda_iou.pt",
        "iou_nber": "best_soda_iou.pt",
        "soda_iou_nber": "best_soda_iou.pt",
    }
    if normalized_name in aliases:
        return aliases[normalized_name]
    safe_name = "".join(character if character.isalnum() or character in {"_", "-"} else "_" for character in normalized_name)
    return f"best_{safe_name}.pt"


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and torch.cuda.is_available():
        configure_cuda_runtime()
        return torch.device("cuda")
    return torch.device("cpu")


def get_autocast_context(device: torch.device, use_amp: bool):
    enabled = use_amp and device.type == "cuda"
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled)


def _build_tensorboard_writer(config: RVSDBaselineConfig):
    if not is_main_process() or not config.run.enable_tensorboard:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        print("[tracking] TensorBoard is unavailable; scalar logging disabled")
        return None
    log_dir = config.run.tensorboard_log_dir or str(Path(config.run.output_dir) / "tensorboard")
    writer = SummaryWriter(log_dir=log_dir)
    print(f"[tracking] tensorboard_log_dir={log_dir}")
    return writer


def _build_wandb_run(config: RVSDBaselineConfig):
    if not is_main_process() or not config.run.enable_wandb:
        return None
    try:
        import wandb
    except ImportError:
        print("[tracking] W&B is unavailable; wandb logging disabled")
        return None
    run_name = config.run.wandb_run_name or Path(config.run.output_dir).name
    wandb_dir = Path(config.run.output_dir) / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        project=config.run.wandb_project,
        name=run_name,
        mode=config.run.wandb_mode,
        dir=str(wandb_dir),
        config={
            "output_dir": config.run.output_dir,
            "seed": config.run.seed,
            "decoder_variant": config.model.head.decoder_variant,
            "best_metric_name": config.optimizer.best_metric_name,
            "max_epochs": config.optimizer.max_epochs,
            "batch_size": config.run.batch_size,
            "learning_rate": config.optimizer.learning_rate,
            "backbone_learning_rate": config.optimizer.backbone_learning_rate,
            "positive_bce_weight": config.optimizer.positive_bce_weight,
            "lambda_dark_negative": config.optimizer.lambda_dark_negative,
            "lambda_nonshadow_precision": config.optimizer.lambda_nonshadow_precision,
        },
    )
    print(f"[tracking] wandb_project={config.run.wandb_project} wandb_mode={config.run.wandb_mode}")
    return run


def _log_scalar_dict(writer, wandb_run, prefix: str, values: dict[str, float], step: int) -> None:
    scalar_values = {
        key: float(value)
        for key, value in values.items()
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    }
    if writer is not None:
        for key, value in scalar_values.items():
            writer.add_scalar(f"{prefix}/{key}", value, step)
        writer.flush()
    if wandb_run is not None:
        wandb_run.log({f"{prefix}/{key}": value for key, value in scalar_values.items()}, step=step)


def build_optimizer(model: torch.nn.Module, config: RVSDBaselineConfig) -> torch.optim.Optimizer:
    model = unwrap_model(model)
    backbone_params = [parameter for parameter in model.backbone.trainable_backbone_parameters() if parameter.requires_grad]
    backbone_param_ids = {id(parameter) for parameter in backbone_params}
    decoder_params = [
        parameter for parameter in model.parameters() if parameter.requires_grad and id(parameter) not in backbone_param_ids
    ]

    param_groups = []
    if decoder_params:
        param_groups.append(
            {
                "params": decoder_params,
                "lr": config.optimizer.learning_rate,
                "weight_decay": config.optimizer.weight_decay,
            }
        )
    if backbone_params:
        param_groups.append(
            {
                "params": backbone_params,
                "lr": config.optimizer.backbone_learning_rate,
                "weight_decay": config.optimizer.weight_decay,
            }
        )
    if not param_groups:
        raise RuntimeError("No trainable parameters found for optimizer construction.")
    return torch.optim.AdamW(param_groups)


def build_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup_steps: int, min_lr_ratio: float):
    total_steps = max(total_steps, 1)
    warmup_steps = min(max(warmup_steps, 0), total_steps)

    def lr_lambda(current_step: int) -> float:
        step = current_step + 1
        if warmup_steps > 0 and step <= warmup_steps:
            return step / warmup_steps
        if total_steps <= warmup_steps:
            return 1.0
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def should_use_shadow_free_step(sample_ratio: float, global_step: int) -> bool:
    if sample_ratio <= 0:
        return False
    if sample_ratio >= 1:
        return True
    interval = max(1, int(round(1.0 / sample_ratio)))
    return global_step % interval == 0


def next_shadow_free_batch(loader, iterator_holder: dict[str, object]):
    if loader is None:
        return None
    iterator = iterator_holder.get("iterator")
    if iterator is None:
        iterator = iter(loader)
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        batch = next(iterator)
    iterator_holder["iterator"] = iterator
    return batch


def _resolve_existing_path(path_str: str, output_dir: str | None = None) -> Path:
    candidate = Path(path_str).expanduser()
    if candidate.is_absolute():
        return candidate

    search_roots: list[Path] = [Path.cwd(), workspace_root()]
    if output_dir and len(candidate.parts) == 1:
        search_roots.append(Path(output_dir))

    for root in search_roots:
        resolved = (root / candidate).resolve()
        if resolved.exists():
            return resolved

    if output_dir and len(candidate.parts) == 1:
        return (Path(output_dir) / candidate).resolve()
    return (Path.cwd() / candidate).resolve()


def _resolve_checkpoint_dir(config: RVSDBaselineConfig) -> Path:
    checkpoint_dir = Path(config.optimizer.checkpoint_dir).expanduser()
    if checkpoint_dir.is_absolute():
        resolved = checkpoint_dir
    elif len(checkpoint_dir.parts) == 1:
        resolved = (Path(config.run.output_dir) / checkpoint_dir).resolve()
    else:
        cwd_resolved = (Path.cwd() / checkpoint_dir).resolve()
        workspace_resolved = (workspace_root() / checkpoint_dir).resolve()
        resolved = workspace_resolved if workspace_resolved.exists() else cwd_resolved
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def default_checkpoint_path(config: RVSDBaselineConfig) -> Path:
    return _resolve_checkpoint_dir(config) / "latest.pt"


def has_explicit_checkpoint(config: RVSDBaselineConfig) -> bool:
    return bool(config.run.checkpoint_path or config.optimizer.resume_checkpoint)


def _maybe_reset_legacy_batch_size_correction(model: torch.nn.Module, checkpoint: dict | object, prefix: str) -> None:
    model = unwrap_model(model)
    checkpoint_meta = checkpoint.get("meta", {}) if isinstance(checkpoint, dict) else {}
    if bool(checkpoint_meta.get("batch_size_correction_v1", False)):
        return
    backbone = getattr(model, "backbone", None)
    size_correction = getattr(backbone, "size_correction", None)
    if size_correction is None:
        return
    with torch.no_grad():
        torch.nn.init.zeros_(size_correction.scale_proj.weight)
        torch.nn.init.zeros_(size_correction.scale_proj.bias)
        torch.nn.init.zeros_(size_correction.bias_proj.weight)
        torch.nn.init.zeros_(size_correction.bias_proj.bias)
    print(f"{prefix} legacy checkpoint detected; reset size_correction to neutral")


def warm_start_model_if_available(config: RVSDBaselineConfig, model: torch.nn.Module) -> bool:
    model = unwrap_model(model)
    if not config.optimizer.warm_start_checkpoint:
        return False

    checkpoint_path = _resolve_existing_path(
        config.optimizer.warm_start_checkpoint,
        output_dir=config.run.output_dir,
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Warm-start checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_model = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    current_model = model.state_dict()
    filtered_checkpoint_model = {}
    skipped_shape_keys = []
    skipped_unknown_keys = []
    for key, value in checkpoint_model.items():
        if key not in current_model:
            skipped_unknown_keys.append(key)
            continue
        if tuple(current_model[key].shape) != tuple(value.shape):
            skipped_shape_keys.append(key)
            continue
        filtered_checkpoint_model[key] = value
    incompatible = model.load_state_dict(filtered_checkpoint_model, strict=False)
    _maybe_reset_legacy_batch_size_correction(model, checkpoint, prefix="[warm-start]")
    print(f"[warm-start] Loaded compatible weights from {checkpoint_path}")
    if skipped_shape_keys:
        print(f"[warm-start] skipped shape-mismatched keys: {len(skipped_shape_keys)}")
    if skipped_unknown_keys:
        print(f"[warm-start] skipped unknown keys: {len(skipped_unknown_keys)}")
    if incompatible.missing_keys:
        print(f"[warm-start] missing keys: {len(incompatible.missing_keys)}")
    if incompatible.unexpected_keys:
        print(f"[warm-start] unexpected keys: {len(incompatible.unexpected_keys)}")
    return True


def save_checkpoint(
    config: RVSDBaselineConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    epoch: int,
    global_step: int,
    filename: str = "latest.pt",
) -> Path:
    model = unwrap_model(model)
    checkpoint_path = default_checkpoint_path(config).with_name(filename)
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "meta": {
            "batch_size_correction_v1": True,
        },
    }
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def resolve_checkpoint_path(config: RVSDBaselineConfig, allow_default: bool = True) -> Path | None:
    explicit_candidates: list[str] = []
    if config.run.checkpoint_path:
        explicit_candidates.append(config.run.checkpoint_path)
    if config.optimizer.resume_checkpoint:
        explicit_candidates.append(config.optimizer.resume_checkpoint)

    if explicit_candidates:
        for candidate in explicit_candidates:
            resolved = _resolve_existing_path(candidate, output_dir=config.run.output_dir)
            if resolved.exists():
                return resolved
        return None

    if allow_default:
        candidate = default_checkpoint_path(config)
        if candidate.exists():
            return candidate
    return None


def load_checkpoint_if_available(
    config: RVSDBaselineConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
    scaler=None,
    allow_default: bool = True,
) -> tuple[int, int]:
    model = unwrap_model(model)
    explicit_request = config.run.checkpoint_path or config.optimizer.resume_checkpoint
    checkpoint_path = resolve_checkpoint_path(config, allow_default=allow_default)
    if checkpoint_path is None:
        if explicit_request:
            resolved = _resolve_existing_path(explicit_request, output_dir=config.run.output_dir)
            raise FileNotFoundError(f"Checkpoint not found: {resolved}")
        return 0, 0

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    strict_load = has_explicit_checkpoint(config)

    try:
        model.load_state_dict(checkpoint["model"], strict=True)
    except RuntimeError as exc:
        if strict_load:
            raise RuntimeError(f"Failed to load checkpoint from {checkpoint_path}: {exc}") from exc
        print(f"[checkpoint] Skipping incompatible auto checkpoint at {checkpoint_path}: {exc}")
        return 0, 0
    _maybe_reset_legacy_batch_size_correction(model, checkpoint, prefix="[checkpoint]")

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    print(f"[checkpoint] Loaded checkpoint from {checkpoint_path}")
    return int(checkpoint.get("epoch", 0)), int(checkpoint.get("global_step", 0))


def _capture_trainable_defaults(model: torch.nn.Module) -> None:
    model = unwrap_model(model)
    if hasattr(model, "_risk_v1_default_trainable"):
        return
    model._risk_v1_default_trainable = {
        name: parameter.requires_grad for name, parameter in model.named_parameters()
    }
    model._risk_v1_training_stage = None
    if hasattr(model, "backbone") and hasattr(model.backbone, "adaptive_config"):
        model._risk_v1_target_keep_ratio = float(model.backbone.adaptive_config.fixed_keep_ratio)
        model._risk_v1_target_keep_ratios = tuple(
            float(value)
            for value in (getattr(model.backbone.adaptive_config, "fixed_keep_ratios", ()) or ())
        )


def configure_risk_runtime_schedule(model: torch.nn.Module, config: RVSDBaselineConfig, epoch_index: int) -> None:
    model = unwrap_model(model)
    runtime_mode = config.model.adaptive.runtime_mode.lower()
    if runtime_mode != "risk_v1":
        return

    _capture_trainable_defaults(model)
    adaptive_cfg = model.backbone.adaptive_config
    target_keep_ratio = float(getattr(model, "_risk_v1_target_keep_ratio", config.model.adaptive.fixed_keep_ratio))
    target_keep_ratios = tuple(
        float(value)
        for value in getattr(model, "_risk_v1_target_keep_ratios", ())
    )
    head_only_epochs = max(int(config.model.adaptive.risk_head_only_epochs), 0)
    head_only_keep_ratio = float(config.model.adaptive.risk_head_only_keep_ratio)
    joint_warmup_epochs = max(int(config.model.adaptive.risk_joint_warmup_epochs), 0)
    joint_warmup_keep_ratio = float(config.model.adaptive.risk_joint_warmup_keep_ratio)

    if epoch_index < head_only_epochs:
        scheduled_keep_ratio = max(target_keep_ratio, head_only_keep_ratio)
        schedule_stage = "head_only_safe_keep"
    elif epoch_index < head_only_epochs + joint_warmup_epochs:
        scheduled_keep_ratio = max(target_keep_ratio, joint_warmup_keep_ratio)
        schedule_stage = "joint_warmup_keep"
    else:
        scheduled_keep_ratio = target_keep_ratio
        schedule_stage = "target_keep"

    previous_keep_ratio = getattr(model, "_risk_v1_runtime_keep_ratio", None)
    previous_keep_ratios = getattr(model, "_risk_v1_runtime_keep_ratios", None)
    if target_keep_ratios:
        scheduled_keep_ratios = tuple(max(float(target), scheduled_keep_ratio) for target in target_keep_ratios)
        adaptive_cfg.fixed_keep_ratios = scheduled_keep_ratios
        adaptive_cfg.fixed_keep_ratio = float(scheduled_keep_ratios[-1])
        model._risk_v1_runtime_keep_ratios = scheduled_keep_ratios
    else:
        scheduled_keep_ratios = tuple()
        adaptive_cfg.fixed_keep_ratio = scheduled_keep_ratio
        model._risk_v1_runtime_keep_ratios = tuple()
    model._risk_v1_runtime_keep_ratio = adaptive_cfg.fixed_keep_ratio
    model._risk_v1_runtime_schedule_stage = schedule_stage

    schedule_changed = (
        previous_keep_ratio != adaptive_cfg.fixed_keep_ratio
        or previous_keep_ratios != scheduled_keep_ratios
    )
    if is_main_process() and schedule_changed:
        keep_ratio_message = (
            f"fixed_keep_ratios={list(scheduled_keep_ratios)} "
            f"(target={list(target_keep_ratios)})"
            if scheduled_keep_ratios
            else f"fixed_keep_ratio={scheduled_keep_ratio:.4f} (target={target_keep_ratio:.4f})"
        )
        print(
            f"[risk_v1] runtime schedule="
            f"{schedule_stage} epoch={epoch_index + 1}/{config.optimizer.max_epochs} "
            f"{keep_ratio_message}"
        )


def configure_training_stage(model: torch.nn.Module, config: RVSDBaselineConfig, epoch_index: int) -> None:
    model = unwrap_model(model)
    runtime_mode = config.model.adaptive.runtime_mode.lower()
    if runtime_mode != "risk_v1":
        return

    _capture_trainable_defaults(model)
    head_only_epochs = max(int(config.model.adaptive.risk_head_only_epochs), 0)
    head_only_active = epoch_index < head_only_epochs
    stage_name = "risk_head_only" if head_only_active else "joint_finetune"
    if getattr(model, "_risk_v1_training_stage", None) == stage_name:
        return

    default_trainable = model._risk_v1_default_trainable
    for name, parameter in model.named_parameters():
        should_train = default_trainable.get(name, False)
        if head_only_active:
            should_train = should_train and (
                name.startswith("backbone.group_risk_head.")
                or name.startswith("backbone.group_compressibility_head.")
                or name.startswith("backbone.group_task_preserve_head.")
                or name.startswith("backbone.task_token_shadow_head.")
            )
        parameter.requires_grad_(should_train)

    model._risk_v1_training_stage = stage_name
    if is_main_process():
        trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        print(
            f"[risk_v1] training stage={stage_name} "
            f"epoch={epoch_index + 1}/{config.optimizer.max_epochs} "
            f"trainable_params={trainable_params}"
        )


def build_risk_teacher_model(config: RVSDBaselineConfig, device: torch.device) -> torch.nn.Module | None:
    adaptive_cfg = config.model.adaptive
    if adaptive_cfg.runtime_mode.lower() != "risk_v1" or adaptive_cfg.risk_distill_weight <= 0:
        return None
    teacher_reference = adaptive_cfg.risk_teacher_checkpoint or config.optimizer.warm_start_checkpoint
    if not teacher_reference:
        if is_main_process():
            print("[risk_v1] distillation disabled: no teacher checkpoint configured")
        return None

    teacher_checkpoint = _resolve_existing_path(
        teacher_reference,
        output_dir=config.run.output_dir,
    )
    if not teacher_checkpoint.exists():
        raise FileNotFoundError(f"Risk distillation teacher checkpoint not found: {teacher_checkpoint}")

    teacher_config = deepcopy(config)
    teacher_config.model.adaptive.enable_token_merging = False
    teacher_config.model.adaptive.enable_existence_calibration = False
    teacher_config.model.adaptive.runtime_mode = "accuracy"

    teacher_model = build_vsd_model(teacher_config).to(device)
    checkpoint = torch.load(teacher_checkpoint, map_location="cpu")
    teacher_state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    current_state = teacher_model.state_dict()
    compatible_state = {}
    skipped_shape = []
    skipped_unknown = []
    for key, value in teacher_state.items():
        if key not in current_state:
            skipped_unknown.append(key)
            continue
        if tuple(current_state[key].shape) != tuple(value.shape):
            skipped_shape.append(key)
            continue
        compatible_state[key] = value
    incompatible = teacher_model.load_state_dict(compatible_state, strict=False)
    if not compatible_state:
        raise RuntimeError(
            "[risk_v1] distillation teacher checkpoint has no compatible parameters for the current teacher. "
            f"checkpoint={teacher_checkpoint}"
        )
    teacher_model.eval()
    for parameter in teacher_model.parameters():
        parameter.requires_grad_(False)
    if is_main_process():
        print(f"[risk_v1] distillation teacher loaded from {teacher_checkpoint}")
        if skipped_shape:
            print(f"[risk_v1] teacher skipped shape-mismatched keys: {len(skipped_shape)}")
        if skipped_unknown:
            print(f"[risk_v1] teacher skipped unknown keys: {len(skipped_unknown)}")
        if incompatible.missing_keys:
            print(f"[risk_v1] teacher missing keys: {len(incompatible.missing_keys)}")
        if incompatible.unexpected_keys:
            print(f"[risk_v1] teacher unexpected keys: {len(incompatible.unexpected_keys)}")
    return teacher_model


def build_distillation_teacher_model(config: RVSDBaselineConfig, device: torch.device) -> torch.nn.Module | None:
    if config.optimizer.distill_weight <= 0:
        return None
    teacher_reference = config.optimizer.distill_checkpoint
    if not teacher_reference:
        raise ValueError("optimizer.distill_checkpoint must be set when optimizer.distill_weight > 0")

    teacher_checkpoint = _resolve_existing_path(
        teacher_reference,
        output_dir=config.run.output_dir,
    )
    if not teacher_checkpoint.exists():
        raise FileNotFoundError(f"Distillation teacher checkpoint not found: {teacher_checkpoint}")

    teacher_config = deepcopy(config)
    teacher_config.model.adaptive.enable_token_merging = False
    teacher_config.model.adaptive.enable_existence_calibration = False
    teacher_config.model.adaptive.runtime_mode = "accuracy"

    teacher_model = build_vsd_model(teacher_config).to(device)
    checkpoint = torch.load(teacher_checkpoint, map_location="cpu")
    teacher_state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    current_state = teacher_model.state_dict()
    compatible_state = {}
    skipped_shape = []
    skipped_unknown = []
    for key, value in teacher_state.items():
        if key not in current_state:
            skipped_unknown.append(key)
            continue
        if tuple(current_state[key].shape) != tuple(value.shape):
            skipped_shape.append(key)
            continue
        compatible_state[key] = value
    incompatible = teacher_model.load_state_dict(compatible_state, strict=False)
    if not compatible_state:
        raise RuntimeError(
            "Distillation teacher checkpoint has no compatible parameters for the current model. "
            f"checkpoint={teacher_checkpoint}"
        )
    teacher_model.eval()
    for parameter in teacher_model.parameters():
        parameter.requires_grad_(False)
    if is_main_process():
        print(f"[distill] teacher loaded from {teacher_checkpoint}")
        if skipped_shape:
            print(f"[distill] teacher skipped shape-mismatched keys: {len(skipped_shape)}")
        if skipped_unknown:
            print(f"[distill] teacher skipped unknown keys: {len(skipped_unknown)}")
        if incompatible.missing_keys:
            print(f"[distill] teacher missing keys: {len(incompatible.missing_keys)}")
        if incompatible.unexpected_keys:
            print(f"[distill] teacher unexpected keys: {len(incompatible.unexpected_keys)}")
    return teacher_model


def print_debug(debug: dict[str, object], loss_value: float, prefix: str = "") -> None:
    if not is_main_process():
        return
    if prefix:
        prefix = f"{prefix} "
    print(f"{prefix}input shape: {debug['input_shape']}")
    print(f"{prefix}backbone feature shapes: {debug['backbone_feature_shapes']}")
    print(f"{prefix}fusion output shape: {debug['fusion_output_shape']}")
    print(f"{prefix}head output shape: {debug['head_output_shape']}")
    print(f"{prefix}final logits shape: {debug['final_logits_shape']}")
    if "decoder_variant" in debug:
        print(f"{prefix}decoder variant: {debug['decoder_variant']}")
    if "sparse_decoder_dense_source_indices" in debug:
        print(f"{prefix}sparse decoder dense source indices: {debug['sparse_decoder_dense_source_indices']}")
    if "sparse_decoder_dense_shapes" in debug:
        print(f"{prefix}sparse decoder dense shapes: {debug['sparse_decoder_dense_shapes']}")
    if "sparse_decoder_sparse_tokens_shape" in debug:
        print(f"{prefix}sparse decoder sparse tokens shape: {debug['sparse_decoder_sparse_tokens_shape']}")
    if "sparse_decoder_sparse_coords_shape" in debug:
        print(f"{prefix}sparse decoder sparse coords shape: {debug['sparse_decoder_sparse_coords_shape']}")
    if "sparse_decoder_sparse_sizes_shape" in debug:
        print(f"{prefix}sparse decoder sparse sizes shape: {debug['sparse_decoder_sparse_sizes_shape']}")
    if "sparse_decoder_patch_hw" in debug:
        print(f"{prefix}sparse decoder patch hw: {debug['sparse_decoder_patch_hw']}")
    if "keep_ratios" in debug:
        print(f"{prefix}keep ratios: {debug['keep_ratios']}")
        print(f"{prefix}merge block indices: {debug['merge_block_indices']}")
    if "runtime_mode" in debug:
        print(f"{prefix}runtime mode: {debug['runtime_mode']}")
    if "tail_stop_block_index" in debug:
        print(
            f"{prefix}tail stop block index: {debug['tail_stop_block_index']} "
            f"(skipped tail blocks={debug.get('skipped_tail_blocks', 0)})"
        )
    if "group_contract_extension_enabled" in debug:
        print(f"{prefix}group contract extension enabled: {debug['group_contract_extension_enabled']}")
    if "attention_bias_correction_enabled" in debug:
        print(f"{prefix}attention bias correction enabled: {debug['attention_bias_correction_enabled']}")
    if "risk_hard_gate_ratio" in debug:
        print(f"{prefix}risk hard gate ratio: {debug['risk_hard_gate_ratio']:.4f}")
    if "risk_dynamic_keep_enabled" in debug:
        print(f"{prefix}risk dynamic keep enabled: {debug['risk_dynamic_keep_enabled']}")
    if "risk_dynamic_hard_gate_enabled" in debug:
        print(f"{prefix}risk dynamic hard gate enabled: {debug['risk_dynamic_hard_gate_enabled']}")
    if "stage_merge_stats" in debug:
        for stage_stats in debug["stage_merge_stats"]:
            print(
                f"{prefix}stage {stage_stats['merge_block_index']} groups: "
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
    if torch.cuda.is_available():
        allocated_gb = torch.cuda.memory_allocated() / (1024**3)
        reserved_gb = torch.cuda.memory_reserved() / (1024**3)
        max_allocated_gb = torch.cuda.max_memory_allocated() / (1024**3)
        max_reserved_gb = torch.cuda.max_memory_reserved() / (1024**3)
        print(
            f"{prefix}cuda memory: "
            f"allocated={allocated_gb:.2f}GB "
            f"reserved={reserved_gb:.2f}GB "
            f"max_allocated={max_allocated_gb:.2f}GB "
            f"max_reserved={max_reserved_gb:.2f}GB"
        )
    if "final_token_keep_ratio" in debug:
        keep_ratio = float(debug["final_token_keep_ratio"])
        saved_ratio = float(debug.get("final_token_saved_ratio", 1.0 - keep_ratio))
        print(f"{prefix}final token keep ratio: {keep_ratio:.4f} (saved {saved_ratio:.4f})")
    if "final_interface_token_keep_ratio" in debug:
        print(f"{prefix}final interface token keep ratio: {debug['final_interface_token_keep_ratio']:.4f}")
    if "existence_logits_shape" in debug:
        print(f"{prefix}existence logits shape: {debug['existence_logits_shape']}")
    print(f"{prefix}loss: {loss_value:.6f}")


def save_train_debug_snapshot(
    *,
    config: RVSDBaselineConfig,
    debug: dict[str, object],
    epoch_index: int,
    global_step: int,
    lr: float,
    loss_value: float,
    loss_dict: dict[str, float],
    metric_dict: dict[str, float] | None,
) -> Path:
    summary = build_debug_summary(
        debug,
        requested_keep_ratio=float(config.model.adaptive.fixed_keep_ratio),
        seed=int(config.run.seed),
        context="train_epoch_snapshot",
        extra={
            "epoch": int(epoch_index),
            "global_step": int(global_step),
            "learning_rate": float(lr),
            "loss_total": float(loss_value),
            "loss_items": {key: float(value) for key, value in loss_dict.items()},
            "train_metrics": None if metric_dict is None else {key: float(value) for key, value in metric_dict.items()},
        },
    )
    diagnostics_dir = Path(config.run.output_dir) / "diagnostics"
    return save_debug_summary(diagnostics_dir / f"train_epoch_{epoch_index:02d}_snapshot.json", summary)


def _format_loss_items(loss_dict: dict[str, float]) -> str:
    ordered_names = (
        "loss_main",
        "loss_bce",
        "loss_dice",
        "loss_tversky",
        "loss_lovasz",
        "loss_shadow_body",
        "loss_boundary_main",
        "loss_temporal_consistency",
        "loss_nonshadow_precision",
        "loss_dark_negative",
        "loss_distill",
        "loss_risk",
        "loss_risk_distill",
        "risk_teacher_damage",
        "risk_teacher_error",
        "risk_rank_spearman",
        "risk_topk_overlap",
        "risk_damage_gap",
        "risk_boundary_protect_rate",
        "loss_exist",
        "loss_shadow_free_total",
        "loss_shadow_free_zero_mask",
        "loss_shadow_free_existence_negative",
        "loss_shadow_free_temporal_consistency",
        "loss_shadow_free_confidence_suppression",
        "loss_shadow_free_easy_risk",
    )
    parts = [f"loss_total={loss_dict['loss_total']:.6f}"]
    for name in ordered_names:
        if name in loss_dict:
            parts.append(f"{name}={loss_dict[name]:.6f}")
    return " ".join(parts)


def run_smoke(config: RVSDBaselineConfig, split: str | None = None) -> dict[str, object]:
    split = split or config.run.split_name
    device = resolve_device(config.run.device)
    model = build_vsd_model(config).to(device)
    criterion = build_segmentation_loss(
        lambda_bce=config.optimizer.lambda_bce,
        positive_bce_weight=config.optimizer.positive_bce_weight,
        lambda_dice=config.optimizer.lambda_dice,
        lambda_tversky=config.optimizer.lambda_tversky,
        tversky_alpha=config.optimizer.tversky_alpha,
        tversky_beta=config.optimizer.tversky_beta,
        lambda_lovasz=config.optimizer.lambda_lovasz,
        lambda_shadow_body=config.optimizer.lambda_shadow_body,
        lambda_body_aux=config.optimizer.lambda_body_aux,
        lambda_detail_aux=config.optimizer.lambda_detail_aux,
        lambda_boundary_aux=config.optimizer.lambda_boundary_aux,
        lambda_boundary_main=config.optimizer.lambda_boundary_main,
        lambda_temporal_consistency=config.optimizer.lambda_temporal_consistency,
        lambda_nonshadow_precision=config.optimizer.lambda_nonshadow_precision,
        lambda_dark_negative=config.optimizer.lambda_dark_negative,
        dark_negative_threshold=config.optimizer.dark_negative_threshold,
        dark_negative_boundary_kernel=config.optimizer.dark_negative_boundary_kernel,
        adaptive_config=config.model.adaptive,
    )

    training_split = split == "train"
    dataloader = build_dataloader(config, split=split, training=training_split, shuffle=False)
    batch = next(iter(dataloader))
    clip = batch["clip"].to(device, non_blocking=True)
    mask_clip = batch["mask_clip"].to(device, non_blocking=True)

    model.eval()
    with torch.no_grad():
        logits, outputs = model(clip, return_outputs=True)
        loss, loss_dict = criterion(
            logits,
            mask_clip,
            adaptive_outputs=outputs["adaptive"],
            existence_outputs=outputs["existence"],
            auxiliary_outputs=outputs.get("auxiliary"),
            clip=clip,
        )

    print_debug(outputs["debug"], float(loss.item()))
    return {
        "loss": float(loss.item()),
        "loss_dict": {key: float(value.item()) for key, value in loss_dict.items()},
        "debug": outputs["debug"],
        "meta": batch["meta"],
    }


def _run_step(
    model: torch.nn.Module,
    teacher_model: torch.nn.Module | None,
    criterion,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    clip: torch.Tensor,
    mask_clip: torch.Tensor,
    device: torch.device,
    use_amp: bool,
    grad_clip_norm: float,
    shadow_free_batch: dict[str, object] | None = None,
    shadow_free_criterion=None,
    compute_metrics: bool = True,
) -> tuple[float, dict[str, float], dict[str, float] | None, dict[str, object]]:
    optimizer.zero_grad(set_to_none=True)
    teacher_logits = None
    if teacher_model is not None:
        with torch.no_grad():
            with get_autocast_context(device, use_amp):
                teacher_logits = teacher_model(clip, return_outputs=False)
    with get_autocast_context(device, use_amp):
        logits, outputs = model(clip, return_outputs=True)
        loss, loss_dict = criterion(
            logits,
            mask_clip,
            adaptive_outputs=outputs["adaptive"],
            existence_outputs=outputs["existence"],
            auxiliary_outputs=outputs.get("auxiliary"),
            teacher_logits=teacher_logits,
            clip=clip,
        )
        main_loss = loss
        if shadow_free_batch is not None and shadow_free_criterion is not None:
            shadow_free_clip = shadow_free_batch["clip"].to(device, non_blocking=True)
            shadow_free_logits, shadow_free_outputs = model(shadow_free_clip, return_outputs=True)
            shadow_free_loss, shadow_free_loss_dict = shadow_free_criterion(
                shadow_free_logits,
                existence_outputs=shadow_free_outputs["existence"],
                adaptive_outputs=shadow_free_outputs["adaptive"],
            )
            loss = loss + shadow_free_loss
        else:
            shadow_free_loss = None
            shadow_free_loss_dict = {}

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    if grad_clip_norm > 0:
        torch.nn.utils.clip_grad_norm_([parameter for parameter in model.parameters() if parameter.requires_grad], grad_clip_norm)
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()

    loss_dict = dict(loss_dict)
    if shadow_free_loss is not None:
        loss_dict["loss_main"] = main_loss.detach()
        loss_dict["loss_total"] = loss.detach()
        loss_dict.update(shadow_free_loss_dict)
    detached_losses = {key: float(value.item()) for key, value in loss_dict.items()}
    metric_dict = None
    if compute_metrics:
        with torch.no_grad():
            metric_dict = compute_shadow_metrics(logits.detach(), mask_clip.detach())
    return float(loss.item()), detached_losses, metric_dict, outputs["debug"]


def _load_binary_mask(label_path: str) -> torch.Tensor:
    mask = Image.open(label_path).convert("L")
    array = (np.asarray(mask, dtype=np.float32) > 0).astype(np.float32)
    return torch.from_numpy(array).unsqueeze(0)


@torch.no_grad()
def validate_model(
    config: RVSDBaselineConfig,
    model: torch.nn.Module,
    criterion,
    device: torch.device,
    epoch_index: int | None = None,
) -> dict[str, float]:
    if not is_main_process():
        return {}
    eval_model = unwrap_model(model)

    val_config = config
    if is_distributed() and os.name == "nt" and config.run.num_workers > 0:
        val_config = deepcopy(config)
        val_config.run.num_workers = 0
        if is_main_process():
            print("[val] Windows DDP detected, forcing validation num_workers=0 to avoid dataloader deadlock")

    val_loader = build_dataloader(val_config, split="val", training=False, shuffle=False)
    metric_aggregator = build_soda_metric_aggregator(config)
    val_loss_sum = 0.0
    val_steps = 0
    val_iou_sum = 0.0
    val_iou_count = 0
    aggregate_iou_sum = 0.0
    aggregate_iou_count = 0
    aggregated: dict[str, dict[str, dict[str, object]]] = {}

    eval_model.eval()
    progress_bar = tqdm(
        val_loader,
        total=len(val_loader),
        desc=(
            f"Val {epoch_index}/{config.optimizer.max_epochs}"
            if epoch_index is not None
            else "Val"
        ),
        dynamic_ncols=True,
        leave=True,
        disable=not is_main_process(),
    )
    for batch in progress_bar:
        clip = batch["clip"].to(device, non_blocking=True)
        mask_clip = batch["mask_clip"].to(device, non_blocking=True)
        with get_autocast_context(device, config.optimizer.use_amp):
            logits, outputs = eval_model(clip, return_outputs=True)
            loss, _loss_dict = criterion(
                logits,
                mask_clip,
                adaptive_outputs=outputs["adaptive"],
                existence_outputs=outputs["existence"],
                auxiliary_outputs=outputs.get("auxiliary"),
                clip=clip,
            )

        val_loss_sum += float(loss.item())
        val_steps += 1
        batch_metrics = compute_shadow_metrics(logits.detach(), mask_clip.detach())
        val_iou_sum += batch_metrics["iou"]
        val_iou_count += 1
        progress_bar.set_postfix(
            loss=f"{val_loss_sum / val_steps:.4f}",
            iou=f"{val_iou_sum / max(val_iou_count, 1):.4f}",
        )

        for batch_index, meta in enumerate(batch["meta"]):
            video_id = meta["video_id"]
            video_store = aggregated.setdefault(video_id, {})
            for time_index, frame_name in enumerate(meta["frame_names"]):
                if meta["is_padding"][time_index]:
                    continue
                if config.dataset.dummy_mode:
                    frame_logit = logits[batch_index, time_index].detach().cpu()
                    frame_target = mask_clip[batch_index, time_index].detach().cpu().to(frame_logit.dtype)
                    metric_aggregator.update(frame_logit, frame_target)
                    aggregate_metrics = compute_shadow_metrics(
                        frame_logit.unsqueeze(0),
                        frame_target.unsqueeze(0),
                        threshold=config.run.threshold,
                    )
                    aggregate_iou_sum += aggregate_metrics["iou"]
                    aggregate_iou_count += 1
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
                        "label_path": meta["label_paths"][time_index],
                    },
                )
                frame_store["logit_sum"] += frame_logit
                frame_store["count"] += 1
    progress_bar.close()

    if not config.dataset.dummy_mode:
        for video_predictions in aggregated.values():
            for prediction in video_predictions.values():
                logit = prediction["logit_sum"] / max(prediction["count"], 1)
                target = _load_binary_mask(prediction["label_path"]).to(logit.dtype)
                if tuple(target.shape[-2:]) != tuple(logit.shape[-2:]):
                    target = F.interpolate(target.unsqueeze(0), size=logit.shape[-2:], mode="nearest").squeeze(0)
                metric_aggregator.update(logit, target)
                aggregate_metrics = compute_shadow_metrics(logit.unsqueeze(0), target.unsqueeze(0), threshold=config.run.threshold)
                aggregate_iou_sum += aggregate_metrics["iou"]
                aggregate_iou_count += 1

    metrics = metric_aggregator.summarize()
    metrics["loss"] = val_loss_sum / max(val_steps, 1)
    metrics["iou"] = aggregate_iou_sum / max(aggregate_iou_count, 1)
    return metrics


def train_model(config: RVSDBaselineConfig, overfit_one_batch: bool = False) -> dict[str, object]:
    device, rank, world_size = init_distributed_mode(config.run.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = bool(config.run.enable_cudnn_benchmark)
    Path(config.run.output_dir).mkdir(parents=True, exist_ok=True)

    model = build_vsd_model(config).to(device)
    teacher_model = build_distillation_teacher_model(config, device) or build_risk_teacher_model(config, device)
    criterion = build_segmentation_loss(
        lambda_bce=config.optimizer.lambda_bce,
        positive_bce_weight=config.optimizer.positive_bce_weight,
        lambda_dice=config.optimizer.lambda_dice,
        lambda_tversky=config.optimizer.lambda_tversky,
        tversky_alpha=config.optimizer.tversky_alpha,
        tversky_beta=config.optimizer.tversky_beta,
        lambda_lovasz=config.optimizer.lambda_lovasz,
        lambda_shadow_body=config.optimizer.lambda_shadow_body,
        lambda_body_aux=config.optimizer.lambda_body_aux,
        lambda_detail_aux=config.optimizer.lambda_detail_aux,
        lambda_boundary_aux=config.optimizer.lambda_boundary_aux,
        lambda_boundary_main=config.optimizer.lambda_boundary_main,
        lambda_temporal_consistency=config.optimizer.lambda_temporal_consistency,
        lambda_nonshadow_precision=config.optimizer.lambda_nonshadow_precision,
        lambda_dark_negative=config.optimizer.lambda_dark_negative,
        dark_negative_threshold=config.optimizer.dark_negative_threshold,
        dark_negative_boundary_kernel=config.optimizer.dark_negative_boundary_kernel,
        distill_weight=config.optimizer.distill_weight,
        distill_boundary_weight=config.optimizer.distill_boundary_weight,
        distill_bce_weight=config.optimizer.distill_bce_weight,
        distill_dice_weight=config.optimizer.distill_dice_weight,
        distill_logit_weight=config.optimizer.distill_logit_weight,
        adaptive_config=config.model.adaptive,
    )
    shadow_free_loader = None
    shadow_free_iterator_holder: dict[str, object] = {}
    shadow_free_criterion = None
    shadow_free_enabled = bool(config.shadow_free_aux.enabled)
    if shadow_free_enabled:
        shadow_free_loader = build_shadow_free_dataloader(config, training=True, shuffle=True)
        shadow_free_criterion = build_shadow_free_aux_loss(config)
    tensorboard_writer = _build_tensorboard_writer(config)
    wandb_run = _build_wandb_run(config)
    warm_start_model_if_available(config, model)
    if is_main_process():
        print(
            "[train-config] "
            f"seed={config.run.seed} "
            f"output_dir={config.run.output_dir} "
            f"batch_size={config.run.batch_size} "
            f"num_workers={config.run.num_workers} "
            f"prefetch_factor={config.run.prefetch_factor} "
            f"persistent_workers={config.run.persistent_workers} "
            f"metric_hard_threshold={config.run.metric_hard_threshold:.6f} "
            f"metric_mae_threshold={config.run.metric_mae_threshold:.6f} "
            f"metric_use_soft_mae={config.run.metric_use_soft_mae} "
            f"positive_bce_weight={config.optimizer.positive_bce_weight:.4f} "
            f"lambda_tversky={config.optimizer.lambda_tversky:.4f} "
            f"tversky_alpha={config.optimizer.tversky_alpha:.4f} "
            f"tversky_beta={config.optimizer.tversky_beta:.4f} "
            f"lambda_nonshadow_precision={config.optimizer.lambda_nonshadow_precision:.4f} "
            f"lambda_dark_negative={config.optimizer.lambda_dark_negative:.4f} "
            f"dark_negative_threshold={config.optimizer.dark_negative_threshold:.4f} "
            f"dark_negative_boundary_kernel={config.optimizer.dark_negative_boundary_kernel} "
            f"decoder_variant={config.model.head.decoder_variant} "
            f"decoder_token_dim={config.model.head.decoder_token_dim} "
            f"decoder_token_stride={config.model.head.decoder_token_sparse_coarse_stride} "
            f"runtime_mode={config.model.adaptive.runtime_mode} "
            f"merge_block_index={config.model.adaptive.merge_block_index} "
            f"tail_stop_block_index={config.model.adaptive.tail_stop_block_index} "
            f"fixed_keep_ratio={config.model.adaptive.fixed_keep_ratio:.4f} "
            f"fixed_keep_ratios={list(config.model.adaptive.fixed_keep_ratios)} "
            f"fixed_keep_ratio_is_token_keep={config.model.adaptive.fixed_keep_ratio_is_token_keep} "
            f"risk_hard_gate_ratio={config.model.adaptive.risk_hard_gate_ratio:.4f} "
            f"use_group_contract_extension={config.model.adaptive.use_group_contract_extension} "
            f"existence_calibration={config.model.adaptive.enable_existence_calibration} "
            f"shadow_free_aux={config.shadow_free_aux.enabled}"
        )
        print(
            "[train-config] "
            f"warm_start_checkpoint={config.optimizer.warm_start_checkpoint or '<none>'}"
        )
        print(
            "[train-config] "
            f"distill_checkpoint={config.optimizer.distill_checkpoint or '<none>'} "
            f"distill_weight={config.optimizer.distill_weight:.4f} "
            f"distill_boundary_weight={config.optimizer.distill_boundary_weight:.4f} "
            f"distill_bce_weight={config.optimizer.distill_bce_weight:.4f} "
            f"distill_dice_weight={config.optimizer.distill_dice_weight:.4f} "
            f"distill_logit_weight={config.optimizer.distill_logit_weight:.4f}"
        )
        print(
            "[train-config] "
            f"risk_head_only_epochs={config.model.adaptive.risk_head_only_epochs} "
            f"risk_joint_warmup_epochs={config.model.adaptive.risk_joint_warmup_epochs} "
            f"risk_joint_warmup_keep_ratio={config.model.adaptive.risk_joint_warmup_keep_ratio:.4f} "
            f"risk_distill_weight={config.model.adaptive.risk_distill_weight:.4f} "
            f"risk_distill_bce_weight={config.model.adaptive.risk_distill_bce_weight:.4f} "
            f"risk_distill_dice_weight={config.model.adaptive.risk_distill_dice_weight:.4f} "
            f"risk_distill_logit_weight={config.model.adaptive.risk_distill_logit_weight:.4f} "
            f"risk_relative_rank_weight={config.model.adaptive.risk_relative_rank_weight:.4f} "
            f"risk_dynamic_keep_enabled={config.model.adaptive.risk_dynamic_keep_enabled} "
            f"risk_dynamic_hard_gate_enabled={config.model.adaptive.risk_dynamic_hard_gate_enabled}"
        )
        if shadow_free_enabled:
            print(
                "[train-config] "
                f"shadow_free_root={config.shadow_free_aux.dataset_root or '<unset>'} "
                f"shadow_free_sample_ratio={config.shadow_free_aux.sample_ratio:.4f} "
                f"shadow_free_lambda_total={config.shadow_free_aux.lambda_total:.4f} "
                f"shadow_free_batch_size={config.shadow_free_aux.batch_size if config.shadow_free_aux.batch_size > 0 else config.run.batch_size}"
            )
        if config.model.adaptive.risk_dynamic_keep_enabled:
            keep_min = float(config.model.adaptive.risk_dynamic_keep_min_ratio)
            keep_max = float(config.model.adaptive.risk_dynamic_keep_max_ratio)
            target_keep = float(config.model.adaptive.fixed_keep_ratio)
            if not (keep_min <= target_keep <= keep_max):
                print(
                    "[train-config] "
                    f"warning: fixed_keep_ratio={target_keep:.4f} is outside "
                    f"dynamic_keep_range=[{keep_min:.4f}, {keep_max:.4f}]; "
                    "runtime will auto-expand the range to include the base keep ratio."
                )
    if world_size > 1:
        model = DDP(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            output_device=device.index if device.type == "cuda" else None,
            find_unused_parameters=True,
        )
    optimizer = build_optimizer(model, config)

    train_sampler = None
    train_loader_template = build_dataloader(config, split="train", training=True, shuffle=False)
    if world_size > 1 and not overfit_one_batch:
        train_dataset = train_loader_template.dataset
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=False,
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.run.batch_size,
            shuffle=False,
            sampler=train_sampler,
            num_workers=config.run.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=config.run.num_workers > 0 and bool(config.run.persistent_workers),
            prefetch_factor=config.run.prefetch_factor if config.run.num_workers > 0 else None,
            collate_fn=train_loader_template.collate_fn,
        )
    else:
        train_loader = build_dataloader(config, split="train", training=True, shuffle=not overfit_one_batch)
    steps_per_epoch = max(len(train_loader), 1)
    total_steps = config.optimizer.max_steps if config.optimizer.max_steps > 0 else config.optimizer.max_epochs * steps_per_epoch
    total_steps = max(total_steps, 1)
    scheduler = build_scheduler(
        optimizer=optimizer,
        total_steps=total_steps,
        warmup_steps=config.optimizer.warmup_steps,
        min_lr_ratio=config.optimizer.min_lr_ratio,
    )
    scaler = torch.amp.GradScaler(device.type, enabled=config.optimizer.use_amp and device.type == "cuda")
    best_metric_value = float("-inf")
    extra_best_metric_values = {
        str(metric_name): float("-inf")
        for metric_name in tuple(config.optimizer.extra_best_metric_names)
    }

    start_epoch, global_step = load_checkpoint_if_available(
        config,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        allow_default=(not overfit_one_batch) and config.optimizer.auto_resume,
    )
    configure_risk_runtime_schedule(model, config, start_epoch)
    configure_training_stage(model, config, start_epoch)

    if overfit_one_batch:
        batch = next(iter(train_loader))
        clip = batch["clip"].to(device, non_blocking=True)
        mask_clip = batch["mask_clip"].to(device, non_blocking=True)
        model.train()
        for iteration in range(global_step, config.optimizer.overfit_steps):
            loss_value, loss_dict, metric_dict, debug = _run_step(
                model=model,
                teacher_model=teacher_model,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                clip=clip,
                mask_clip=mask_clip,
                device=device,
                use_amp=config.optimizer.use_amp,
                grad_clip_norm=config.optimizer.grad_clip_norm,
                compute_metrics=True,
            )
            if iteration == global_step and is_main_process():
                print_debug(debug, loss_value, prefix="[overfit]")
            if ((iteration + 1) % max(config.optimizer.log_interval, 1) == 0 or iteration == global_step) and is_main_process():
                print(
                    f"[overfit] step={iteration + 1}/{config.optimizer.overfit_steps} "
                    f"iou={(metric_dict['iou'] if metric_dict is not None else 0.0):.6f} "
                    f"dice={(metric_dict['dice'] if metric_dict is not None else 0.0):.6f} "
                    f"{_format_loss_items(loss_dict)}"
                )
                tracking_values = dict(loss_dict)
                if metric_dict is not None:
                    tracking_values.update({f"metric_{key}": value for key, value in metric_dict.items()})
                _log_scalar_dict(tensorboard_writer, wandb_run, "overfit", tracking_values, iteration + 1)
        if config.optimizer.save_checkpoint:
            save_checkpoint(config, model, optimizer, scheduler, scaler, epoch=0, global_step=config.optimizer.overfit_steps)
        if tensorboard_writer is not None:
            tensorboard_writer.close()
        if wandb_run is not None:
            wandb_run.finish()
        barrier()
        return {"global_step": config.optimizer.overfit_steps}

    for epoch in range(start_epoch, config.optimizer.max_epochs):
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        configure_risk_runtime_schedule(model, config, epoch)
        configure_training_stage(model, config, epoch)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        epoch_loss_sum = 0.0
        epoch_iou_sum = 0.0
        epoch_dice_sum = 0.0
        epoch_steps = 0
        epoch_metric_steps = 0
        last_iou = 0.0
        last_dice = 0.0
        progress_bar = tqdm(
            train_loader,
            total=len(train_loader),
            desc=f"Epoch {epoch + 1}/{config.optimizer.max_epochs}",
            dynamic_ncols=True,
            leave=True,
            disable=not is_main_process(),
        )
        for batch in progress_bar:
            clip = batch["clip"].to(device, non_blocking=True)
            mask_clip = batch["mask_clip"].to(device, non_blocking=True)
            next_step = global_step + 1
            should_compute_metrics = next_step == 1 or next_step % max(config.optimizer.train_metric_interval, 1) == 0
            shadow_free_batch = None
            if shadow_free_enabled and should_use_shadow_free_step(config.shadow_free_aux.sample_ratio, global_step):
                shadow_free_batch = next_shadow_free_batch(shadow_free_loader, shadow_free_iterator_holder)
            loss_value, loss_dict, metric_dict, debug = _run_step(
                model=model,
                teacher_model=teacher_model,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                clip=clip,
                mask_clip=mask_clip,
                device=device,
                use_amp=config.optimizer.use_amp,
                grad_clip_norm=config.optimizer.grad_clip_norm,
                shadow_free_batch=shadow_free_batch,
                shadow_free_criterion=shadow_free_criterion,
                compute_metrics=should_compute_metrics,
            )
            global_step += 1
            epoch_steps += 1
            epoch_loss_sum += loss_dict["loss_total"]
            if metric_dict is not None:
                epoch_iou_sum += metric_dict["iou"]
                epoch_dice_sum += metric_dict["dice"]
                epoch_metric_steps += 1
                last_iou = metric_dict["iou"]
                last_dice = metric_dict["dice"]
            if is_main_process():
                progress_bar.set_postfix(
                    loss=f"{epoch_loss_sum / epoch_steps:.4f}",
                    iou=f"{last_iou:.4f}",
                    dice=f"{last_dice:.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                )
                if epoch_steps == 1:
                    print_debug(debug, loss_value, prefix=f"[train {epoch + 1}][debug]")
                    snapshot_path = save_train_debug_snapshot(
                        config=config,
                        debug=debug,
                        epoch_index=epoch + 1,
                        global_step=global_step,
                        lr=float(optimizer.param_groups[0]["lr"]),
                        loss_value=float(loss_value),
                        loss_dict=loss_dict,
                        metric_dict=metric_dict,
                    )
                    print(f"[train {epoch + 1}][debug] snapshot saved to {snapshot_path}")
                    print(
                        f"[train] epoch={epoch + 1}/{config.optimizer.max_epochs} "
                        f"step={global_step}/{total_steps} "
                        f"lr={optimizer.param_groups[0]['lr']:.2e} "
                        f"iou={(metric_dict['iou'] if metric_dict is not None else 0.0):.6f} "
                        f"dice={(metric_dict['dice'] if metric_dict is not None else 0.0):.6f} "
                        f"{_format_loss_items(loss_dict)}"
                    )
            if global_step >= total_steps:
                break
        progress_bar.close()
        epoch_loss_avg = reduce_float(epoch_loss_sum / max(epoch_steps, 1), device)
        epoch_iou_avg = reduce_float(epoch_iou_sum / max(epoch_metric_steps, 1), device)
        epoch_dice_avg = reduce_float(epoch_dice_sum / max(epoch_metric_steps, 1), device)
        if epoch_steps > 0 and is_main_process():
            print(
                f"[epoch {epoch + 1}] "
                f"loss={epoch_loss_avg:.6f} "
                f"iou={epoch_iou_avg:.6f} "
                f"dice={epoch_dice_avg:.6f}"
            )
            if device.type == "cuda":
                peak_allocated_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
                peak_reserved_gb = torch.cuda.max_memory_reserved(device) / (1024**3)
                print(
                    f"[epoch {epoch + 1}] "
                    f"cuda_peak_allocated={peak_allocated_gb:.2f}GB "
                    f"cuda_peak_reserved={peak_reserved_gb:.2f}GB"
                )
            _log_scalar_dict(
                tensorboard_writer,
                wandb_run,
                "train_epoch",
                {
                    "loss": epoch_loss_avg,
                    "iou": epoch_iou_avg,
                    "dice": epoch_dice_avg,
                    "lr": float(optimizer.param_groups[0]["lr"]),
                },
                global_step,
            )

        val_metrics: dict[str, float] | None = None
        if config.optimizer.run_validation and (epoch + 1) % max(config.optimizer.validation_interval, 1) == 0:
            if is_main_process():
                val_metrics = validate_model(config, model, criterion, device, epoch_index=epoch + 1)
                print(
                    f"[val {epoch + 1}] "
                    f"loss={val_metrics['loss']:.6f} "
                    f"iou={val_metrics['iou']:.6f} "
                    f"fmeasure={val_metrics['fmeasure']:.6f} "
                    f"mae={val_metrics['mae']:.6f} "
                    f"jaccard={val_metrics['jaccard']:.6f} "
                    f"ber={val_metrics['ber']:.6f} "
                    f"s_ber={val_metrics['s_ber']:.6f} "
                    f"n_ber={val_metrics['n_ber']:.6f}"
                )
                _log_scalar_dict(tensorboard_writer, wandb_run, "val", val_metrics, global_step)
                metric_name = config.optimizer.best_metric_name
                metric_value = compute_best_metric_value(val_metrics, metric_name)

                warmup_skip_epochs = 0
                if config.model.adaptive.runtime_mode.lower() == "risk_v1":
                    warmup_skip_epochs = max(int(config.model.adaptive.risk_head_only_epochs), 0) + max(
                        int(config.model.adaptive.risk_joint_warmup_epochs), 0
                    )
                best_update_enabled = (epoch + 1) > warmup_skip_epochs

                if best_update_enabled:
                    if metric_value > best_metric_value and config.optimizer.save_checkpoint:
                        best_metric_value = metric_value
                        best_path = save_checkpoint(
                            config,
                            model,
                            optimizer,
                            scheduler,
                            scaler,
                            epoch=epoch + 1,
                            global_step=global_step,
                            filename="best.pt",
                        )
                        print(f"[val {epoch + 1}] best checkpoint updated: {best_path} ({metric_name}={metric_value:.6f})")
                    elif metric_value > best_metric_value:
                        best_metric_value = metric_value
                    if config.optimizer.save_checkpoint and config.optimizer.save_extra_best_checkpoints:
                        for extra_metric_name in extra_best_metric_values:
                            extra_metric_value = compute_best_metric_value(val_metrics, extra_metric_name)
                            if extra_metric_value <= extra_best_metric_values[extra_metric_name]:
                                continue
                            extra_best_metric_values[extra_metric_name] = extra_metric_value
                            extra_path = save_checkpoint(
                                config,
                                model,
                                optimizer,
                                scheduler,
                                scaler,
                                epoch=epoch + 1,
                                global_step=global_step,
                                filename=best_checkpoint_filename(extra_metric_name),
                            )
                            print(
                                f"[val {epoch + 1}] extra best checkpoint updated: "
                                f"{extra_path} ({extra_metric_name}={extra_metric_value:.6f})"
                            )
                elif is_main_process():
                    print(
                        f"[val {epoch + 1}] best checkpoint update skipped during risk_v1 warm-up "
                        f"(eligible from epoch {warmup_skip_epochs + 1})"
                    )
            barrier()

        if config.optimizer.save_checkpoint and is_main_process():
            save_checkpoint(config, model, optimizer, scheduler, scaler, epoch=epoch + 1, global_step=global_step)
        barrier()
        if global_step >= total_steps:
            break

    if tensorboard_writer is not None:
        tensorboard_writer.close()
    if wandb_run is not None:
        wandb_run.finish()
    if is_distributed():
        dist.destroy_process_group()
    return {"global_step": global_step}
