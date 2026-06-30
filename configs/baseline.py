from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[2]


DEFAULT_DATASET_ROOT = str(workspace_root() / "data" / "visha")
DEFAULT_WEIGHTS_PATH = str(workspace_root() / "weights" / "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth")
DEFAULT_OUTPUT_DIR = str(workspace_root() / "outputs" / "baseline")
DEFAULT_CHECKPOINT_DIR = str(Path(DEFAULT_OUTPUT_DIR) / "checkpoints")


@dataclass
class DatasetConfig:
    # Data paths and clip construction.
    dataset_root: str = DEFAULT_DATASET_ROOT
    train_split_name: str = "train"
    test_split_name: str = "test"
    image_dir_name: str = "images"
    label_dir_name: str = "labels"
    clip_length: int = 4
    stride: int = 1
    image_size: tuple[int, int] = (512, 512)

    # Debugging / sampling shortcuts.
    dummy_mode: bool = False
    dummy_num_samples: int = 8
    max_videos: int | None = None
    max_clips_per_video: int | None = None
    train_video_split_ratio: float = 0.8
    train_video_split_seed: int = 42

    # Clip-consistent augmentation.
    hflip_prob: float = 0.5
    color_jitter_prob: float = 0.8
    color_jitter_brightness: float = 0.1
    color_jitter_contrast: float = 0.1
    color_jitter_saturation: float = 0.05
    color_jitter_hue: float = 0.02


@dataclass
class ShadowFreeAuxConfig:
    # Optional training-only shadow-free auxiliary branch.
    enabled: bool = False
    dataset_root: str = ""
    split_name: str = "train"
    image_dir_name: str = "images"
    clip_length: int = 4
    stride: int = 1
    image_size: tuple[int, int] = (512, 512)

    # Data loading overrides. Non-positive values fall back to run.* defaults.
    batch_size: int = 0
    num_workers: int = -1
    prefetch_factor: int = 0
    persistent_workers: bool | None = None

    # Debugging / sampling shortcuts.
    dummy_mode: bool = False
    dummy_num_samples: int = 8
    max_videos: int | None = None
    max_clips_per_video: int | None = None
    train_video_split_ratio: float = 0.8
    train_video_split_seed: int = 42
    sample_ratio: float = 0.25

    # Clip-consistent augmentation.
    hflip_prob: float = 0.5
    color_jitter_prob: float = 0.8
    color_jitter_brightness: float = 0.1
    color_jitter_contrast: float = 0.1
    color_jitter_saturation: float = 0.05
    color_jitter_hue: float = 0.02

    # Auxiliary loss weights.
    lambda_total: float = 0.2
    lambda_zero_mask: float = 1.0
    lambda_existence_negative: float = 0.5
    lambda_temporal_consistency: float = 0.1
    lambda_confidence_suppression: float = 0.1
    lambda_easy_risk: float = 0.05

    # Shadow-free calibration targets.
    confidence_margin: float = 0.05
    easy_risk_score_target: float = 0.1


@dataclass
class ShadowExistenceCalibrationConfig:
    # Standalone frame-wise calibration branch kept for SEC experiments.
    enabled: bool = False
    frame_wise: bool = True
    hidden_dim: int = 256
    temporal_kernel_size: int = 3
    dropout: float = 0.0
    clip_aggregation: str = "max"  # max | mean | logsumexp

    # Inference-time soft correction settings.
    inference_enabled: bool = True
    use_soft_logit_correction: bool = True
    logit_correction_scale: float = 1.0
    logit_correction_power: float = 1.0
    use_keep_ratio_adjustment: bool = True
    keep_ratio_min_scale: float = 0.75
    keep_ratio_max_scale: float = 1.00
    keep_ratio_adjustment_power: float = 1.0

    # Training weights.
    lambda_frame_existence: float = 1.0
    lambda_clip_existence: float = 0.5
    lambda_no_shadow_suppression: float = 0.25
    lambda_calibration_consistency: float = 0.1


@dataclass
class BackboneConfig:
    # Backbone construction and fine-tuning policy.
    weights_path: str = DEFAULT_WEIGHTS_PATH
    out_indices: tuple[int, int, int, int] = (2, 5, 8, 11)
    use_backbone_norm: bool = True
    train_policy: str = "last_n_blocks"  # frozen | last_n_blocks | full
    train_last_n_blocks: int = 2


@dataclass
class FusionConfig:
    # Temporal fusion depth and behavior.
    variant: str = "conv"  # conv | context_attn
    kernel_size: int = 3
    use_residual: bool = True
    use_gating: bool = True
    num_layers_per_scale: int = 2
    attention_dim: int = 128
    attention_heads: int = 4
    attention_dropout: float = 0.0


@dataclass
class HeadConfig:
    # Sparse token decoder capacity.
    decoder_variant: str = "decoder_token_pure"
    decoder_dim: int = 256
    decoder_detail_refine_layers: int = 0
    decoder_context_refine_dilation: int = 2
    decoder_simple_dim: int = 192
    decoder_simple_num_dense_sources: int = 2
    decoder_simple_sparse_coarse_stride: int = 2
    decoder_simple_sparse_temporal_layers: int = 1
    decoder_token_dim: int = 192
    decoder_token_sparse_coarse_stride: int = 2
    decoder_token_sparse_temporal_layers: int = 1
    decoder_token_use_dark_guidance: bool = True
    decoder_token_use_structure_decoupling: bool = True
    decoder_token_use_structure_aux: bool = True
    output_dim: int = 128
    use_batchnorm: bool = True


@dataclass
class AdaptiveComputeConfig:
    # Sparse-decoder mainline defaults.
    enable_token_merging: bool = False
    enable_existence_calibration: bool = False
    runtime_mode: str = "accuracy"  # accuracy | risk_v1

    # Merge schedule.
    merge_block_index: int = 6
    merge_block_indices: tuple[int, ...] = ()
    tail_stop_block_index: int = -1
    risk_keep_one_bypass_enabled: bool = True
    risk_keep_one_bypass_threshold: float = 0.999
    risk_persistent_commit_keep_threshold: float = 0.90
    risk_persistent_fallback_to_transient: bool = True

    # Merge granularity and keep-ratio settings.
    group_size: tuple[int, int] = (2, 2)
    fixed_keep_ratio: float = 0.85
    # Optional per-stage keep ratios, ordered by ascending merge block index.
    # When empty, every merge stage uses fixed_keep_ratio.
    fixed_keep_ratios: tuple[float, ...] = ()
    fixed_keep_ratio_is_token_keep: bool = False

    # Small heads used by the adaptive-compute branch.
    token_label_hidden_dim: int = 256
    existence_hidden_dim: int = 256
    risk_hidden_dim: int = 256
    compressibility_hidden_dim: int = 256
    task_preserve_hidden_dim: int = 256

    # Teacher weighting.
    teacher_uncertainty_weight: float = 1.0
    teacher_temporal_weight: float = 1.0
    teacher_ambiguity_weight: float = 1.0
    ambiguity_kernel_size: int = 5

    # Auxiliary loss weights.
    lambda_selection: float = 0.03
    lambda_exist: float = 0.1
    lambda_exist_negative: float = 0.05
    lambda_exist_calibration: float = 0.05
    ranking_margin: float = 0.05

    # Merge behavior controls.
    use_weighted_merge: bool = True
    use_size_aware_correction: bool = True
    use_size_bias: bool = False
    risk_inference_fast_path: bool = True
    compile_risk_head: bool = False
    use_attention_bias_correction: bool = False
    attention_bias_correction_scale: float = 1.0
    use_merge_correction: bool = False
    merge_correction_hidden_dim: int = 256
    merge_correction_scale: float = 1.0
    merge_correction_bias_scale: float = 1.0
    use_independent_compressibility_predictor: bool = False
    compressibility_score_scale: float = 1.0
    compressibility_compactness_scale: float = 1.0
    compressibility_temporal_scale: float = 0.5
    use_task_aware_preservation: bool = False
    task_preserve_score_scale: float = 1.0
    task_preserve_boundary_scale: float = 1.0
    task_preserve_penumbra_scale: float = 0.75
    task_preserve_temporal_scale: float = 0.75
    task_preserve_dark_ambiguity_scale: float = 0.5
    task_preserve_dark_threshold: float = 0.45
    risk_feature_prior_scale: float = 0.5
    use_token_transform_reducer: bool = False
    token_transform_hidden_dim: int = 256
    token_transform_coord_scale: float = 1.0
    token_transform_size_scale: float = 1.0
    use_temporal_consensus_correction: bool = False
    temporal_consensus_hidden_dim: int = 256
    temporal_consensus_scale: float = 1.0
    temporal_consensus_stride: int = 4
    use_source_target_matching: bool = True
    use_boundary_protection: bool = True
    use_group_contract_extension: bool = True
    use_fused_mask_contract: bool = False
    matching_similarity_threshold: float = 0.3
    matching_topk: int = 4
    size_correction_scale: float = 1.0
    merge_protection_uncertainty_threshold: float = 0.32
    merge_protection_boundary_threshold: float = 0.18
    merge_protection_shadow_low: float = 0.08
    merge_protection_shadow_high: float = 0.92
    risk_hard_gate_ratio: float = 0.05
    risk_teacher_damage_weight: float = 5.0
    risk_teacher_error_weight: float = 0.05
    risk_teacher_uncertainty_weight: float = 0.05
    risk_teacher_temporal_weight: float = 0.05
    risk_teacher_boundary_weight: float = 0.15
    risk_boundary_loss_weight: float = 2.0
    risk_head_only_epochs: int = 0
    risk_head_only_keep_ratio: float = 1.0
    risk_joint_warmup_epochs: int = 1
    risk_joint_warmup_keep_ratio: float = 1.0
    risk_distill_weight: float = 0.05
    risk_distill_boundary_weight: float = 2.0
    risk_distill_bce_weight: float = 1.0
    risk_distill_dice_weight: float = 0.5
    risk_distill_logit_weight: float = 0.0
    risk_teacher_checkpoint: str = ""
    risk_relative_rank_weight: float = 0.0
    risk_relative_rank_ratio: float = 0.05
    risk_debug_topk_ratio: float = 0.05
    risk_dynamic_keep_enabled: bool = False
    risk_dynamic_keep_min_ratio: float = 0.75
    risk_dynamic_keep_max_ratio: float = 1.0
    risk_dynamic_keep_reference: float = 0.5
    risk_dynamic_keep_scale: float = 0.15
    risk_dynamic_keep_easy_scale: float = 0.15
    risk_dynamic_keep_hard_scale: float = 0.15
    risk_dynamic_keep_topk_ratio: float = 0.05
    risk_dynamic_keep_temporal_scale: float = 0.0
    risk_dynamic_keep_momentum: float = 0.0
    risk_dynamic_keep_step_limit: float = 0.0
    risk_dynamic_hard_gate_enabled: bool = False
    risk_dynamic_hard_gate_min_ratio: float = 0.02
    risk_dynamic_hard_gate_max_ratio: float = 0.05
    risk_dynamic_hard_gate_reference: float = 0.5
    risk_dynamic_hard_gate_scale: float = 0.03
    risk_dynamic_hard_gate_easy_scale: float = 0.03
    risk_dynamic_hard_gate_hard_scale: float = 0.03
    risk_dynamic_hard_gate_topk_ratio: float = 0.05
    risk_merge_strategy: str = "low_risk"  # low_risk/ours | random | high_risk | cam | gradient


@dataclass
class ModelConfig:
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    head: HeadConfig = field(default_factory=HeadConfig)
    adaptive: AdaptiveComputeConfig = field(default_factory=AdaptiveComputeConfig)


@dataclass
class OptimizerConfig:
    # Core optimization hyperparameters.
    learning_rate: float = 1e-4
    backbone_learning_rate: float = 2e-5
    weight_decay: float = 1e-2
    max_epochs: int = 10
    max_steps: int = 0
    warmup_steps: int = 50
    min_lr_ratio: float = 0.1
    grad_clip_norm: float = 1.0
    use_amp: bool = True

    # Checkpointing and logging.
    save_checkpoint: bool = True
    checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR
    resume_checkpoint: str = ""
    auto_resume: bool = False
    warm_start_checkpoint: str = ""
    log_interval: int = 100
    train_metric_interval: int = 20
    run_validation: bool = True
    validation_interval: int = 1
    best_metric_name: str = "jaccard"
    save_extra_best_checkpoints: bool = False
    extra_best_metric_names: tuple[str, ...] = ("jaccard", "ber", "s_ber", "soda_balanced")

    # Segmentation loss weights.
    lambda_bce: float = 1.0
    positive_bce_weight: float = 1.0
    lambda_dice: float = 1.0
    lambda_tversky: float = 0.0
    tversky_alpha: float = 0.3
    tversky_beta: float = 0.7
    lambda_lovasz: float = 0.0
    lambda_shadow_body: float = 0.0
    lambda_body_aux: float = 0.0
    lambda_detail_aux: float = 0.0
    lambda_boundary_aux: float = 0.0
    lambda_boundary_main: float = 0.0
    lambda_temporal_consistency: float = 0.0
    lambda_nonshadow_precision: float = 0.0
    lambda_dark_negative: float = 0.0
    dark_negative_threshold: float = 0.35
    dark_negative_boundary_kernel: int = 9

    # Dense-teacher distillation. Used only during training.
    distill_checkpoint: str = ""
    distill_weight: float = 0.0
    distill_boundary_weight: float = 2.0
    distill_bce_weight: float = 1.0
    distill_dice_weight: float = 0.5
    distill_logit_weight: float = 0.0

    # Debug helper for overfit mode.
    overfit_steps: int = 20


@dataclass
class RunConfig:
    # Main entrypoint.
    mode: str = "train"  # smoke | train | eval | infer | overfit | speed | speed_real
    seed: int = 42
    split_name: str = "train"
    batch_size: int = 8
    num_workers: int = 12
    device: str = "cuda"
    output_dir: str = DEFAULT_OUTPUT_DIR
    checkpoint_path: str = ""
    save_predictions: bool = True
    save_visualizations: bool = False
    compute_metrics: bool = False
    threshold: float = 0.5
    metric_hard_threshold: float = 102.0 / 255.0
    metric_mae_threshold: float = 12.0 / 255.0
    metric_use_soft_mae: bool = False
    enable_cudnn_benchmark: bool = True
    allow_tf32: bool = True
    fps_warmup_steps: int = 10
    fps_measure_steps: int = 100
    speed_profile_enabled: bool = False
    use_cuda_graph: bool = False
    cuda_graph_warmup_replays: int = 3
    cuda_graph_disable_group_contract_extension: bool = False
    compile_inference: bool = False
    compile_mode: str = "reduce-overhead"
    compile_fullgraph: bool = False
    compile_backbone_blocks: bool = False
    compile_backbone_block_indices: tuple[int, ...] = ()
    distributed_world_size: int = 1
    distributed_master_addr: str = "127.0.0.1"
    distributed_master_port: int = 29541
    prefetch_factor: int = 4
    persistent_workers: bool = True
    enable_tensorboard: bool = True
    tensorboard_log_dir: str = ""
    enable_wandb: bool = False
    wandb_project: str = "rvsd"
    wandb_run_name: str = ""
    wandb_mode: str = "offline"


@dataclass
class RVSDBaselineConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    shadow_free_aux: ShadowFreeAuxConfig = field(default_factory=ShadowFreeAuxConfig)
    shadow_existence_calibration: ShadowExistenceCalibrationConfig = field(
        default_factory=ShadowExistenceCalibrationConfig
    )
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    run: RunConfig = field(default_factory=RunConfig)


def build_baseline_config(eval_args: dict[str, Any] | None = None) -> RVSDBaselineConfig:
    eval_args = eval_args or {}
    if "config" in eval_args:
        config_path = eval_args.pop("config")
        base_config = OmegaConf.load(config_path)
        merged = OmegaConf.merge(
            OmegaConf.structured(RVSDBaselineConfig),
            base_config,
            OmegaConf.create(eval_args),
        )
    else:
        merged = OmegaConf.merge(
            OmegaConf.structured(RVSDBaselineConfig),
            OmegaConf.create(eval_args),
        )
    return OmegaConf.to_object(merged)  # type: ignore[return-value]
