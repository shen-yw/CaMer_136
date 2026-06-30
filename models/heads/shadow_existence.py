from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ShadowExistenceCalibrationOutput:
    frame_logits: torch.Tensor
    frame_probs: torch.Tensor
    clip_logits: torch.Tensor
    clip_probs: torch.Tensor
    corrected_segmentation_logits: torch.Tensor | None = None
    adjusted_keep_ratio: torch.Tensor | None = None
    keep_ratio_scale: torch.Tensor | None = None


class FrameShadowExistenceHead(nn.Module):
    """
    Lightweight frame-wise shadow existence branch.

    This module is intentionally standalone and is not wired into the main model
    yet. Later, it can consume frame CLS tokens and produce frame/clip presence
    estimates for soft segmentation calibration and merge scheduling.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 256,
        temporal_kernel_size: int = 3,
        dropout: float = 0.0,
        clip_aggregation: str = "max",
    ):
        super().__init__()
        if temporal_kernel_size < 1 or temporal_kernel_size % 2 == 0:
            raise ValueError(
                f"temporal_kernel_size must be a positive odd integer, got {temporal_kernel_size}"
            )
        if clip_aggregation not in {"max", "mean", "logsumexp"}:
            raise ValueError(f"Unsupported clip_aggregation: {clip_aggregation}")

        self.clip_aggregation = clip_aggregation
        self.pre = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
        )
        self.temporal = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=temporal_kernel_size,
            padding=temporal_kernel_size // 2,
        )
        self.dropout = nn.Dropout(dropout)
        self.frame_out = nn.Linear(hidden_dim, 1)

    def _aggregate_clip_logits(self, frame_logits: torch.Tensor) -> torch.Tensor:
        if self.clip_aggregation == "max":
            return frame_logits.max(dim=1).values
        if self.clip_aggregation == "mean":
            return frame_logits.mean(dim=1)
        return torch.logsumexp(frame_logits, dim=1) - torch.log(
            torch.tensor(frame_logits.shape[1], device=frame_logits.device, dtype=frame_logits.dtype)
        )

    def forward(self, frame_cls_tokens: torch.Tensor) -> ShadowExistenceCalibrationOutput:
        if frame_cls_tokens.ndim != 3:
            raise ValueError(
                f"FrameShadowExistenceHead expects [B, T, C], got {tuple(frame_cls_tokens.shape)}"
            )
        hidden = self.pre(frame_cls_tokens)
        hidden = self.temporal(hidden.transpose(1, 2)).transpose(1, 2)
        hidden = self.dropout(hidden)
        frame_logits = self.frame_out(hidden).squeeze(-1)
        clip_logits = self._aggregate_clip_logits(frame_logits)
        return ShadowExistenceCalibrationOutput(
            frame_logits=frame_logits,
            frame_probs=torch.sigmoid(frame_logits),
            clip_logits=clip_logits,
            clip_probs=torch.sigmoid(clip_logits),
        )


class ShadowExistenceCalibrator(nn.Module):
    """
    Stateless soft-correction helper for inference-time use.

    Given frame-wise existence logits, it can:
    - softly suppress segmentation logits on likely no-shadow frames
    - suggest lower keep ratios for likely no-shadow frames
    """

    def __init__(
        self,
        use_soft_logit_correction: bool = True,
        logit_correction_scale: float = 1.0,
        logit_correction_power: float = 1.0,
        use_keep_ratio_adjustment: bool = True,
        keep_ratio_min_scale: float = 0.75,
        keep_ratio_max_scale: float = 1.0,
        keep_ratio_adjustment_power: float = 1.0,
    ):
        super().__init__()
        if keep_ratio_min_scale <= 0 or keep_ratio_max_scale <= 0:
            raise ValueError("keep_ratio scales must be positive.")
        if keep_ratio_min_scale > keep_ratio_max_scale:
            raise ValueError("keep_ratio_min_scale cannot exceed keep_ratio_max_scale.")
        self.use_soft_logit_correction = use_soft_logit_correction
        self.logit_correction_scale = logit_correction_scale
        self.logit_correction_power = logit_correction_power
        self.use_keep_ratio_adjustment = use_keep_ratio_adjustment
        self.keep_ratio_min_scale = keep_ratio_min_scale
        self.keep_ratio_max_scale = keep_ratio_max_scale
        self.keep_ratio_adjustment_power = keep_ratio_adjustment_power

    def _keep_ratio_scale(self, frame_probs: torch.Tensor) -> torch.Tensor:
        scaled = frame_probs.pow(self.keep_ratio_adjustment_power)
        return self.keep_ratio_min_scale + (self.keep_ratio_max_scale - self.keep_ratio_min_scale) * scaled

    def _broadcast_base_keep_ratio(
        self,
        base_keep_ratio: float | torch.Tensor,
        frame_probs: torch.Tensor,
    ) -> torch.Tensor:
        if not torch.is_tensor(base_keep_ratio):
            return frame_probs.new_full(frame_probs.shape, float(base_keep_ratio))
        if base_keep_ratio.ndim == 0:
            return frame_probs.new_full(frame_probs.shape, float(base_keep_ratio.item()))
        if base_keep_ratio.ndim == 1:
            if base_keep_ratio.shape[0] != frame_probs.shape[0]:
                raise ValueError("Per-batch keep ratio must have shape [B].")
            return base_keep_ratio[:, None].to(frame_probs.device, dtype=frame_probs.dtype).expand_as(frame_probs)
        if base_keep_ratio.shape != frame_probs.shape:
            raise ValueError(
                f"Expected keep ratio shape {tuple(frame_probs.shape)}, got {tuple(base_keep_ratio.shape)}"
            )
        return base_keep_ratio.to(frame_probs.device, dtype=frame_probs.dtype)

    def forward(
        self,
        frame_existence_logits: torch.Tensor,
        segmentation_logits: torch.Tensor | None = None,
        base_keep_ratio: float | torch.Tensor | None = None,
    ) -> ShadowExistenceCalibrationOutput:
        if frame_existence_logits.ndim != 2:
            raise ValueError(
                f"ShadowExistenceCalibrator expects frame_existence_logits [B, T], got {tuple(frame_existence_logits.shape)}"
            )

        frame_probs = torch.sigmoid(frame_existence_logits)
        clip_logits = frame_existence_logits.max(dim=1).values
        clip_probs = torch.sigmoid(clip_logits)

        corrected_segmentation_logits = None
        if segmentation_logits is not None and self.use_soft_logit_correction:
            if segmentation_logits.ndim != 5:
                raise ValueError(
                    "segmentation_logits must have shape [B, T, 1, H, W] for soft calibration."
                )
            suppression = (1.0 - frame_probs).pow(self.logit_correction_power)
            corrected_segmentation_logits = segmentation_logits - self.logit_correction_scale * suppression[:, :, None, None, None]

        keep_ratio_scale = None
        adjusted_keep_ratio = None
        if base_keep_ratio is not None and self.use_keep_ratio_adjustment:
            keep_ratio_scale = self._keep_ratio_scale(frame_probs)
            adjusted_keep_ratio = self._broadcast_base_keep_ratio(base_keep_ratio, frame_probs) * keep_ratio_scale

        return ShadowExistenceCalibrationOutput(
            frame_logits=frame_existence_logits,
            frame_probs=frame_probs,
            clip_logits=clip_logits,
            clip_probs=clip_probs,
            corrected_segmentation_logits=corrected_segmentation_logits,
            adjusted_keep_ratio=adjusted_keep_ratio,
            keep_ratio_scale=keep_ratio_scale,
        )
