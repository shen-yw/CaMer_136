from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from rvsd.configs.baseline import RVSDBaselineConfig, ShadowExistenceCalibrationConfig


class ShadowExistenceCalibrationLoss(nn.Module):
    """
    Dormant training loss for a future frame-wise existence + calibration branch.
    """

    def __init__(
        self,
        lambda_frame_existence: float = 1.0,
        lambda_clip_existence: float = 0.5,
        lambda_no_shadow_suppression: float = 0.25,
        lambda_calibration_consistency: float = 0.1,
    ):
        super().__init__()
        self.lambda_frame_existence = lambda_frame_existence
        self.lambda_clip_existence = lambda_clip_existence
        self.lambda_no_shadow_suppression = lambda_no_shadow_suppression
        self.lambda_calibration_consistency = lambda_calibration_consistency
        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self,
        frame_existence_logits: torch.Tensor,
        mask_targets: torch.Tensor,
        segmentation_logits: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if frame_existence_logits.ndim != 2:
            raise ValueError(
                f"frame_existence_logits must be [B, T], got {tuple(frame_existence_logits.shape)}"
            )
        if mask_targets.ndim != 5:
            raise ValueError(f"mask_targets must be [B, T, 1, H, W], got {tuple(mask_targets.shape)}")

        frame_targets = (mask_targets.flatten(2).sum(dim=2) > 0).to(frame_existence_logits.dtype)
        clip_targets = frame_targets.max(dim=1).values
        clip_logits = frame_existence_logits.max(dim=1).values

        frame_existence = self.bce(frame_existence_logits, frame_targets)
        clip_existence = self.bce(clip_logits, clip_targets)

        no_shadow_suppression = frame_existence_logits.new_tensor(0.0)
        calibration_consistency = frame_existence_logits.new_tensor(0.0)
        if segmentation_logits is not None:
            if segmentation_logits.ndim != 5:
                raise ValueError(
                    f"segmentation_logits must be [B, T, 1, H, W], got {tuple(segmentation_logits.shape)}"
                )
            mean_shadow = torch.sigmoid(segmentation_logits).mean(dim=(2, 3, 4))
            no_shadow_mask = (frame_targets < 0.5).to(mean_shadow.dtype)
            no_shadow_suppression = (mean_shadow * no_shadow_mask).sum() / no_shadow_mask.sum().clamp_min(1.0)
            calibration_consistency = F.mse_loss(torch.sigmoid(frame_existence_logits), mean_shadow.detach())

        total = (
            self.lambda_frame_existence * frame_existence
            + self.lambda_clip_existence * clip_existence
            + self.lambda_no_shadow_suppression * no_shadow_suppression
            + self.lambda_calibration_consistency * calibration_consistency
        )
        return total, {
            "loss_shadow_existence_total": total.detach(),
            "loss_shadow_existence_frame": frame_existence.detach(),
            "loss_shadow_existence_clip": clip_existence.detach(),
            "loss_shadow_existence_no_shadow_suppression": no_shadow_suppression.detach(),
            "loss_shadow_existence_calibration_consistency": calibration_consistency.detach(),
        }


def build_shadow_existence_calibration_loss(
    config: RVSDBaselineConfig | ShadowExistenceCalibrationConfig,
) -> ShadowExistenceCalibrationLoss:
    cfg = (
        config.shadow_existence_calibration
        if isinstance(config, RVSDBaselineConfig)
        else config
    )
    return ShadowExistenceCalibrationLoss(
        lambda_frame_existence=cfg.lambda_frame_existence,
        lambda_clip_existence=cfg.lambda_clip_existence,
        lambda_no_shadow_suppression=cfg.lambda_no_shadow_suppression,
        lambda_calibration_consistency=cfg.lambda_calibration_consistency,
    )
