from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from rvsd.configs.baseline import RVSDBaselineConfig, ShadowFreeAuxConfig


class ShadowFreeAuxLoss(nn.Module):
    """Training-only auxiliary loss for shadow-free clips."""

    def __init__(
        self,
        lambda_total: float = 0.2,
        lambda_zero_mask: float = 1.0,
        lambda_existence_negative: float = 0.5,
        lambda_temporal_consistency: float = 0.1,
        lambda_confidence_suppression: float = 0.1,
        lambda_easy_risk: float = 0.05,
        confidence_margin: float = 0.05,
        easy_risk_score_target: float = 0.1,
    ):
        super().__init__()
        self.lambda_total = lambda_total
        self.lambda_zero_mask = lambda_zero_mask
        self.lambda_existence_negative = lambda_existence_negative
        self.lambda_temporal_consistency = lambda_temporal_consistency
        self.lambda_confidence_suppression = lambda_confidence_suppression
        self.lambda_easy_risk = lambda_easy_risk
        self.confidence_margin = confidence_margin
        self.easy_risk_score_target = easy_risk_score_target
        self.bce = nn.BCEWithLogitsLoss()

    def _zero_mask_loss(self, logits: torch.Tensor) -> torch.Tensor:
        return self.bce(logits, torch.zeros_like(logits))

    def _existence_negative_loss(
        self,
        existence_outputs: dict[str, torch.Tensor] | None,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if existence_outputs is None or "existence_logits" not in existence_outputs:
            return reference.new_tensor(0.0)
        existence_logits = existence_outputs["existence_logits"]
        return self.bce(existence_logits, torch.zeros_like(existence_logits))

    def _temporal_consistency_loss(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.shape[1] <= 1:
            return logits.new_tensor(0.0)
        probs = torch.sigmoid(logits)
        return (probs[:, 1:] - probs[:, :-1]).abs().mean()

    def _confidence_suppression_loss(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        if self.confidence_margin > 0:
            return F.relu(probs - self.confidence_margin).mean()
        return probs.mean()

    def _easy_risk_loss(
        self,
        adaptive_outputs: dict[str, object] | None,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if adaptive_outputs is None or not adaptive_outputs.get("enabled", False):
            return reference.new_tensor(0.0)
        stage_outputs = adaptive_outputs.get("stage_outputs")
        if not stage_outputs:
            return reference.new_tensor(0.0)

        stage_losses = []
        for stage in stage_outputs:
            frame_group_scores = stage.get("frame_group_scores")
            if not frame_group_scores:
                continue
            stacked_scores = torch.stack(
                [torch.sigmoid(group_scores) for group_scores in frame_group_scores],
                dim=0,
            )
            stage_losses.append(F.relu(stacked_scores - self.easy_risk_score_target).mean())

        if not stage_losses:
            return reference.new_tensor(0.0)
        return torch.stack(stage_losses).mean()

    def forward(
        self,
        logits: torch.Tensor,
        existence_outputs: dict[str, torch.Tensor] | None = None,
        adaptive_outputs: dict[str, object] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        zero_mask = self._zero_mask_loss(logits)
        existence_negative = self._existence_negative_loss(existence_outputs, logits)
        temporal_consistency = self._temporal_consistency_loss(logits)
        confidence_suppression = self._confidence_suppression_loss(logits)
        easy_risk = self._easy_risk_loss(adaptive_outputs, logits)

        total = self.lambda_total * (
            self.lambda_zero_mask * zero_mask
            + self.lambda_existence_negative * existence_negative
            + self.lambda_temporal_consistency * temporal_consistency
            + self.lambda_confidence_suppression * confidence_suppression
            + self.lambda_easy_risk * easy_risk
        )
        return total, {
            "loss_shadow_free_total": total.detach(),
            "loss_shadow_free_zero_mask": zero_mask.detach(),
            "loss_shadow_free_existence_negative": existence_negative.detach(),
            "loss_shadow_free_temporal_consistency": temporal_consistency.detach(),
            "loss_shadow_free_confidence_suppression": confidence_suppression.detach(),
            "loss_shadow_free_easy_risk": easy_risk.detach(),
        }


def build_shadow_free_aux_loss(
    config: RVSDBaselineConfig | ShadowFreeAuxConfig,
) -> ShadowFreeAuxLoss:
    aux_cfg = config.shadow_free_aux if isinstance(config, RVSDBaselineConfig) else config
    return ShadowFreeAuxLoss(
        lambda_total=aux_cfg.lambda_total,
        lambda_zero_mask=aux_cfg.lambda_zero_mask,
        lambda_existence_negative=aux_cfg.lambda_existence_negative,
        lambda_temporal_consistency=aux_cfg.lambda_temporal_consistency,
        lambda_confidence_suppression=aux_cfg.lambda_confidence_suppression,
        lambda_easy_risk=aux_cfg.lambda_easy_risk,
        confidence_margin=aux_cfg.confidence_margin,
        easy_risk_score_target=aux_cfg.easy_risk_score_target,
    )
