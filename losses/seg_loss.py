from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from rvsd.configs.baseline import AdaptiveComputeConfig
from rvsd.models.backbones.adaptive_compute import build_layout_from_members, compute_group_scores


def dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    return dice_loss_from_probs(probs, targets, eps=eps)


def dice_loss_from_probs(probs: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = probs.flatten(1)
    targets = targets.flatten(1)

    intersection = (probs * targets).sum(dim=1)
    denominator = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def tversky_loss_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.3,
    beta: float = 0.7,
    eps: float = 1e-6,
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs = probs.flatten(1)
    targets = targets.flatten(1)

    true_positive = (probs * targets).sum(dim=1)
    false_positive = (probs * (1.0 - targets)).sum(dim=1)
    false_negative = ((1.0 - probs) * targets).sum(dim=1)
    tversky = (true_positive + eps) / (true_positive + alpha * false_positive + beta * false_negative + eps)
    return 1.0 - tversky.mean()


def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    total_positive = gt_sorted.sum()
    intersection = total_positive - gt_sorted.cumsum(dim=0)
    union = total_positive + (1.0 - gt_sorted).cumsum(dim=0)
    jaccard = 1.0 - intersection / union.clamp_min(1e-6)
    if gt_sorted.numel() > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


def lovasz_hinge_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    flat_logits = logits.reshape(-1)
    flat_targets = targets.reshape(-1)
    if flat_targets.numel() == 0:
        return logits.new_tensor(0.0)
    signs = 2.0 * flat_targets - 1.0
    errors = 1.0 - flat_logits * signs
    errors_sorted, permutation = torch.sort(errors, descending=True)
    gt_sorted = flat_targets[permutation]
    return torch.dot(F.relu(errors_sorted), _lovasz_grad(gt_sorted))


class VideoShadowSegLoss(nn.Module):
    def __init__(
        self,
        lambda_bce: float = 1.0,
        positive_bce_weight: float = 1.0,
        lambda_dice: float = 1.0,
        lambda_tversky: float = 0.0,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
        lambda_lovasz: float = 0.0,
        lambda_shadow_body: float = 0.0,
        lambda_body_aux: float = 0.0,
        lambda_detail_aux: float = 0.0,
        lambda_boundary_aux: float = 0.0,
        lambda_boundary_main: float = 0.0,
        lambda_temporal_consistency: float = 0.0,
        lambda_nonshadow_precision: float = 0.0,
        lambda_dark_negative: float = 0.0,
        dark_negative_threshold: float = 0.35,
        dark_negative_boundary_kernel: int = 9,
        distill_weight: float = 0.0,
        distill_boundary_weight: float = 2.0,
        distill_bce_weight: float = 1.0,
        distill_dice_weight: float = 0.5,
        distill_logit_weight: float = 0.0,
        adaptive_config: AdaptiveComputeConfig | None = None,
    ):
        super().__init__()
        self.lambda_bce = lambda_bce
        self.positive_bce_weight = positive_bce_weight
        self.lambda_dice = lambda_dice
        self.lambda_tversky = lambda_tversky
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta
        self.lambda_lovasz = lambda_lovasz
        self.lambda_shadow_body = lambda_shadow_body
        self.lambda_body_aux = lambda_body_aux
        self.lambda_detail_aux = lambda_detail_aux
        self.lambda_boundary_aux = lambda_boundary_aux
        self.lambda_boundary_main = lambda_boundary_main
        self.lambda_temporal_consistency = lambda_temporal_consistency
        self.lambda_nonshadow_precision = lambda_nonshadow_precision
        self.lambda_dark_negative = lambda_dark_negative
        self.dark_negative_threshold = dark_negative_threshold
        self.dark_negative_boundary_kernel = dark_negative_boundary_kernel
        self.distill_weight = distill_weight
        self.distill_boundary_weight = distill_boundary_weight
        self.distill_bce_weight = distill_bce_weight
        self.distill_dice_weight = distill_dice_weight
        self.distill_logit_weight = distill_logit_weight
        self.adaptive_config = adaptive_config or AdaptiveComputeConfig()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_prob = nn.BCELoss()

    def _segmentation_bce(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.positive_bce_weight <= 1.0:
            return F.binary_cross_entropy_with_logits(logits, targets)
        weight = 1.0 + (float(self.positive_bce_weight) - 1.0) * targets
        return F.binary_cross_entropy_with_logits(logits, targets, weight=weight)

    def _normalize_map(self, values: torch.Tensor) -> torch.Tensor:
        flat = values.reshape(-1)
        min_value = flat.min()
        max_value = flat.max()
        return (values - min_value) / (max_value - min_value + 1e-6)

    def _stage_frame_layout(
        self,
        stage: dict[str, object],
        frame_index: int,
        patch_height: int,
        patch_width: int,
    ):
        group_members = stage["frame_group_members"][frame_index]
        group_mask = stage["frame_group_member_mask"][frame_index]
        patch_coords = torch.tensor(
            [(h + 0.5, w + 0.5) for h in range(patch_height) for w in range(patch_width)],
            dtype=torch.float32,
        )
        return build_layout_from_members(
            group_members=[
                group_members[group_idx][group_mask[group_idx]].tolist()
                for group_idx in range(group_members.shape[0])
            ],
            patch_coords=patch_coords,
            patch_height=patch_height,
            patch_width=patch_width,
        )

    def _boundary_band(self, targets: torch.Tensor, kernel_size: int) -> torch.Tensor:
        padding = kernel_size // 2
        dilation = F.max_pool2d(targets, kernel_size=kernel_size, stride=1, padding=padding)
        erosion = -F.max_pool2d(-targets, kernel_size=kernel_size, stride=1, padding=padding)
        return (dilation - erosion).clamp(0.0, 1.0)

    def _erode_mask(self, targets: torch.Tensor, kernel_size: int) -> torch.Tensor:
        padding = kernel_size // 2
        return -F.max_pool2d(-targets, kernel_size=kernel_size, stride=1, padding=padding)

    def _blur_mask(self, targets: torch.Tensor, kernel_size: int) -> torch.Tensor:
        padding = kernel_size // 2
        return F.avg_pool2d(targets, kernel_size=kernel_size, stride=1, padding=padding)

    def _dilate_sequence_mask(self, targets: torch.Tensor, kernel_size: int) -> torch.Tensor:
        kernel_size = max(int(kernel_size), 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        flat_targets = targets.reshape(targets.shape[0] * targets.shape[1], 1, targets.shape[-2], targets.shape[-1])
        padding = kernel_size // 2
        dilated = F.max_pool2d(flat_targets, kernel_size=kernel_size, stride=1, padding=padding)
        return dilated.reshape_as(targets)

    def _auxiliary_structure_targets(
        self,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat_targets = targets.reshape(targets.shape[0] * targets.shape[1], 1, targets.shape[-2], targets.shape[-1])
        body_core = self._erode_mask(flat_targets, kernel_size=5).clamp(0.0, 1.0)
        body_mid = ((self._blur_mask(flat_targets, kernel_size=7) - 0.15) / 0.85).clamp(0.0, 1.0)
        body_wide = ((self._blur_mask(flat_targets, kernel_size=15) - 0.05) / 0.95).clamp(0.0, 1.0)
        body_target = torch.maximum(body_core, 0.6 * body_mid + 0.4 * body_wide).clamp(0.0, 1.0)
        boundary_band = self._boundary_band(flat_targets, kernel_size=5)
        penumbra_target = (self._blur_mask(flat_targets, kernel_size=9) - body_target).clamp(0.0, 1.0)
        boundary_target = torch.maximum(boundary_band, penumbra_target).clamp(0.0, 1.0)
        detail_target = torch.maximum((flat_targets - body_target).clamp(0.0, 1.0), boundary_target * flat_targets)
        return (
            body_target.reshape_as(targets),
            detail_target.reshape_as(targets),
            boundary_target.reshape_as(targets),
        )

    def _main_boundary_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        flat_targets = targets.reshape(targets.shape[0] * targets.shape[1], 1, targets.shape[-2], targets.shape[-1])
        flat_probs = torch.sigmoid(logits).reshape_as(flat_targets)
        boundary_target = self._boundary_band(flat_targets, kernel_size=5)
        predicted_boundary = self._boundary_band(flat_probs, kernel_size=3).clamp(1e-4, 1.0 - 1e-4)
        with torch.autocast(device_type=logits.device.type, enabled=False):
            predicted_boundary_f = predicted_boundary.float()
            boundary_target_f = boundary_target.float()
            boundary_bce = F.binary_cross_entropy(predicted_boundary_f, boundary_target_f)
            boundary_dice = dice_loss_from_probs(predicted_boundary_f, boundary_target_f)
        return 0.5 * (boundary_bce + boundary_dice)

    def _shadow_body_recall_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        flat_targets = targets.reshape(targets.shape[0] * targets.shape[1], 1, targets.shape[-2], targets.shape[-1])
        body_target = self._erode_mask(flat_targets, kernel_size=5).clamp(0.0, 1.0)
        if float(body_target.sum().item()) <= 1e-6:
            body_target = flat_targets
        probs = torch.sigmoid(logits).reshape_as(body_target)
        denominator = body_target.sum().clamp_min(1.0)
        return ((1.0 - probs) * body_target).sum() / denominator

    def _temporal_consistency_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        if logits.shape[1] < 2:
            return logits.new_tensor(0.0)
        probs = torch.sigmoid(logits)
        body_target, detail_target, boundary_target = self._auxiliary_structure_targets(targets)
        stable_body = 0.5 * (body_target[:, 1:] + body_target[:, :-1])
        stable_detail = 0.5 * (detail_target[:, 1:] + detail_target[:, :-1])
        unstable_boundary = torch.maximum(boundary_target[:, 1:], boundary_target[:, :-1])
        target_transition = torch.abs(targets[:, 1:] - targets[:, :-1])
        stable_weight = (stable_body + 0.25 * stable_detail) * (1.0 - unstable_boundary) * (1.0 - target_transition)
        stable_weight = stable_weight.clamp(0.0, 1.0)
        if float(stable_weight.sum().item()) <= 1e-6:
            return logits.new_tensor(0.0)
        temporal_delta = torch.abs(probs[:, 1:] - probs[:, :-1])
        return (temporal_delta * stable_weight).sum() / stable_weight.sum().clamp_min(1.0)

    def _nonshadow_precision_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        non_shadow = (1.0 - targets).clamp(0.0, 1.0)
        return (probs * non_shadow).sum() / non_shadow.sum().clamp_min(1.0)

    def _dark_negative_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        clip: torch.Tensor | None,
    ) -> torch.Tensor:
        if clip is None:
            return logits.new_tensor(0.0)
        if clip.ndim != 5:
            raise ValueError(f"Expected clip shape [B, 3, T, H, W], got {tuple(clip.shape)}")
        if clip.shape[0] != logits.shape[0] or clip.shape[2] != logits.shape[1] or clip.shape[1] != 3:
            raise ValueError(f"Clip/logit shape mismatch: clip={tuple(clip.shape)} logits={tuple(logits.shape)}")

        batch_size, num_frames = logits.shape[:2]
        frame_clip = clip.permute(0, 2, 1, 3, 4).to(device=logits.device, dtype=logits.dtype)
        if tuple(frame_clip.shape[-2:]) != tuple(logits.shape[-2:]):
            frame_clip = F.interpolate(
                frame_clip.reshape(batch_size * num_frames, 3, frame_clip.shape[-2], frame_clip.shape[-1]),
                size=logits.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).reshape(batch_size, num_frames, 3, logits.shape[-2], logits.shape[-1])
        frame_clip = frame_clip.clamp(0.0, 1.0)
        rgb_weights = torch.tensor(
            [0.299, 0.587, 0.114],
            device=logits.device,
            dtype=logits.dtype,
        ).view(1, 1, 3, 1, 1)
        luminance = (frame_clip * rgb_weights).sum(dim=2, keepdim=True)
        dark_region = (luminance < float(self.dark_negative_threshold)).to(dtype=logits.dtype)
        shadow_exclusion = self._dilate_sequence_mask(targets, self.dark_negative_boundary_kernel)
        reliable_dark_nonshadow = dark_region * (1.0 - shadow_exclusion).clamp(0.0, 1.0)
        probs = torch.sigmoid(logits)
        return (probs.square() * reliable_dark_nonshadow).sum() / reliable_dark_nonshadow.sum().clamp_min(1.0)

    def _auxiliary_structure_losses(
        self,
        auxiliary_outputs: dict[str, torch.Tensor] | None,
        targets: torch.Tensor,
        reference_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if auxiliary_outputs is None:
            zero = reference_logits.new_tensor(0.0)
            return zero, {
                "loss_body_aux": zero,
                "loss_detail_aux": zero,
                "loss_boundary_aux": zero,
            }

        body_target, detail_target, boundary_target = self._auxiliary_structure_targets(targets)
        total = reference_logits.new_tensor(0.0)
        losses = {
            "loss_body_aux": reference_logits.new_tensor(0.0),
            "loss_detail_aux": reference_logits.new_tensor(0.0),
            "loss_boundary_aux": reference_logits.new_tensor(0.0),
        }

        if self.lambda_body_aux > 0 and "body_logits" in auxiliary_outputs:
            body_loss = self.bce(auxiliary_outputs["body_logits"], body_target)
            total = total + self.lambda_body_aux * body_loss
            losses["loss_body_aux"] = body_loss.detach()
        if self.lambda_detail_aux > 0 and "detail_logits" in auxiliary_outputs:
            detail_loss = self.bce(auxiliary_outputs["detail_logits"], detail_target)
            total = total + self.lambda_detail_aux * detail_loss
            losses["loss_detail_aux"] = detail_loss.detach()
        if self.lambda_boundary_aux > 0 and "boundary_logits" in auxiliary_outputs:
            boundary_loss = self.bce(auxiliary_outputs["boundary_logits"], boundary_target)
            total = total + self.lambda_boundary_aux * boundary_loss
            losses["loss_boundary_aux"] = boundary_loss.detach()

        return total, losses

    def _teacher_patch_difficulty(self, logits: torch.Tensor, targets: torch.Tensor, patch_hw: tuple[int, int]) -> torch.Tensor:
        batch_size, num_frames, _, _, _ = logits.shape
        patch_height, patch_width = patch_hw
        probs = torch.sigmoid(logits.detach()).reshape(batch_size * num_frames, 1, logits.shape[-2], logits.shape[-1])
        pooled_probs = F.adaptive_avg_pool2d(probs, output_size=patch_hw).reshape(batch_size, num_frames, patch_height, patch_width)
        uncertainty = 1.0 - torch.abs(2.0 * pooled_probs - 1.0)

        temporal = torch.zeros_like(uncertainty)
        counts = torch.zeros_like(uncertainty)
        if num_frames > 1:
            temporal_diff = torch.abs(pooled_probs[:, 1:] - pooled_probs[:, :-1])
            temporal[:, 1:] += temporal_diff
            temporal[:, :-1] += temporal_diff
            counts[:, 1:] += 1.0
            counts[:, :-1] += 1.0
        temporal = temporal / counts.clamp_min(1.0)

        mask_bt = targets.reshape(batch_size * num_frames, 1, targets.shape[-2], targets.shape[-1])
        ambiguity = self._boundary_band(mask_bt, self.adaptive_config.ambiguity_kernel_size)
        ambiguity = F.adaptive_avg_pool2d(ambiguity, output_size=patch_hw).reshape(batch_size, num_frames, patch_height, patch_width)

        weighted = (
            self.adaptive_config.teacher_uncertainty_weight * uncertainty
            + self.adaptive_config.teacher_temporal_weight * temporal
            + self.adaptive_config.teacher_ambiguity_weight * ambiguity
        )
        weighted_flat = weighted.reshape(batch_size, -1)
        weighted_min = weighted_flat.min(dim=1).values.view(batch_size, 1, 1, 1)
        weighted_max = weighted_flat.max(dim=1).values.view(batch_size, 1, 1, 1)
        return (weighted - weighted_min) / (weighted_max - weighted_min + 1e-6)

    def _risk_patch_signals(self, logits: torch.Tensor, targets: torch.Tensor, patch_hw: tuple[int, int]) -> dict[str, torch.Tensor]:
        batch_size, num_frames, _, _, _ = logits.shape
        patch_height, patch_width = patch_hw
        probs = torch.sigmoid(logits.detach()).reshape(batch_size * num_frames, 1, logits.shape[-2], logits.shape[-1])
        pooled_probs = F.adaptive_avg_pool2d(probs, output_size=patch_hw).reshape(batch_size, num_frames, patch_height, patch_width)
        pooled_targets = F.adaptive_avg_pool2d(
            targets.reshape(batch_size * num_frames, 1, targets.shape[-2], targets.shape[-1]),
            output_size=patch_hw,
        ).reshape(batch_size, num_frames, patch_height, patch_width)
        uncertainty = 1.0 - torch.abs(2.0 * pooled_probs - 1.0)
        prediction_error = torch.abs(pooled_probs - pooled_targets)

        temporal = torch.zeros_like(uncertainty)
        counts = torch.zeros_like(uncertainty)
        if num_frames > 1:
            temporal_diff = torch.abs(pooled_probs[:, 1:] - pooled_probs[:, :-1])
            temporal[:, 1:] += temporal_diff
            temporal[:, :-1] += temporal_diff
            counts[:, 1:] += 1.0
            counts[:, :-1] += 1.0
        temporal = temporal / counts.clamp_min(1.0)

        ambiguity = self._boundary_band(
            targets.reshape(batch_size * num_frames, 1, targets.shape[-2], targets.shape[-1]),
            self.adaptive_config.ambiguity_kernel_size,
        )
        ambiguity = F.adaptive_avg_pool2d(ambiguity, output_size=patch_hw).reshape(batch_size, num_frames, patch_height, patch_width)
        return {
            "probs": pooled_probs,
            "targets": pooled_targets,
            "uncertainty": uncertainty,
            "temporal": temporal,
            "ambiguity": ambiguity,
            "prediction_error": prediction_error,
        }

    def _group_merge_damage(
        self,
        token_probs: torch.Tensor,
        token_targets: torch.Tensor,
        layout,
    ) -> torch.Tensor:
        member_indices = layout.group_member_indices.to(token_probs.device)
        member_mask = layout.group_member_mask.to(token_probs.device)
        group_sizes = member_mask.sum(dim=1).clamp_min(1)

        gathered_probs = token_probs[member_indices.clamp_min(0)]
        gathered_targets = token_targets[member_indices.clamp_min(0)]
        gathered_probs = gathered_probs.clamp(1e-6, 1.0 - 1e-6)
        member_mask_float = member_mask.to(gathered_probs.dtype)
        with torch.autocast(device_type=token_probs.device.type, enabled=False):
            gathered_probs_f = gathered_probs.float()
            gathered_targets_f = gathered_targets.float()
            member_mask_float_f = member_mask_float.float()
            group_sizes_f = group_sizes.float()

            baseline = F.binary_cross_entropy(gathered_probs_f, gathered_targets_f, reduction="none")
            baseline = (baseline * member_mask_float_f).sum(dim=1) / group_sizes_f

            merged_probs = ((gathered_probs_f * member_mask_float_f).sum(dim=1) / group_sizes_f).clamp(1e-6, 1.0 - 1e-6)
            merged_probs = merged_probs.unsqueeze(1).expand_as(gathered_probs_f)
            merged = F.binary_cross_entropy(merged_probs, gathered_targets_f, reduction="none")
            merged = (merged * member_mask_float_f).sum(dim=1) / group_sizes_f
        return (merged - baseline).clamp_min(0.0)

    def _selection_loss(
        self,
        predicted_groups: torch.Tensor,
        teacher_groups: torch.Tensor,
        group_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        losses = []
        predicted_groups = predicted_groups.reshape(-1, predicted_groups.shape[-1])
        teacher_groups = teacher_groups.reshape(-1, teacher_groups.shape[-1])
        weight_samples = None
        if group_weights is not None:
            weight_samples = group_weights.reshape(-1, group_weights.shape[-1])

        for sample_index, (pred_sample, teacher_sample) in enumerate(zip(predicted_groups, teacher_groups)):
            teacher_diff = teacher_sample.unsqueeze(1) - teacher_sample.unsqueeze(0)
            valid = teacher_diff > 0
            if valid.any():
                pred_diff = pred_sample.unsqueeze(1) - pred_sample.unsqueeze(0)
                pair_loss = F.softplus(-pred_diff)
                if weight_samples is None:
                    losses.append(pair_loss[valid].mean())
                else:
                    sample_weights = weight_samples[sample_index]
                    pair_weights = 0.5 * (sample_weights.unsqueeze(1) + sample_weights.unsqueeze(0))
                    valid_weights = pair_weights[valid].clamp_min(1e-6)
                    losses.append((pair_loss[valid] * valid_weights).sum() / valid_weights.sum())
        if not losses:
            return predicted_groups.new_tensor(0.0)
        return torch.stack(losses).mean()

    def _rank_spearman(self, predicted: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        if predicted.numel() < 2:
            return predicted.new_tensor(1.0)
        predicted_rank = predicted.argsort().argsort().to(dtype=torch.float32)
        teacher_rank = teacher.argsort().argsort().to(dtype=torch.float32)
        predicted_rank = predicted_rank - predicted_rank.mean()
        teacher_rank = teacher_rank - teacher_rank.mean()
        denominator = predicted_rank.norm() * teacher_rank.norm()
        if float(denominator.item()) <= 1e-6:
            return predicted.new_tensor(0.0)
        return (predicted_rank * teacher_rank).sum() / denominator

    def _topk_overlap(self, predicted: torch.Tensor, teacher: torch.Tensor, ratio: float) -> torch.Tensor:
        num_groups = int(predicted.numel())
        topk = max(1, min(num_groups, int(round(num_groups * ratio))))
        predicted_indices = torch.topk(predicted, k=topk).indices
        teacher_indices = torch.topk(teacher, k=topk).indices
        predicted_mask = torch.zeros_like(predicted, dtype=torch.bool)
        teacher_mask = torch.zeros_like(teacher, dtype=torch.bool)
        predicted_mask[predicted_indices] = True
        teacher_mask[teacher_indices] = True
        return (predicted_mask & teacher_mask).float().sum() / float(topk)

    def _binary_partition_ranking_loss(
        self,
        predicted: torch.Tensor,
        positive_mask: torch.Tensor,
        negative_mask: torch.Tensor,
    ) -> torch.Tensor:
        positive_values = predicted[positive_mask]
        negative_values = predicted[negative_mask]
        if positive_values.numel() == 0 or negative_values.numel() == 0:
            return predicted.new_tensor(0.0)
        pair_diff = positive_values.unsqueeze(1) - negative_values.unsqueeze(0)
        return F.softplus(-pair_diff).mean()

    def _score_partition_masks(self, scores: torch.Tensor, ratio: float) -> tuple[torch.Tensor, torch.Tensor]:
        num_groups = int(scores.numel())
        topk = max(1, min(num_groups, int(round(num_groups * ratio))))
        positive_indices = torch.topk(scores, k=topk, largest=True).indices
        negative_indices = torch.topk(scores, k=topk, largest=False).indices
        positive_mask = torch.zeros_like(scores, dtype=torch.bool)
        negative_mask = torch.zeros_like(scores, dtype=torch.bool)
        positive_mask[positive_indices] = True
        negative_mask[negative_indices] = True
        negative_mask &= ~positive_mask
        return positive_mask, negative_mask

    def _boundary_aware_distill_loss(
        self,
        logits: torch.Tensor,
        teacher_logits: torch.Tensor | None,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        if teacher_logits is None or self.adaptive_config.risk_distill_weight <= 0:
            return logits.new_tensor(0.0)
        eps = 1e-6
        teacher_probs = torch.sigmoid(teacher_logits.detach())
        boundary = self._boundary_band(
            targets.reshape(targets.shape[0] * targets.shape[1], 1, targets.shape[-2], targets.shape[-1]),
            self.adaptive_config.ambiguity_kernel_size,
        ).reshape_as(logits)
        boundary_gain = max(self.adaptive_config.risk_distill_boundary_weight - 1.0, 0.0)
        weights = 1.0 + boundary_gain * boundary
        bce_weight = float(getattr(self.adaptive_config, "risk_distill_bce_weight", 1.0))
        dice_weight = float(getattr(self.adaptive_config, "risk_distill_dice_weight", 0.0))
        logit_weight = float(getattr(self.adaptive_config, "risk_distill_logit_weight", 0.0))

        total = logits.new_tensor(0.0)
        if bce_weight > 0:
            bce_distill = F.binary_cross_entropy_with_logits(logits, teacher_probs, reduction="none")
            total = total + bce_weight * (bce_distill * weights).mean()

        if dice_weight > 0:
            student_probs = torch.sigmoid(logits)
            student_flat = (student_probs * weights).flatten(1)
            teacher_flat = (teacher_probs * weights).flatten(1)
            intersection = (student_flat * teacher_flat).sum(dim=1)
            denominator = student_flat.sum(dim=1) + teacher_flat.sum(dim=1)
            dice_distill = 1.0 - ((2.0 * intersection + eps) / (denominator + eps))
            total = total + dice_weight * dice_distill.mean()

        if logit_weight > 0:
            logit_distill = F.mse_loss(logits, teacher_logits.detach(), reduction="none")
            total = total + logit_weight * (logit_distill * weights).mean()

        return total

    def _segmentation_distill_loss(
        self,
        logits: torch.Tensor,
        teacher_logits: torch.Tensor | None,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        if teacher_logits is None or self.distill_weight <= 0:
            return logits.new_tensor(0.0)
        eps = 1e-6
        teacher_probs = torch.sigmoid(teacher_logits.detach())
        boundary = self._boundary_band(
            targets.reshape(targets.shape[0] * targets.shape[1], 1, targets.shape[-2], targets.shape[-1]),
            kernel_size=5,
        ).reshape_as(logits)
        boundary_gain = max(float(self.distill_boundary_weight) - 1.0, 0.0)
        weights = 1.0 + boundary_gain * boundary

        total = logits.new_tensor(0.0)
        if self.distill_bce_weight > 0:
            bce_distill = F.binary_cross_entropy_with_logits(logits, teacher_probs, reduction="none")
            total = total + float(self.distill_bce_weight) * (bce_distill * weights).mean()

        if self.distill_dice_weight > 0:
            student_probs = torch.sigmoid(logits)
            student_flat = (student_probs * weights).flatten(1)
            teacher_flat = (teacher_probs * weights).flatten(1)
            intersection = (student_flat * teacher_flat).sum(dim=1)
            denominator = student_flat.sum(dim=1) + teacher_flat.sum(dim=1)
            dice_distill = 1.0 - ((2.0 * intersection + eps) / (denominator + eps))
            total = total + float(self.distill_dice_weight) * dice_distill.mean()

        if self.distill_logit_weight > 0:
            logit_distill = F.mse_loss(logits, teacher_logits.detach(), reduction="none")
            total = total + float(self.distill_logit_weight) * (logit_distill * weights).mean()

        return total

    def _risk_losses(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        adaptive_outputs: dict[str, object],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        signals = self._risk_patch_signals(logits, targets, adaptive_outputs["patch_hw"])
        batch_size, num_frames, patch_height, patch_width = signals["probs"].shape

        risk = logits.new_tensor(0.0)
        risk_relative_rank = logits.new_tensor(0.0)
        teacher_damage_total = logits.new_tensor(0.0)
        teacher_error_total = logits.new_tensor(0.0)
        rank_spearman_total = logits.new_tensor(0.0)
        topk_overlap_total = logits.new_tensor(0.0)
        damage_gap_total = logits.new_tensor(0.0)
        boundary_protect_total = logits.new_tensor(0.0)
        teacher_frames = 0
        probs_flat = signals["probs"].reshape(batch_size, num_frames, patch_height * patch_width)
        targets_flat = signals["targets"].reshape(batch_size, num_frames, patch_height * patch_width)
        uncertainty_flat = signals["uncertainty"].reshape(batch_size, num_frames, patch_height * patch_width)
        temporal_flat = signals["temporal"].reshape(batch_size, num_frames, patch_height * patch_width)
        ambiguity_flat = signals["ambiguity"].reshape(batch_size, num_frames, patch_height * patch_width)
        error_flat = signals["prediction_error"].reshape(batch_size, num_frames, patch_height * patch_width)

        for stage in adaptive_outputs["stage_outputs"]:
            teacher_groups_per_frame = []
            predicted_groups_per_frame = []
            boundary_weights_per_frame = []
            for frame_index, predicted_groups in enumerate(stage["frame_group_scores"]):
                batch_index = frame_index // num_frames
                time_index = frame_index % num_frames
                layout = self._stage_frame_layout(stage, frame_index, patch_height, patch_width)

                token_probs = probs_flat[batch_index, time_index]
                token_targets = targets_flat[batch_index, time_index]
                token_uncertainty = uncertainty_flat[batch_index, time_index]
                token_temporal = temporal_flat[batch_index, time_index]
                token_ambiguity = ambiguity_flat[batch_index, time_index]
                token_error = error_flat[batch_index, time_index]

                group_damage = self._group_merge_damage(token_probs, token_targets, layout)
                group_error = compute_group_scores(token_error.unsqueeze(0), layout).squeeze(0)
                group_uncertainty = compute_group_scores(token_uncertainty.unsqueeze(0), layout).squeeze(0)
                group_temporal = compute_group_scores(token_temporal.unsqueeze(0), layout).squeeze(0)
                group_ambiguity = compute_group_scores(token_ambiguity.unsqueeze(0), layout).squeeze(0)
                teacher_damage_total = teacher_damage_total + group_damage.mean()
                teacher_error_total = teacher_error_total + group_error.mean()
                teacher_frames += 1

                normalized_damage = self._normalize_map(group_damage)
                normalized_error = self._normalize_map(group_error)
                normalized_uncertainty = self._normalize_map(group_uncertainty)
                normalized_temporal = self._normalize_map(group_temporal)
                normalized_boundary = self._normalize_map(group_ambiguity)
                topk_ratio = max(
                    float(self.adaptive_config.risk_debug_topk_ratio),
                    float(self.adaptive_config.risk_hard_gate_ratio),
                )

                boundary_weights = 1.0 + max(self.adaptive_config.risk_boundary_loss_weight - 1.0, 0.0) * normalized_boundary
                teacher_groups = (
                    self.adaptive_config.risk_teacher_damage_weight * normalized_damage
                    + self.adaptive_config.risk_teacher_error_weight * normalized_error
                    + self.adaptive_config.risk_teacher_uncertainty_weight * normalized_uncertainty
                    + self.adaptive_config.risk_teacher_temporal_weight * normalized_temporal
                    + self.adaptive_config.risk_teacher_boundary_weight * normalized_boundary
                )
                teacher_groups = self._normalize_map(teacher_groups)
                teacher_groups_per_frame.append(teacher_groups)
                predicted_groups_per_frame.append(predicted_groups)
                boundary_weights_per_frame.append(boundary_weights)
                if self.adaptive_config.risk_relative_rank_weight > 0:
                    teacher_positive_mask, teacher_negative_mask = self._score_partition_masks(
                        teacher_groups,
                        float(self.adaptive_config.risk_relative_rank_ratio),
                    )
                    risk_relative_rank = risk_relative_rank + self._binary_partition_ranking_loss(
                        predicted_groups,
                        teacher_positive_mask,
                        teacher_negative_mask,
                    )
                rank_spearman_total = rank_spearman_total + self._rank_spearman(predicted_groups.detach(), teacher_groups.detach())
                topk_overlap_total = topk_overlap_total + self._topk_overlap(predicted_groups.detach(), teacher_groups.detach(), topk_ratio)

                protected_mask = stage["frame_group_protected_mask"][frame_index].to(logits.device)
                merged_mask = stage["frame_merged_group_mask"][frame_index].to(logits.device)
                if bool(protected_mask.any()) and bool(merged_mask.any()):
                    damage_gap_total = damage_gap_total + (group_damage[protected_mask].mean() - group_damage[merged_mask].mean())
                elif bool(protected_mask.any()):
                    damage_gap_total = damage_gap_total + group_damage[protected_mask].mean()
                elif bool(merged_mask.any()):
                    damage_gap_total = damage_gap_total - group_damage[merged_mask].mean()

                boundary_topk = max(1, min(int(normalized_boundary.numel()), int(round(normalized_boundary.numel() * topk_ratio))))
                boundary_indices = torch.topk(normalized_boundary, k=boundary_topk).indices
                boundary_mask = torch.zeros_like(normalized_boundary, dtype=torch.bool)
                boundary_mask[boundary_indices] = True
                boundary_protect_total = boundary_protect_total + protected_mask[boundary_mask].float().mean()

            if predicted_groups_per_frame:
                risk = risk + self._selection_loss(
                    torch.stack(predicted_groups_per_frame, dim=0),
                    torch.stack(teacher_groups_per_frame, dim=0),
                    torch.stack(boundary_weights_per_frame, dim=0),
                )

        num_stages = max(len(adaptive_outputs["stage_outputs"]), 1)
        teacher_damage = teacher_damage_total / max(teacher_frames, 1)
        teacher_error = teacher_error_total / max(teacher_frames, 1)
        risk_stats = {
            "risk_rank_spearman": rank_spearman_total / max(teacher_frames, 1),
            "risk_topk_overlap": topk_overlap_total / max(teacher_frames, 1),
            "risk_damage_gap": damage_gap_total / max(teacher_frames, 1),
            "risk_boundary_protect_rate": boundary_protect_total / max(teacher_frames, 1),
        }
        return (
            risk / num_stages,
            teacher_damage,
            teacher_error,
            risk_relative_rank / max(teacher_frames, 1),
            risk_stats,
        )

    def _existence_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        existence_outputs: dict[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        if existence_outputs is None:
            return logits.new_tensor(0.0)
        existence_logits = existence_outputs["existence_logits"]
        existence_targets = (targets.flatten(1).sum(dim=1) > 0).to(existence_logits.dtype)
        global_shadow = torch.sigmoid(logits).mean(dim=(1, 2, 3, 4))
        existence_prob = torch.sigmoid(existence_logits)
        bce_exist = self.bce(existence_logits, existence_targets)
        negative_penalty = ((1.0 - existence_targets) * global_shadow).mean()
        with torch.autocast(device_type=logits.device.type, enabled=False):
            calibration_prob = F.binary_cross_entropy(
                global_shadow.float(),
                existence_prob.detach().float(),
            )
        calibration = self.bce(existence_logits, global_shadow.detach()) + calibration_prob
        return (
            self.adaptive_config.lambda_exist * bce_exist
            + self.adaptive_config.lambda_exist_negative * negative_penalty
            + self.adaptive_config.lambda_exist_calibration * calibration
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        adaptive_outputs: dict[str, object] | None = None,
        existence_outputs: dict[str, torch.Tensor] | None = None,
        auxiliary_outputs: dict[str, torch.Tensor] | None = None,
        teacher_logits: torch.Tensor | None = None,
        clip: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if logits.shape != targets.shape:
            raise ValueError(f"Shape mismatch: logits={tuple(logits.shape)} targets={tuple(targets.shape)}")

        bce = self._segmentation_bce(logits, targets)
        dice = dice_loss_from_logits(logits, targets)
        tversky = (
            tversky_loss_from_logits(
                logits,
                targets,
                alpha=self.tversky_alpha,
                beta=self.tversky_beta,
            )
            if self.lambda_tversky > 0
            else logits.new_tensor(0.0)
        )
        lovasz = lovasz_hinge_loss_from_logits(logits, targets) if self.lambda_lovasz > 0 else logits.new_tensor(0.0)
        shadow_body = (
            self._shadow_body_recall_loss(logits, targets)
            if self.lambda_shadow_body > 0
            else logits.new_tensor(0.0)
        )
        risk = logits.new_tensor(0.0)
        exist = logits.new_tensor(0.0)
        risk_teacher_damage = logits.new_tensor(0.0)
        risk_teacher_error = logits.new_tensor(0.0)
        risk_distill = logits.new_tensor(0.0)
        risk_relative_rank = logits.new_tensor(0.0)
        distill = logits.new_tensor(0.0)
        auxiliary_total = logits.new_tensor(0.0)
        boundary_main = logits.new_tensor(0.0)
        temporal_consistency = logits.new_tensor(0.0)
        nonshadow_precision = logits.new_tensor(0.0)
        dark_negative = logits.new_tensor(0.0)
        risk_stats: dict[str, torch.Tensor] = {}

        use_risk_aux = (
            adaptive_outputs is not None
            and adaptive_outputs.get("enabled", False)
            and self.adaptive_config.enable_token_merging
        )
        if use_risk_aux:
            runtime_mode = adaptive_outputs.get("runtime_mode")
            if runtime_mode != "risk_v1":
                raise ValueError(f"Unsupported adaptive loss runtime_mode: {runtime_mode!r}")
            risk, risk_teacher_damage, risk_teacher_error, risk_relative_rank, risk_stats = self._risk_losses(
                logits,
                targets,
                adaptive_outputs,
            )
            risk_distill = self._boundary_aware_distill_loss(logits, teacher_logits, targets)

        if self.adaptive_config.enable_existence_calibration:
            exist = self._existence_loss(logits, targets, existence_outputs)
        distill = self._segmentation_distill_loss(logits, teacher_logits, targets)
        auxiliary_total, auxiliary_loss_dict = self._auxiliary_structure_losses(
            auxiliary_outputs,
            targets,
            logits,
        )
        if self.lambda_boundary_main > 0:
            boundary_main = self._main_boundary_loss(logits, targets)
        if self.lambda_temporal_consistency > 0:
            temporal_consistency = self._temporal_consistency_loss(logits, targets)
        if self.lambda_nonshadow_precision > 0:
            nonshadow_precision = self._nonshadow_precision_loss(logits, targets)
        if self.lambda_dark_negative > 0:
            dark_negative = self._dark_negative_loss(logits, targets, clip)

        total = (
            self.lambda_bce * bce
            + self.lambda_dice * dice
            + self.lambda_tversky * tversky
            + self.lambda_lovasz * lovasz
            + self.lambda_shadow_body * shadow_body
            + self.lambda_boundary_main * boundary_main
            + self.lambda_temporal_consistency * temporal_consistency
            + self.lambda_nonshadow_precision * nonshadow_precision
            + self.lambda_dark_negative * dark_negative
            + self.distill_weight * distill
            + self.adaptive_config.lambda_selection * risk
            + self.adaptive_config.risk_distill_weight * risk_distill
            + self.adaptive_config.risk_relative_rank_weight * risk_relative_rank
            + exist
            + auxiliary_total
        )
        loss_dict = {
            "loss_total": total.detach(),
            "loss_bce": bce.detach(),
            "loss_dice": dice.detach(),
            "loss_tversky": tversky.detach(),
            "loss_lovasz": lovasz.detach(),
            "loss_shadow_body": shadow_body.detach(),
            "loss_exist": exist.detach(),
            "loss_distill": distill.detach(),
            "loss_boundary_main": boundary_main.detach(),
            "loss_temporal_consistency": temporal_consistency.detach(),
            "loss_nonshadow_precision": nonshadow_precision.detach(),
            "loss_dark_negative": dark_negative.detach(),
        }
        loss_dict.update(auxiliary_loss_dict)
        if adaptive_outputs is not None and adaptive_outputs.get("runtime_mode") == "risk_v1":
            loss_dict["loss_risk"] = risk.detach()
            loss_dict["loss_risk_distill"] = risk_distill.detach()
            loss_dict["loss_risk_relative_rank"] = risk_relative_rank.detach()
            loss_dict["risk_teacher_damage"] = risk_teacher_damage.detach()
            loss_dict["risk_teacher_error"] = risk_teacher_error.detach()
            for key, value in risk_stats.items():
                loss_dict[key] = value.detach()
        return total, loss_dict


def build_segmentation_loss(
    lambda_bce: float = 1.0,
    positive_bce_weight: float = 1.0,
    lambda_dice: float = 1.0,
    lambda_tversky: float = 0.0,
    tversky_alpha: float = 0.3,
    tversky_beta: float = 0.7,
    lambda_lovasz: float = 0.0,
    lambda_shadow_body: float = 0.0,
    lambda_body_aux: float = 0.0,
    lambda_detail_aux: float = 0.0,
    lambda_boundary_aux: float = 0.0,
    lambda_boundary_main: float = 0.0,
    lambda_temporal_consistency: float = 0.0,
    lambda_nonshadow_precision: float = 0.0,
    lambda_dark_negative: float = 0.0,
    dark_negative_threshold: float = 0.35,
    dark_negative_boundary_kernel: int = 9,
    distill_weight: float = 0.0,
    distill_boundary_weight: float = 2.0,
    distill_bce_weight: float = 1.0,
    distill_dice_weight: float = 0.5,
    distill_logit_weight: float = 0.0,
    adaptive_config: AdaptiveComputeConfig | None = None,
) -> VideoShadowSegLoss:
    return VideoShadowSegLoss(
        lambda_bce=lambda_bce,
        positive_bce_weight=positive_bce_weight,
        lambda_dice=lambda_dice,
        lambda_tversky=lambda_tversky,
        tversky_alpha=tversky_alpha,
        tversky_beta=tversky_beta,
        lambda_lovasz=lambda_lovasz,
        lambda_shadow_body=lambda_shadow_body,
        lambda_body_aux=lambda_body_aux,
        lambda_detail_aux=lambda_detail_aux,
        lambda_boundary_aux=lambda_boundary_aux,
        lambda_boundary_main=lambda_boundary_main,
        lambda_temporal_consistency=lambda_temporal_consistency,
        lambda_nonshadow_precision=lambda_nonshadow_precision,
        lambda_dark_negative=lambda_dark_negative,
        dark_negative_threshold=dark_negative_threshold,
        dark_negative_boundary_kernel=dark_negative_boundary_kernel,
        distill_weight=distill_weight,
        distill_boundary_weight=distill_boundary_weight,
        distill_bce_weight=distill_bce_weight,
        distill_dice_weight=distill_dice_weight,
        distill_logit_weight=distill_logit_weight,
        adaptive_config=adaptive_config,
    )
