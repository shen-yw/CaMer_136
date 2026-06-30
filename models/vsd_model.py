from time import perf_counter

import torch
import torch.nn.functional as F
from torch import nn

from rvsd.configs.baseline import RVSDBaselineConfig
from rvsd.models.backbones import DinoV3ViTBWrapper
from rvsd.models.heads import DecoderSimpleHead, DecoderTokenPureHead, SlimDPTHead
from rvsd.models.temporal import TemporalFeatureFusion


class RVSDVideoShadowModel(nn.Module):
    def __init__(self, config: RVSDBaselineConfig):
        super().__init__()
        backbone_cfg = config.model.backbone
        fusion_cfg = config.model.fusion
        head_cfg = config.model.head
        adaptive_cfg = config.model.adaptive

        decoder_variant = head_cfg.decoder_variant.lower()
        if decoder_variant not in {"decoder_token_pure", "decoder_simple", "slim_dpt"}:
            raise ValueError(f"Unsupported decoder variant: {head_cfg.decoder_variant!r}.")

        self.backbone = DinoV3ViTBWrapper(
            weights_path=backbone_cfg.weights_path,
            out_indices=backbone_cfg.out_indices,
            use_backbone_norm=backbone_cfg.use_backbone_norm,
            train_policy=backbone_cfg.train_policy,
            train_last_n_blocks=backbone_cfg.train_last_n_blocks,
            adaptive_config=adaptive_cfg,
        )
        self.decoder_variant = decoder_variant
        self.backbone.decoder_variant = self.decoder_variant
        self.backbone.decoder_simple_num_dense_sources = int(head_cfg.decoder_simple_num_dense_sources)
        if self.decoder_variant == "decoder_simple":
            self.fusion = TemporalFeatureFusion(
                in_channels=[self.backbone.embed_dim] * int(head_cfg.decoder_simple_num_dense_sources),
                variant=fusion_cfg.variant,
                kernel_size=fusion_cfg.kernel_size,
                use_residual=fusion_cfg.use_residual,
                use_gating=fusion_cfg.use_gating,
                num_layers_per_scale=fusion_cfg.num_layers_per_scale,
                attention_dim=fusion_cfg.attention_dim,
                attention_heads=fusion_cfg.attention_heads,
                attention_dropout=fusion_cfg.attention_dropout,
            )
            self.head = DecoderSimpleHead(
                in_channels=self.backbone.embed_dim,
                decoder_dim=head_cfg.decoder_simple_dim,
                output_dim=head_cfg.output_dim,
                use_batchnorm=head_cfg.use_batchnorm,
                num_dense_sources=head_cfg.decoder_simple_num_dense_sources,
                sparse_coarse_stride=head_cfg.decoder_simple_sparse_coarse_stride,
                sparse_temporal_layers=head_cfg.decoder_simple_sparse_temporal_layers,
            )
        elif self.decoder_variant == "decoder_token_pure":
            self.fusion = None
            self.head = DecoderTokenPureHead(
                in_channels=self.backbone.embed_dim,
                decoder_dim=head_cfg.decoder_token_dim,
                output_dim=head_cfg.output_dim,
                use_batchnorm=head_cfg.use_batchnorm,
                sparse_coarse_stride=head_cfg.decoder_token_sparse_coarse_stride,
                sparse_temporal_layers=head_cfg.decoder_token_sparse_temporal_layers,
                use_dark_guidance=head_cfg.decoder_token_use_dark_guidance,
                use_structure_decoupling=head_cfg.decoder_token_use_structure_decoupling,
                use_structure_aux=head_cfg.decoder_token_use_structure_aux,
            )
        else:
            self.fusion = TemporalFeatureFusion(
                in_channels=[self.backbone.embed_dim] * len(backbone_cfg.out_indices),
                variant=fusion_cfg.variant,
                kernel_size=fusion_cfg.kernel_size,
                use_residual=fusion_cfg.use_residual,
                use_gating=fusion_cfg.use_gating,
                num_layers_per_scale=fusion_cfg.num_layers_per_scale,
                attention_dim=fusion_cfg.attention_dim,
                attention_heads=fusion_cfg.attention_heads,
                attention_dropout=fusion_cfg.attention_dropout,
            )
            self.head = SlimDPTHead(
                in_channels=[self.backbone.embed_dim] * len(backbone_cfg.out_indices),
                decoder_dim=head_cfg.decoder_dim,
                output_dim=head_cfg.output_dim,
                use_batchnorm=head_cfg.use_batchnorm,
                detail_refine_layers=head_cfg.decoder_detail_refine_layers,
                context_refine_dilation=head_cfg.decoder_context_refine_dilation,
            )
        self.classifier = nn.Conv2d(head_cfg.output_dim, 1, kernel_size=1)
        self.enable_speed_profile = False
        self.last_speed_profile: dict[str, float] = {}

    def _profile_marker(self, device: torch.device):
        if not self.enable_speed_profile:
            return None
        if device.type == "cuda":
            event = torch.cuda.Event(enable_timing=True)
            event.record(torch.cuda.current_stream(device))
            return event
        return perf_counter()

    def _profile_elapsed_ms(self, start_marker, end_marker, device: torch.device) -> float:
        if start_marker is None or end_marker is None:
            return 0.0
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            return float(start_marker.elapsed_time(end_marker))
        return float((end_marker - start_marker) * 1000.0)

    def forward(self, clip: torch.Tensor, return_outputs: bool = False):
        batch_size, _, num_frames, image_height, image_width = clip.shape
        backbone_outputs = self.backbone(clip, collect_adaptive_outputs=return_outputs)
        sparse_decoder_inputs = backbone_outputs.get("sparse_decoder_inputs")
        adaptive_outputs = backbone_outputs["adaptive_outputs"]
        existence_outputs = backbone_outputs["existence_outputs"]
        if self.decoder_variant != "slim_dpt" and sparse_decoder_inputs is None:
            raise RuntimeError("sparse_decoder_inputs missing from backbone outputs.")

        decoder_start = self._profile_marker(clip.device)
        if self.decoder_variant == "decoder_simple":
            dense_feature_maps = sparse_decoder_inputs["dense_feature_maps"]
            if self.fusion is None:
                raise RuntimeError("fusion module missing for decoder_simple.")
            fused_feature_maps = self.fusion(dense_feature_maps)
            head_features = self.head(
                dense_feature_maps=fused_feature_maps,
                sparse_patch_tokens=sparse_decoder_inputs["sparse_patch_tokens"],
                sparse_patch_coords=sparse_decoder_inputs["sparse_patch_coords"],
                sparse_token_sizes=sparse_decoder_inputs["sparse_token_sizes"],
            )
            auxiliary_outputs = None
            fusion_output_shape = tuple(fused_feature_maps[-1].shape)
        elif self.decoder_variant == "decoder_token_pure":
            head_output = self.head(
                sparse_patch_tokens=sparse_decoder_inputs["sparse_patch_tokens"],
                sparse_patch_coords=sparse_decoder_inputs["sparse_patch_coords"],
                sparse_token_sizes=sparse_decoder_inputs["sparse_token_sizes"],
                restore_index=sparse_decoder_inputs.get("restore_index"),
                patch_hw=sparse_decoder_inputs["patch_hw"],
                batch_size=batch_size,
                num_frames=num_frames,
                dark_prior_map=sparse_decoder_inputs.get("dark_prior_map"),
            )
            auxiliary_outputs = head_output.get("auxiliary") or None
            head_features = head_output["features"]
            fusion_output_shape = tuple()
        else:
            backbone_feature_maps = backbone_outputs["feature_maps"]
            fused_feature_maps = self.fusion(backbone_feature_maps)
            fused_feature_maps_flat = [
                feature_map.reshape(
                    batch_size * num_frames,
                    feature_map.shape[2],
                    feature_map.shape[3],
                    feature_map.shape[4],
                )
                for feature_map in fused_feature_maps
            ]
            head_features = self.head(fused_feature_maps_flat)
            auxiliary_outputs = None
            fusion_output_shape = tuple(fused_feature_maps[0].shape)
        logits = self.classifier(head_features)
        logits = F.interpolate(logits, size=(image_height, image_width), mode="bilinear", align_corners=False)
        logits = logits.reshape(batch_size, num_frames, 1, image_height, image_width)
        if auxiliary_outputs is not None:
            resized_auxiliary_outputs = {}
            for key, value in auxiliary_outputs.items():
                resized = F.interpolate(value, size=(image_height, image_width), mode="bilinear", align_corners=False)
                resized_auxiliary_outputs[key] = resized.reshape(batch_size, num_frames, 1, image_height, image_width)
            auxiliary_outputs = resized_auxiliary_outputs
        decoder_end = self._profile_marker(clip.device)
        if self.enable_speed_profile:
            backbone_profile = getattr(self.backbone, "last_speed_profile", {})
            self.last_speed_profile = {key: float(value) for key, value in backbone_profile.items()}
            self.last_speed_profile["decoder_head_ms"] = self._profile_elapsed_ms(decoder_start, decoder_end, clip.device)

        if not return_outputs:
            return logits

        outputs = {
            "debug": {
                "input_shape": tuple(clip.shape),
                "backbone_feature_shapes": [tuple(feature_map.shape) for feature_map in backbone_outputs["feature_maps"]],
                "fusion_output_shape": fusion_output_shape,
                "head_output_shape": tuple(head_features.shape),
                "final_logits_shape": tuple(logits.shape),
                "decoder_variant": self.decoder_variant,
            },
            "adaptive": adaptive_outputs,
            "existence": existence_outputs,
        }
        if sparse_decoder_inputs is not None:
            outputs["debug"]["sparse_decoder_dense_shapes"] = [
                tuple(feature_map.shape) for feature_map in sparse_decoder_inputs["dense_feature_maps"]
            ]
            outputs["debug"]["sparse_decoder_dense_source_indices"] = tuple(sparse_decoder_inputs["dense_source_indices"])
            outputs["debug"]["sparse_decoder_sparse_tokens_shape"] = tuple(sparse_decoder_inputs["sparse_patch_tokens"].shape)
            outputs["debug"]["sparse_decoder_sparse_coords_shape"] = tuple(sparse_decoder_inputs["sparse_patch_coords"].shape)
            outputs["debug"]["sparse_decoder_sparse_sizes_shape"] = tuple(sparse_decoder_inputs["sparse_token_sizes"].shape)
            if sparse_decoder_inputs.get("restore_index") is not None:
                outputs["debug"]["sparse_decoder_restore_index_shape"] = tuple(
                    sparse_decoder_inputs["restore_index"].shape
                )
            outputs["debug"]["sparse_decoder_patch_hw"] = tuple(sparse_decoder_inputs["patch_hw"])
            if sparse_decoder_inputs.get("dark_prior_map") is not None:
                outputs["debug"]["sparse_decoder_dark_prior_shape"] = tuple(sparse_decoder_inputs["dark_prior_map"].shape)
        if self.decoder_variant == "decoder_simple":
            outputs["debug"]["decoder_simple_num_dense_sources"] = int(self.backbone.decoder_simple_num_dense_sources)
        if auxiliary_outputs is not None:
            outputs["auxiliary"] = auxiliary_outputs
            outputs["debug"]["auxiliary_output_shapes"] = {
                key: tuple(value.shape) for key, value in auxiliary_outputs.items()
            }
        if adaptive_outputs is not None:
            outputs["debug"]["keep_ratios"] = adaptive_outputs["keep_ratios_summary"]
            outputs["debug"]["merge_block_indices"] = tuple(int(index) for index in adaptive_outputs["merge_block_indices"])
            for key in (
                "runtime_mode",
                "tail_stop_block_index",
                "requested_tail_stop_block_index",
                "skipped_tail_blocks",
                "risk_hard_gate_ratio",
                "risk_dynamic_keep_enabled",
                "risk_dynamic_hard_gate_enabled",
                "stage_merge_stats",
                "transient_merge_block_indices",
                "persistent_merge_block_index",
                "requested_stage_keep_ratios",
                "requested_stage_group_keep_ratios",
                "fixed_keep_ratio_is_token_keep",
                "effective_merge_applied",
                "final_token_keep_ratio",
                "final_token_saved_ratio",
                "final_interface_token_keep_ratio",
                "execution_granularity",
                "merge_semantics",
                "risk_merge_strategy",
                "group_contract_extension_enabled",
                "attention_bias_correction_enabled",
                "merge_correction_enabled",
                "independent_compressibility_enabled",
                "task_aware_preservation_enabled",
                "token_transform_reducer_enabled",
                "temporal_consensus_correction_enabled",
                "post_block_indices",
                "post_block_token_counts",
                "post_block_all_compact",
            ):
                if key in adaptive_outputs:
                    outputs["debug"][key] = adaptive_outputs[key]
        if existence_outputs is not None:
            outputs["debug"]["existence_logits_shape"] = tuple(existence_outputs["existence_logits"].shape)
        return logits, outputs


def build_vsd_model(config: RVSDBaselineConfig) -> RVSDVideoShadowModel:
    return RVSDVideoShadowModel(config)
