import torch
from torch import nn


class TemporalFusionBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        use_residual: bool = True,
        use_gating: bool = True,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.use_residual = use_residual
        self.use_gating = use_gating
        self.pre_norm = nn.BatchNorm3d(channels)
        self.depthwise_temporal = nn.Conv3d(
            channels,
            channels,
            kernel_size=(kernel_size, 1, 1),
            padding=(padding, 0, 0),
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.post_norm = nn.BatchNorm3d(channels)
        self.act = nn.GELU()
        self.gate = nn.Conv3d(channels, channels, kernel_size=1, bias=True) if use_gating else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected [B, C, T, H, W], got {tuple(x.shape)}")

        residual = x
        x = self.pre_norm(x)
        x = self.depthwise_temporal(x)
        x = self.pointwise(x)
        x = self.post_norm(x)
        x = self.act(x)

        if self.gate is not None:
            x = x * torch.sigmoid(self.gate(x))

        if self.use_residual:
            x = x + residual
        return x.contiguous()

    def _depthwise_temporal_selected(self, x: torch.Tensor, selected_indices: list[int]) -> torch.Tensor:
        batch_size, channels, _num_frames, height, width = x.shape
        weight = self.depthwise_temporal.weight[:, 0, :, 0, 0]
        padding = int(self.depthwise_temporal.padding[0])
        kernel_size = int(weight.shape[1])
        outputs = []
        for time_index in selected_indices:
            selected = x.new_zeros((batch_size, channels, height, width))
            for kernel_index in range(kernel_size):
                source_index = int(time_index) + kernel_index - padding
                if 0 <= source_index < x.shape[2]:
                    selected = selected + x[:, :, source_index] * weight[:, kernel_index].view(1, channels, 1, 1)
            outputs.append(selected)
        return torch.stack(outputs, dim=2)

    def forward_selected(self, x: torch.Tensor, selected_indices: list[int]) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected [B, C, T, H, W], got {tuple(x.shape)}")
        if self.training:
            return self.forward(x)[:, :, selected_indices]
        if len(selected_indices) == x.shape[2]:
            return self.forward(x)

        selected_input = x[:, :, selected_indices]
        conv = self.pre_norm(x)
        conv = self._depthwise_temporal_selected(conv, selected_indices)
        conv = self.pointwise(conv)
        conv = self.post_norm(conv)
        conv = self.act(conv)

        if self.gate is not None:
            conv = conv * torch.sigmoid(self.gate(selected_input))

        if self.use_residual:
            conv = conv + selected_input
        return conv.contiguous()


class TemporalContextFusionBlock(nn.Module):
    """
    Lightweight temporal context fusion for dense video features.

    The convolution branch keeps the old local temporal smoothing behavior.
    The attention branch mixes the T features at each spatial location through
    a low-dimensional bottleneck, so it adds temporal context without changing
    token count or becoming part of the token-reduction method.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        use_residual: bool = True,
        use_gating: bool = True,
        attention_dim: int = 128,
        attention_heads: int = 4,
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        if attention_dim < 1:
            raise ValueError(f"attention_dim must be >= 1, got {attention_dim}")
        if attention_heads < 1:
            raise ValueError(f"attention_heads must be >= 1, got {attention_heads}")
        if attention_dim % attention_heads != 0:
            raise ValueError(
                f"attention_dim must be divisible by attention_heads, got {attention_dim} and {attention_heads}"
            )
        padding = kernel_size // 2
        self.use_residual = use_residual
        self.use_gating = use_gating

        self.pre_norm = nn.BatchNorm3d(channels)
        self.depthwise_temporal = nn.Conv3d(
            channels,
            channels,
            kernel_size=(kernel_size, 1, 1),
            padding=(padding, 0, 0),
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.post_norm = nn.BatchNorm3d(channels)
        self.act = nn.GELU()
        self.conv_gate = nn.Conv3d(channels, channels, kernel_size=1, bias=True) if use_gating else None

        self.attn_norm = nn.LayerNorm(channels)
        self.attn_down = nn.Linear(channels, attention_dim)
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=attention_dim,
            num_heads=attention_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.attn_up = nn.Linear(attention_dim, channels)
        self.attn_gate = nn.Conv3d(channels, channels, kernel_size=1, bias=True) if use_gating else None
        self.out_norm = nn.BatchNorm3d(channels)

    def _attention_branch(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, num_frames, height, width = x.shape
        sequence = x.permute(0, 3, 4, 2, 1).reshape(batch_size * height * width, num_frames, channels)
        sequence = self.attn_down(self.attn_norm(sequence))
        attended, _ = self.temporal_attn(sequence, sequence, sequence, need_weights=False)
        attended = self.attn_up(attended)
        return attended.reshape(batch_size, height, width, num_frames, channels).permute(0, 4, 3, 1, 2).contiguous()

    def _depthwise_temporal_selected(self, x: torch.Tensor, selected_indices: list[int]) -> torch.Tensor:
        batch_size, channels, _num_frames, height, width = x.shape
        weight = self.depthwise_temporal.weight[:, 0, :, 0, 0]
        padding = int(self.depthwise_temporal.padding[0])
        kernel_size = int(weight.shape[1])
        outputs = []
        for time_index in selected_indices:
            selected = x.new_zeros((batch_size, channels, height, width))
            for kernel_index in range(kernel_size):
                source_index = int(time_index) + kernel_index - padding
                if 0 <= source_index < x.shape[2]:
                    selected = selected + x[:, :, source_index] * weight[:, kernel_index].view(1, channels, 1, 1)
            outputs.append(selected)
        return torch.stack(outputs, dim=2)

    def _attention_branch_selected(self, x: torch.Tensor, selected_indices: list[int]) -> torch.Tensor:
        batch_size, channels, num_frames, height, width = x.shape
        sequence = x.permute(0, 3, 4, 2, 1).reshape(batch_size * height * width, num_frames, channels)
        sequence = self.attn_down(self.attn_norm(sequence))
        query = sequence[:, selected_indices]
        attended, _ = self.temporal_attn(query, sequence, sequence, need_weights=False)
        attended = self.attn_up(attended)
        selected_frames = len(selected_indices)
        return (
            attended.reshape(batch_size, height, width, selected_frames, channels)
            .permute(0, 4, 3, 1, 2)
            .contiguous()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected [B, C, T, H, W], got {tuple(x.shape)}")

        residual = x
        conv = self.pre_norm(x)
        conv = self.depthwise_temporal(conv)
        conv = self.pointwise(conv)
        conv = self.post_norm(conv)
        conv = self.act(conv)
        if self.conv_gate is not None:
            conv = conv * torch.sigmoid(self.conv_gate(x))

        attn = self._attention_branch(x)
        if self.attn_gate is not None:
            attn = attn * torch.sigmoid(self.attn_gate(x))

        x = self.out_norm(conv + attn)
        x = self.act(x)
        if self.use_residual:
            x = x + residual
        return x.contiguous()

    def forward_selected(self, x: torch.Tensor, selected_indices: list[int]) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected [B, C, T, H, W], got {tuple(x.shape)}")
        if self.training:
            return self.forward(x)[:, :, selected_indices]
        if len(selected_indices) == x.shape[2]:
            return self.forward(x)

        selected_input = x[:, :, selected_indices]
        conv = self.pre_norm(x)
        conv = self._depthwise_temporal_selected(conv, selected_indices)
        conv = self.pointwise(conv)
        conv = self.post_norm(conv)
        conv = self.act(conv)
        if self.conv_gate is not None:
            conv = conv * torch.sigmoid(self.conv_gate(selected_input))

        attn = self._attention_branch_selected(x, selected_indices)
        if self.attn_gate is not None:
            attn = attn * torch.sigmoid(self.attn_gate(selected_input))

        x = self.out_norm(conv + attn)
        x = self.act(x)
        if self.use_residual:
            x = x + selected_input
        return x.contiguous()


class TemporalFeatureFusion(nn.Module):
    def __init__(
        self,
        in_channels: list[int],
        variant: str = "conv",
        kernel_size: int = 3,
        use_residual: bool = True,
        use_gating: bool = True,
        num_layers_per_scale: int = 2,
        attention_dim: int = 128,
        attention_heads: int = 4,
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        if num_layers_per_scale < 1:
            raise ValueError(f"num_layers_per_scale must be >= 1, got {num_layers_per_scale}")
        normalized_variant = variant.lower()
        if normalized_variant not in {"conv", "context_attn"}:
            raise ValueError(f"Unsupported temporal fusion variant: {variant!r}")
        block_cls = TemporalFusionBlock if normalized_variant == "conv" else TemporalContextFusionBlock
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        block_cls(
                            channels=channels,
                            kernel_size=kernel_size,
                            use_residual=use_residual,
                            use_gating=use_gating,
                            **(
                                {
                                    "attention_dim": attention_dim,
                                    "attention_heads": attention_heads,
                                    "attention_dropout": attention_dropout,
                                }
                                if normalized_variant == "context_attn"
                                else {}
                            ),
                        )
                        for _ in range(num_layers_per_scale)
                    ]
                )
                for channels in in_channels
            ]
        )

    def forward(self, feature_maps: list[torch.Tensor]) -> list[torch.Tensor]:
        if len(feature_maps) != len(self.blocks):
            raise ValueError(f"Expected {len(self.blocks)} feature levels, got {len(feature_maps)}")
        outputs = []
        for block, feature_map in zip(self.blocks, feature_maps):
            if feature_map.ndim != 5:
                raise ValueError(f"Expected [B, T, C, H, W], got {tuple(feature_map.shape)}")
            fused = block(feature_map.permute(0, 2, 1, 3, 4).contiguous())
            outputs.append(fused.permute(0, 2, 1, 3, 4).contiguous())
        return outputs

    def forward_selected(self, feature_maps: list[torch.Tensor], selected_indices: list[int]) -> list[torch.Tensor]:
        if len(feature_maps) != len(self.blocks):
            raise ValueError(f"Expected {len(self.blocks)} feature levels, got {len(feature_maps)}")
        outputs = []
        for block, feature_map in zip(self.blocks, feature_maps):
            if feature_map.ndim != 5:
                raise ValueError(f"Expected [B, T, C, H, W], got {tuple(feature_map.shape)}")
            x = feature_map.permute(0, 2, 1, 3, 4).contiguous()
            if len(block) == 1 and hasattr(block[0], "forward_selected"):
                fused = block[0].forward_selected(x, selected_indices)
            else:
                fused = block(x)[:, :, selected_indices]
            outputs.append(fused.permute(0, 2, 1, 3, 4).contiguous())
        return outputs
