import torch
import torch.nn.functional as F
from torch import nn


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_batchnorm: bool,
        kernel_size: int = 3,
        dilation: int = 1,
        groups: int = 1,
    ):
        padding = dilation * (kernel_size // 2)
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=not use_batchnorm,
            ),
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.GELU())
        super().__init__(*layers)


class TopDownRefineBlock(nn.Module):
    def __init__(self, channels: int, use_batchnorm: bool):
        super().__init__()
        self.skip_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.refine = nn.Sequential(
            ConvNormAct(channels, channels, use_batchnorm=use_batchnorm),
            ConvNormAct(channels, channels, use_batchnorm=use_batchnorm),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        if skip is not None:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = x + self.skip_proj(skip)
        return self.refine(x)


class DetailContextRefineBlock(nn.Module):
    def __init__(self, channels: int, use_batchnorm: bool, context_dilation: int = 2):
        super().__init__()
        context_dilation = max(int(context_dilation), 1)
        self.detail = nn.Sequential(
            ConvNormAct(channels, channels, use_batchnorm=use_batchnorm, groups=channels),
            ConvNormAct(channels, channels, use_batchnorm=use_batchnorm, kernel_size=1),
        )
        self.context = nn.Sequential(
            ConvNormAct(
                channels,
                channels,
                use_batchnorm=use_batchnorm,
                dilation=context_dilation,
                groups=channels,
            ),
            ConvNormAct(channels, channels, use_batchnorm=use_batchnorm, kernel_size=1),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.out = ConvNormAct(channels, channels, use_batchnorm=use_batchnorm, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        detail = self.detail(x)
        context = self.context(x)
        gate = self.gate(x)
        return self.out(x + gate * detail + (1.0 - gate) * context)


class SlimDPTHead(nn.Module):
    """
    DPT-style dense head kept for strong dense supervision / teacher use.
    """

    def __init__(
        self,
        in_channels: list[int],
        decoder_dim: int = 256,
        output_dim: int = 128,
        use_batchnorm: bool = True,
        detail_refine_layers: int = 0,
        context_refine_dilation: int = 2,
    ):
        super().__init__()
        self.projections = nn.ModuleList([nn.Conv2d(channel, decoder_dim, kernel_size=1) for channel in in_channels])
        self.scale_adapters = nn.ModuleList(
            [
                nn.ConvTranspose2d(decoder_dim, decoder_dim, kernel_size=4, stride=4),
                nn.ConvTranspose2d(decoder_dim, decoder_dim, kernel_size=2, stride=2),
                nn.Identity(),
                nn.Conv2d(decoder_dim, decoder_dim, kernel_size=3, stride=2, padding=1),
            ]
        )
        self.refine_blocks = nn.ModuleList(
            [TopDownRefineBlock(decoder_dim, use_batchnorm=use_batchnorm) for _ in range(4)]
        )
        self.output_projection = nn.Sequential(
            ConvNormAct(decoder_dim, decoder_dim, use_batchnorm=use_batchnorm),
            ConvNormAct(decoder_dim, output_dim, use_batchnorm=use_batchnorm),
        )
        self.detail_refine = nn.Sequential(
            *[
                DetailContextRefineBlock(
                    output_dim,
                    use_batchnorm=use_batchnorm,
                    context_dilation=context_refine_dilation,
                )
                for _ in range(max(int(detail_refine_layers), 0))
            ]
        )

    def forward(self, feature_maps: list[torch.Tensor]) -> torch.Tensor:
        if len(feature_maps) != 4:
            raise ValueError(f"Expected 4 feature levels, got {len(feature_maps)}")

        pyramid = []
        for feature_map, projection, adapter in zip(feature_maps, self.projections, self.scale_adapters):
            pyramid.append(adapter(projection(feature_map)))

        x = self.refine_blocks[0](pyramid[-1])
        x = self.refine_blocks[1](x, pyramid[-2])
        x = self.refine_blocks[2](x, pyramid[-3])
        x = self.refine_blocks[3](x, pyramid[-4])
        x = self.output_projection(x)
        return self.detail_refine(x)
