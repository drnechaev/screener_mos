from typing import Tuple, Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers.helpers import to_3tuple, to_2tuple

from medimm.utils import eval_mode

from screener.descriptor_model import DescriptorModel
from .anomaly_segm import AnomalySegm


class ActNorm(nn.Module):
    def __init__(self, in_channels: int, mode: str = "3d"):
        super().__init__()

        self.loc = nn.Parameter(torch.zeros(in_channels))
        self.log_scale = nn.Parameter(torch.zeros(in_channels))
        self.mode = mode
        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))

    @torch.no_grad()
    def initialize(self, x: torch.Tensor) -> None:
        if self.mode == "3d":
            mean = x.mean(dim=(0, 2, 3, 4))
            std = x.std(dim=(0, 2, 3, 4))
        elif self.mode == "2d":
            mean = x.mean(dim=(0, 2, 3))
            std = x.std(dim=(0, 2, 3))
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        self.loc.data.copy_(-mean)
        self.log_scale.data.copy_(-torch.log(std + 1e-6))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.initialized.item() == 0:
            self.initialize(x)
            self.initialized.fill_(1)

        if self.mode == "3d":
            loc, log_scale = self.loc.view(1, -1, 1, 1, 1), self.log_scale.view(1, -1, 1, 1, 1)
        elif self.mode == "2d":
            loc, log_scale = self.loc.view(1, -1, 1, 1), self.log_scale.view(1, -1, 1, 1)
        x = (x + loc) * torch.exp(log_scale)
        logdetjac = self.log_scale.sum()

        return x, logdetjac


class Inv1x1Conv(nn.Module):
    def __init__(self, in_channels: int, mode: str = "3d") -> None:
        super().__init__()
        
        self.mode = mode
        q, _ = torch.linalg.qr(torch.randn(in_channels, in_channels))

        self.mode = self.mode
        if self.mode == "3d":
            self.weight = nn.Parameter(q.view(in_channels, in_channels, 1, 1, 1))
        elif self.mode == "2d":
            self.weight = nn.Parameter(q.view(in_channels, in_channels, 1, 1))
        else:
            raise ValueError(f"Unknown mode: {self.mode}")


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.mode == "3d":
            x = F.conv3d(x, self.weight)
        elif self.mode == "2d":
            x = F.conv2d(x, self.weight)

        # Cast to float64 from original OpenAI implementation:
        # https://github.com/openai/glow/blob/master/model.py#L454
        dtype = self.weight.dtype
        logdetjac = torch.slogdet(self.weight.squeeze().double())[1].to(dtype=dtype)

        return x, logdetjac


class AffineCoupling(nn.Module):
    def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            mode: str = "3d"
    ) -> None:
        super().__init__()

        c_1 = in_channels // 2
        c_2 = in_channels - c_1
        self.mode = mode

        if self.mode == "3d":
            self.mlp = nn.Sequential(
                nn.Conv3d(c_1, hidden_channels, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(hidden_channels, hidden_channels, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(hidden_channels, c_2 * 2, kernel_size=1),
            )
        elif self.mode == "2d":
            self.mlp = nn.Sequential(
                nn.Conv2d(c_1, hidden_channels, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_channels, c_2 * 2, kernel_size=1),
            )
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        self.mlp[-1].weight.data.zero_()
        self.mlp[-1].bias.data.zero_()
        self.c_1 = c_1

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_1, x_2 = x[:, :self.c_1], x[:, self.c_1:]

        s_logit, t = self.mlp(x_1).chunk(2, dim=1)
        s = F.sigmoid(s_logit + 2.0)  # (n, c_2, h, w, d) or (n, c_2, h, w)
        x_2 = (x_2 + t) * s

        logdetjac = torch.sum(F.logsigmoid(s_logit + 2.0), dim=1)  # (n, h, w, d) or (n, h, w)

        return torch.cat([x_1, x_2], dim=1), logdetjac


class GlowBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            mode: str = "3d"
    ) -> None:
        super().__init__()

        self.actnorm = ActNorm(in_channels, mode=mode)
        self.inv_linear = Inv1x1Conv(in_channels, mode=mode)
        self.affine_coupling = AffineCoupling(in_channels, hidden_channels, mode=mode)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x, logdetjac_1 = self.actnorm(x)
        x, logdetjac_2 = self.inv_linear(x)
        x, logdetjac_3 = self.affine_coupling(x)

        logdetjac = logdetjac_1 + logdetjac_2 + logdetjac_3

        return x, logdetjac


class Glow(AnomalySegm):
    def __init__(
            self,
            mode: str = "3d",
            descriptor_model_path: str = None,
            in_channels: int = 1,
            descriptor_dim: int = 32,
            avg_pool: bool = False,
            avg_pool_kernel_size: Optional[Tuple[int, int, int]] = None,
            avg_pool_stride: Optional[Tuple[int, int, int]] = None,
            avg_pool_padding: Optional[Tuple[int, int, int]] = None,
            sigma: float = 0.1,
            glow_hidden_dim: int = 512,
            glow_depth: int = 64,
            crop_size: Tuple[int, int, int] = (96, 96, 96),
            sw_batch_size: int = 4,
            lr: float = 3e-4,
            weight_decay: float = 1e-6
    ) -> None:
        super().__init__(crop_size, sw_batch_size, lr, weight_decay)

        self.mode = mode
        self.descriptor_model = DescriptorModel(
            in_channels=in_channels,
            descriptor_dim=descriptor_dim,
            mode=mode,
            pretrained_model_path=descriptor_model_path,
        )
        self.glow_blocks = nn.ModuleList([
            GlowBlock(descriptor_dim, glow_hidden_dim, mode=self.mode)
            for _ in range(glow_depth)
        ])

        self.descriptor_dim = descriptor_dim
        self.avg_pool = avg_pool
        self.avg_pool_kernel_size = avg_pool_kernel_size
        self.avg_pool_stride = avg_pool_stride
        self.avg_pool_padding = avg_pool_padding
        self.sigma = sigma

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        with torch.no_grad(), eval_mode(self.descriptor_model):
            x = self.descriptor_model(image)
        if self.avg_pool:
            if self.mode == "3d":
                x = F.avg_pool3d(
                    x,
                    kernel_size=to_3tuple(self.avg_pool_kernel_size),
                    stride=to_3tuple(self.avg_pool_stride),
                    padding=to_3tuple(self.avg_pool_padding)
                )
            elif self.mode == "2d":
                x = F.avg_pool2d(
                    x,
                    kernel_size=to_2tuple(self.avg_pool_kernel_size),
                    stride=to_2tuple(self.avg_pool_stride),
                    padding=to_2tuple(self.avg_pool_padding)
                )
        if self.sigma > 0 and self.training:
            x += torch.randn_like(x) * self.sigma

        glow_ldj = 0.0
        for block in self.glow_blocks:
            x, block_ldj = block(x)  # (N, D, H, W, S), (N, H, W, S)
            glow_ldj += block_ldj

        return x.pow(2).sum(dim=1).div(2.0) + self.descriptor_dim / 2 * math.log(2 * math.pi) - glow_ldj

    def training_step(self, batch, batch_idx):
        nll_map = self.forward(batch['image'])
        # roi_mask = batch['roi_mask']
        # roi_mask = roi_mask.float().unsqueeze(1)
        # roi_mask = F.interpolate(roi_mask, size=nll_map.shape[-3:], mode='trilinear')
        # roi_mask = roi_mask.squeeze(1) > 0.5
        # nll = nll_map[roi_mask].mean()
        nll = nll_map.mean()
        self.log('train/nll_loss', nll, on_step=True, on_epoch=True)

        return nll

    @torch.no_grad()
    def _per_crop_predictor(self, image):
        anomaly_map = self.forward(image)
        anomaly_map = anomaly_map.unsqueeze(1)
        anomaly_map = F.interpolate(anomaly_map, size=image.shape[-3:], mode='trilinear')
        return anomaly_map
