from typing import Sequence, Tuple, Any, Union, Literal
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import OptimizerLRScheduler

from screener.descriptor_model import DescriptorModel


class DenseVICReg(pl.LightningModule):
    def __init__(
            self,
            in_channels: int = 1,
            descriptor_dim: int = 32,
            i_weight: float = 25.0,
            v_weight: float = 25.0,
            c_weight: float = 1.0,
            lr: float = 3e-4,
            weight_decay: float = 1e-6,
    ) -> None:
        super().__init__()

        self.descriptor_model = DescriptorModel(
            in_channels=in_channels,
            descriptor_dim=descriptor_dim
        )
        self.projector = nn.Sequential(
            nn.Linear(descriptor_dim, 8192),
            nn.GroupNorm(32, 8192),
            nn.SiLU(inplace=True),
            nn.Linear(8192, 8192),
            nn.GroupNorm(32, 8192),
            nn.SiLU(inplace=True),
            nn.Linear(8192, 8192, bias=False)
        )
        self.i_weight = i_weight
        self.v_weight = v_weight
        self.c_weight = c_weight
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.descriptor_model(image)

    def forward_voxel_embeds(self, image: torch.Tensor, voxel_indices: Sequence[torch.Tensor]) -> torch.Tensor:
        feature_maps = self.descriptor_model(image)
        features = batched_take_features_from_maps(feature_maps, voxel_indices, mode='nearest')
        embeds = self.projector(features)
        return embeds

    def training_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        embeds_1 = self.forward_voxel_embeds(batch['image_1'], batch['voxel_indices_1'])
        embeds_2 = self.forward_voxel_embeds(batch['image_2'], batch['voxel_indices_2'])
        n, d = embeds_1.shape

        i_reg = F.mse_loss(embeds_1, embeds_2)
        self.log(f'train/vicreg_i_term', i_reg, on_epoch=True, on_step=True)

        embeds_1 = embeds_1 - embeds_1.mean(dim=0)
        embeds_2 = embeds_2 - embeds_2.mean(dim=0)

        eps = 1e-4
        v_reg_1 = torch.mean(F.relu(1 - torch.sqrt(embeds_1.var(dim=0) + eps)))
        v_reg_2 = torch.mean(F.relu(1 - torch.sqrt(embeds_2.var(dim=0) + eps)))
        v_reg = (v_reg_1 + v_reg_2) / 2
        self.log(f'train/vicreg_v_term', v_reg, on_epoch=True, on_step=True)

        c_reg_1 = off_diagonal(embeds_1.T @ embeds_1).div(n - 1).pow_(2).sum().div(d)
        c_reg_2 = off_diagonal(embeds_2.T @ embeds_2).div(n - 1).pow_(2).sum().div(d)
        c_reg = (c_reg_1 + c_reg_2) / 2
        self.log(f'train/vicreg_c_term', c_reg, on_epoch=True, on_step=True)

        loss = (
            self.i_weight * i_reg
            + self.v_weight * v_reg
            + self.c_weight * c_reg
        )
        self.log(f'train/vicreg_loss', loss, on_epoch=True, on_step=True)

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        pass

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.AdamW(
            params=self.parameters(),
            lr=self.lr,
            eps=1e-3,
            weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.1,
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
            }
        }


def take_features_from_maps(
        feature_maps: torch.Tensor,
        voxel_indices: torch.Tensor,
        stride: Union[int, Tuple[int, int, int]] = 1,
        mode: Literal['nearest', 'trilinear'] = 'trilinear',
) -> torch.Tensor:
    if stride == 1:
        return feature_maps.movedim(0, -1)[voxel_indices.unbind(1)]

    stride = torch.tensor(stride).to(voxel_indices)
    min_indices = torch.tensor(0).to(voxel_indices)
    max_indices = torch.tensor(feature_maps.shape[-3:]).to(voxel_indices) - 1
    if mode == 'nearest':
        indices = voxel_indices // stride
        indices = torch.clamp(indices, min_indices, max_indices)
        return feature_maps.movedim(0, -1)[indices.unbind(1)]
    elif mode == 'trilinear':
        x = feature_maps.movedim(0, -1)
        points = (voxel_indices + 0.5) / stride - 0.5
        starts = torch.floor(points).long()  # (n, 3)
        stops = starts + 1  # (n, 3)
        f = 0.0
        for mask in itertools.product((0, 1), repeat=3):
            mask = torch.tensor(mask, device=voxel_indices.device, dtype=bool)
            corners = torch.where(mask, starts, stops)  # (n, 3)
            corners = torch.clamp(corners, min_indices, max_indices)  # (n, 3)
            weights = torch.prod(torch.where(mask, 1 - (points - starts), 1 - (stops - points)), dim=-1, keepdim=True)  # (n, 1)
            f = f + weights.to(x) * x[corners.unbind(-1)]  # (n, d)
        return f
    else:
        raise ValueError(mode)


def batched_take_features_from_maps(
        feature_maps_batch: torch.Tensor,
        voxel_indices_batch: Sequence[torch.Tensor],
        stride: Union[int, Tuple[int, int, int]] = 1,
        mode: Literal['nearest', 'trilinear'] = 'trilinear',
) -> torch.Tensor:
    return torch.cat([
        take_features_from_maps(feature_maps, voxel_indices, stride, mode)
        for feature_maps, voxel_indices in zip(feature_maps_batch, voxel_indices_batch, strict=True)
    ])


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    """Taken from https://github.com/facebookresearch/vicreg/blob/4e12602fd495af83efd1631fbe82523e6db092e0/main_vicreg.py#L239.
    """
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
