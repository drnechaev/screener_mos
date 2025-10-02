from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from medimm.models.unet_3d import unet3d, crop_and_pad_to


class ConditionModel(nn.Module):
    def __init__(
            self,
            in_channels: int = 1,
            condition_dim: int = 32,
            pretrained_model_path: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.condition_dim = condition_dim
        self.unet = unet3d(size='base', in_channels=in_channels + 1)
        self.convs = nn.ModuleList([
            nn.Conv3d(c, condition_dim, kernel_size=1)
            for c in self.unet.config.hidden_channels
        ])
        d = sum(self.unet.config.hidden_channels)
        for conv in self.convs:
            nn.init.uniform_(conv.weight, -(1. / d) ** 0.5, (1. / d) ** 0.5)
        nn.init.uniform_(self.convs[0].bias, -(1. / d) ** 0.5, (1. / d) ** 0.5)

        if pretrained_model_path is not None:
            self.load_state_dict(torch.load(pretrained_model_path, map_location='cpu'))

    def forward(self, image: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is None:
            N, _, H, W, S = image.shape
            mask = torch.ones((N, 1, H, W, S), dtype=image.dtype, device=image.device)

        feature_pyramid = self.unet(torch.cat([image * mask, mask], dim=1)).feature_pyramid

        feature_pyramid = [conv(x) for conv, x in zip(self.convs, feature_pyramid)]

        x = feature_pyramid.pop()
        while feature_pyramid:
            x = F.interpolate(x, scale_factor=2, mode='trilinear')
            y = feature_pyramid.pop()
            x = crop_and_pad_to(x, y)
            x = x + y
        
        return x
