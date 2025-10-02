from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from medimm.models.unet_3d import unet3d, crop_and_pad_to as crop_and_pad_to_3d
from medimm.models.unet_2d import unet2d, crop_and_pad_to as crop_and_pad_to_2d


class DescriptorModel(nn.Module):
    def __init__(
            self,
            in_channels: int = 1,
            descriptor_dim: int = 32,
            mode: str = "3d",
            pretrained_model_path: Optional[str] = None
    ) -> None:
        super().__init__()

        self.descriptor_dim = descriptor_dim
        self.mode = mode

        if self.mode == "3d": 
            self.unet = unet3d(size='base', in_channels=in_channels)
            self.convs = nn.ModuleList([
                nn.Conv3d(c, descriptor_dim, kernel_size=1)
                for c in self.unet.config.hidden_channels
            ])
            self.crop_and_pad_to = crop_and_pad_to_3d
        elif self.mode == "2d":
            self.unet = unet2d(size='base', in_channels=in_channels)
            self.convs = nn.ModuleList([
                nn.Conv2d(c, descriptor_dim, kernel_size=1)
                for c in self.unet.config.hidden_channels
            ])
            self.crop_and_pad_to = crop_and_pad_to_2d
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        d = sum(self.unet.config.hidden_channels)
        for conv in self.convs:
            nn.init.uniform_(conv.weight, -(1. / d) ** 0.5, (1. / d) ** 0.5)
        nn.init.uniform_(self.convs[0].bias, -(1. / d) ** 0.5, (1. / d) ** 0.5)

        if pretrained_model_path is not None:
            self.load_state_dict(torch.load(pretrained_model_path, map_location='cpu'))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        feature_pyramid = self.unet(image).feature_pyramid

        feature_pyramid = [conv(x) for conv, x in zip(self.convs, feature_pyramid)]

        x = feature_pyramid.pop()
        while feature_pyramid:
            if self.mode == "3d":
                x = F.interpolate(x, scale_factor=2, mode='trilinear')
            elif self.mode == "2d":
                x = F.interpolate(x, scale_factor=2, mode='bilinear')
            y = feature_pyramid.pop()
            x = self.crop_and_pad_to(x, y)
            x = x + y

        return x
