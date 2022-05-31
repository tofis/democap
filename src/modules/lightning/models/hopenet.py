# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from .resnet import (
    resnet50, 
    resnet10
)

from .graphunet import (
    GraphNet,
    GraphUNet
)

import typing

class HopeNet(nn.Module):
    def __init__(self):
        super(HopeNet, self).__init__()
        self.resnet = resnet50(pretrained=False, num_classes=19*2)
        self.graphnet = GraphNet(in_features=2050, out_features=2)
        self.graphunet = GraphUNet(in_features=2, out_features=3)

    def forward(self, 
        data : torch.Tensor
    ) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        points2D_init, features = self.resnet(data)
        features = features.unsqueeze(1).repeat(1, 19, 1)
        # batch = points2D.shape[0]
        in_features = torch.cat([points2D_init, features], dim=2)
        points2D = self.graphnet(in_features)
        points3D = self.graphunet(points2D)
        return points2D_init, points2D, points3D
