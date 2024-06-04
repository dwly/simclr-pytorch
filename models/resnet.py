#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch.nn as nn
import torchvision.models as models
import torch
import torch.nn.functional as F


class Flatten(nn.Module):
    def __init__(self, dim=-1):
        super(Flatten, self).__init__()
        self.dim = dim

    def forward(self, feat):
        return torch.flatten(feat, start_dim=self.dim)


class ResNetEncoder(models.resnet.ResNet):
    """Wrapper for TorchVison ResNet Model
    This was needed to remove the final FC Layer from the ResNet Model"""
    def __init__(self, block, layers, cifar_head=False, hparams=None):
        super().__init__(block, layers)
        self.cifar_head = cifar_head
        if cifar_head:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = self._norm_layer(64)
            self.relu = nn.ReLU(inplace=True)
        self.hparams = hparams
        # 添加额外的卷积层用于构建特征金字塔
        self.extra_conv = nn.Conv2d(2048, 256, kernel_size=1, stride=1, bias=False)

        self.pyramid_conv1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False)
        self.pyramid_conv2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, bias=False)
        self.pyramid_conv3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, bias=False)

        print('** Using avgpool **')

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.cifar_head:
            x = self.maxpool(x)

        x = self.layer1(x)
        c1 = x
        x = self.layer2(x)
        c2 = x
        x = self.layer3(x)
        c3 = x
        x = self.layer4(x)

        # 获取不同层次的特征
        c4 = x
        # c3 = self.layer3(x)
        # c2 = self.layer2(c3)
        # c1 = self.layer1(c2)

        # 构建特征金字塔
        # p3 = self.pyramid_conv1(c3)
        p4 = self.extra_conv(c4)
        # p2 = self.pyramid_conv2(c2)
        p3 = self.pyramid_conv1(c3)
        # p2 = self.pyramid_conv3(c2)
        p2 = self.pyramid_conv2(c2)
        # p1 = self.pyramid_conv3(c1)
        #
        # 从高层到低层进行上采样和融合
        # p3 = p3 + F.interpolate(p4, scale_factor=2, mode='nearest')
        p4 = F.interpolate(p4, scale_factor=2, mode='nearest')
        p3 = p3 + p4
        p2 = p2 + F.interpolate(p3, scale_factor=2, mode='nearest')

        # 降采样
        p2 = F.interpolate(p2, scale_factor=0.5, mode='nearest')

        # 使用额外的卷积层
        # p1 = self.extra_conv(c1)

        p1 = F.interpolate(c1, scale_factor=0.25, mode='nearest')

        # 融合所有尺度的特征

        fused_feature = p1 + p2 + p3
        x = self.avgpool(fused_feature)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        # x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x

class ResNet18(ResNetEncoder):
    def __init__(self, cifar_head=True):
        super().__init__(models.resnet.BasicBlock, [2, 2, 2, 2], cifar_head=cifar_head)


class ResNet50(ResNetEncoder):
    def __init__(self, cifar_head=True, hparams=None):
        super().__init__(models.resnet.Bottleneck, [3, 4, 6, 3], cifar_head=cifar_head, hparams=hparams)
