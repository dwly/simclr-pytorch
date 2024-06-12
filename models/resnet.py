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
        # 添加额外的卷积层用于构建特征金字塔
        self.pyramid_conv5 = nn.Conv2d(2048, 2048, kernel_size=1, stride=1, bias=False)

        self.pyramid_conv4 = nn.Conv2d(1024, 2048, kernel_size=1, stride=1, bias=False)
        # self.pyramid_conv3 = nn.Conv2d(512, 256, kernel_size=1, stride=1, bias=False)
        # self.pyramid_conv2 = nn.Conv2d(256, 256, kernel_size=1, stride=1, bias=False)
        self.convC5 = nn.Conv2d(1024, 2048, kernel_size=1, stride=1, bias=False)
        # self.conv332 = nn.Conv2d(1024, 2048, kernel_size=3, stride=2, bias=False)
        # self.conv22 = nn.Conv2d(1024, 2048,kernel_size=2, stride=1, bias=False)
        # self.conv11 = nn.Conv2d(1024,2048,kernel_size=1, stride=1, bias=False)
        # self.conv33 = nn.Conv2d(2048, 2048,kernel_size=3, stride=1, bias=False)
        print('** Using avgpool **')

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.cifar_head:
            x = self.maxpool(x)

        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        # 1x1卷积
        p5 = self.pyramid_conv5(c5)
        p4 = self.pyramid_conv4(c4)
        # p3 = self.pyramid_conv3(c3)
        # p2 = self.pyramid_conv2(c2)

        # 直接用c5上采样，加上p4
        p4 = p4 + F.interpolate(c5, scale_factor=2, mode='nearest')
        l5 = F.interpolate(p4, scale_factor=0.5, mode='nearest')

        # # 从高层到低层进行上采样和融合
        # p4 = p4 + F.interpolate(p5, scale_factor=2, mode='nearest')

        # #对融合后p4下采样，再和p5进行concat
        # l4 = F.interpolate(p4, scale_factor=0.5, mode='nearest')
        # l5 = torch.cat((l4, p5), dim=1)

        # #对融合后p4下采样,再和p5进行add,并转化channel为2048
        # l5 = self.convC5(l4 + p5)

        # # 融合后的p4进行下采样
        # l4 = self.layer4(p4)
        # l5 = l4 + c5

        # 对融合后的特征，进行恢复到c5的channel、size、分辨率
        # # channel恢复到2048
        # l5 = self.convC5(p4)

        # #p4先经过3x3的卷积，stride=2,下采样
        # l4 = self.conv332(p4)
        # #相加后，再经过3x3的卷积
        # l5 = self.conv22(p5)
        #
        # out = self.conv33(l4+l5)

        #是否加非线性
        # out = self.relu(l5)
        out = self.avgpool(l5)
        out = torch.flatten(out, 1)

        return out

class ResNet18(ResNetEncoder):
    def __init__(self, cifar_head=True):
        super().__init__(models.resnet.BasicBlock, [2, 2, 2, 2], cifar_head=cifar_head)


class ResNet50(ResNetEncoder):
    def __init__(self, cifar_head=True, hparams=None):
        super().__init__(models.resnet.Bottleneck, [3, 4, 6, 3], cifar_head=cifar_head, hparams=hparams)
