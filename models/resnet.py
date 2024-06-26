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
        # self.pyramid_conv5 = nn.Conv2d(2048, 256, kernel_size=1, stride=1, bias=False)
        #
        # self.pyramid_conv4 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False)
        # self.pyramid_conv3 = nn.Conv2d(512, 256, kernel_size=1, stride=1, bias=False)
        # self.pyramid_conv2 = nn.Conv2d(256, 256, kernel_size=1, stride=1, bias=False)
        # 1.横向连接，保证通道数相同
        self.toplayer = nn.Conv2d(2048, 256, 1, 1, 0)
        self.latlayer1 = nn.Conv2d(1024, 256, 1, 1, 0)
        self.latlayer2 = nn.Conv2d(512, 256, 1, 1, 0)
        self.latlayer3 = nn.Conv2d(256, 256, 1, 1, 0)
        #采用转置卷积进行上采样
        self.upsample = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1,output_padding=1)
        #横向链接
        self.latconnect = nn.Conv2d(256, 256, 1, 1, 0)
        #最大池化下采样
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.bnlayer = self._norm_layer(2048)
        self.smooth5 = nn.Conv2d(256, 2048, 3, 1, 1)

        self.bnlayer1 = self._norm_layer(256)
        # self.bnlayer2 = self._norm_layer(512)
        # self.bnlayer3 = self._norm_layer(1024)

        # self.sfpn = SFPN(2048, 512)  # 使用SFPN进行特征融合

        self.calllatlayer1 = nn.Conv2d(256, 1024, 1, 1, 0)
        self.calllatlayer2 = nn.Conv2d(256, 512, 1, 1, 0)
        # self.calllatlayer3 = nn.Conv2d(256, 256, 1, 1, 0)

        self.convC5 = nn.Conv2d(256, 512, kernel_size=1, stride=1, bias=False)
        # self.conv332 = nn.Conv2d(256, 256, kernel_size=3, stride=2, bias=False)
        # self.conv22 = nn.Conv2d(1024, 2048,kernel_size=2, stride=1, bias=False)
        # self.conv11 = nn.Conv2d(1024,2048,kernel_size=1, stride=1, bias=False)
        # self.conv33 = nn.Conv2d(2048, 2048,kernel_size=3, stride=1, bias=False)
        #3X3卷积融合特征
        self.smooth1 = nn.Conv2d(256, 256, 3, 1, 1)
        self.smooth2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.smooth3 = nn.Conv2d(256, 256, 3, 1, 1)
        self.smooth4 = nn.Conv2d(256, 256, 3, 1, 1)
        #PAnet融合
        self.paconv = nn.Conv2d(256, 256, kernel_size=3, stride=2, bias=False)
        self.arrage3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, bias=False)
        self.arrage4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, bias=False)
        self.arrage5 = nn.Conv2d(256, 2048, kernel_size=3, stride=1, bias=False)

        print('** Using avgpool **')

    # def _upsample(self,x):
    #     x_up = self.upsample(x)
    #     self._norm_layer()

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
        m5 = self.toplayer(c5)
        m4 = self.latlayer1(c4)
        m3 = self.latlayer2(c3)
        m2 = self.latlayer3(c2)

        # 直接用c5上采样，加上p4
        # p4 = p4 + F.interpolate(c5, scale_factor=2, mode='nearest')

        # l5 = F.interpolate(p4, scale_factor=0.5, mode='nearest')

        # 从高层到低层进行上采样和融合
        # m4 = m4 + F.interpolate(m5, scale_factor=2, mode='nearest')
        # m3 = m3 + F.interpolate(m4, scale_factor=2, mode='nearest')
        # m2 = m2 + F.interpolate(m3, scale_factor=2, mode='nearest')

        #上采样采用转置卷积
        # l4 = self.upsample(m5)
        m4 = m4 + self.relu(self.bnlayer1(self.upsample(m5)))
        m3 = m3 + self.relu(self.bnlayer1(self.upsample(m4)))
        m2 = m2 + self.relu(self.bnlayer1(self.upsample(m3)))


        #look more times from https://cloud.tencent.com/developer/article/1639306 [DetectoRS]
        # p2 = c2 + m2
        # p3 = self.layer2(p2)
        #
        # p4 = self.layer3(p3+self.calllatlayer2(m3))
        # p5 = self.layer4(p4+self.calllatlayer1(m4))
        # l2 = c2 + m2
        # l3 = self.layer2(l2)
        # l4 = self.layer3(l3)
        # l5 = self.layer4(l4)
        # l5 = self.bnlayer(l5)

        #from http://html.rhhz.net/GDGYDXXB/html/1621904024985-574459568.htm
        # m6 = self.maxpool1(m5)

        l2 = self.latconnect(m2)
        l3 = self.latconnect(m3)
        l4 = self.latconnect(m4)
        l5 = self.latconnect(m5)
        # # l6 = self.latconnect(m6)

        # l2 = self.smooth1(m2)
        # l3 = self.smooth2(m3)
        # l4 = self.smooth3(m4)
        # l5 = self.smooth4(m5)
        #

        l3 = l3 + self.maxpool1(l2+c2)
        # l3 = self.smooth1(l3)
        l4 = l4 + self.maxpool1(l3+m3)
        # l4 = self.smooth2(l4)
        l5 = l5 + self.maxpool1(l4+m4)
        # l5 = l5 + self.maxpool1(l4) + self.maxpool1(m4)

        # l3 = l3 + F.interpolate(l2, scale_factor=0.5, mode='nearest')
        # l4 = l4 + F.interpolate(l3, scale_factor=0.5, mode='nearest')
        # l5 = l5 + F.interpolate(l4, scale_factor=0.5, mode='nearest')

        # l6 = l6 + self.maxpool1(l5)
        p5 = self.smooth5(l5)
        # p5 = c5 + p5

        # p5 = self.smooth4(m5)
        # p4 = self.smooth1(m4)
        # p3 = self.smooth2(m3)
        # p2 = self.smooth3(m2)
        # l2 = c2 + p2
        # l3 = self.layer2(l2)
        # l4 = self.layer3(l3)
        # l5 = self.layer4(l4)
        p5 = self.bnlayer(p5)

        # p3 = self.arrage3(p3 + self.paconv(p2))
        # p4 = self.arrage4(p4 + self.paconv(p3))
        # p5 = self.arrage5(p5 + self.paconv(p4))

        #对融合后的p4、p5进行concat



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
        p5 = self.relu(p5)
        # out = self.avgpool(l5)
        out = self.avgpool(p5)
        out = torch.flatten(out, 1)


        return out

class ResNet18(ResNetEncoder):
    def __init__(self, cifar_head=True):
        super().__init__(models.resnet.BasicBlock, [2, 2, 2, 2], cifar_head=cifar_head)


class ResNet50(ResNetEncoder):
    def __init__(self, cifar_head=True, hparams=None):
        super().__init__(models.resnet.Bottleneck, [3, 4, 6, 3], cifar_head=cifar_head, hparams=hparams)
