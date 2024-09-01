import copy

import torch
from torch import nn
import models
from collections import OrderedDict
from argparse import Namespace
import yaml
import os


class BatchNorm1dNoBias(nn.BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bias.requires_grad = False


class EncodeProject(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        if hparams.arch == 'ResNet50':
            cifar_head = (hparams.data == 'cifar')
            self.convnet = models.resnet.ResNet50(cifar_head=cifar_head, hparams=hparams)
            self.encoder_dim = 2048
            # self.encoder_dim = 1024
        elif hparams.arch == 'resnet18':
            self.convnet = models.resnet.ResNet18(cifar_head=(hparams.data == 'cifar'))
            self.encoder_dim = 512
        else:
            raise NotImplementedError

        num_params = sum(p.numel() for p in self.convnet.parameters() if p.requires_grad)

        print(f'======> Encoder: output dim {self.encoder_dim} | {num_params/1e6:.3f}M parameters')

        self.proj_dim = 128
        projection_layers = [
            ('fc1', nn.Linear(self.encoder_dim, self.encoder_dim, bias=False)),
            ('bn1', nn.BatchNorm1d(self.encoder_dim)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(self.encoder_dim, 128, bias=False)),
            ('bn2', BatchNorm1dNoBias(128)),
        ]

        self.projection = nn.Sequential(OrderedDict(projection_layers))

        # dim = 128
        # pred_dim = 64
        # self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
        #                                nn.BatchNorm1d(pred_dim),
        #                                nn.ReLU(inplace=True),  # hidden layer
        #                                nn.Linear(pred_dim, dim))  # output layer

        # 初始化动量编码器
        self.m = 0.999
        self.encoder_m = copy.deepcopy(self.convnet)
        for param in self.encoder_m.parameters():
            param.requires_grad = False
        # self.momentum_update_key_encoder = self._momentum_update_key_encoder
        # self.projection_m = copy.deepcopy(self.projection)
        # for param in self.projection_m.parameters():
        #     param.requires_grad = False

        # self.encoder_b = self.convnet
        # self.encoder_m = self.convnet
        # for param_b, param_m in zip(
        #     self.convnet.parameters(), self.encoder_m.parameters()
        # ):
        #     param_m.data.copy_(param_b.data)  # initialize
        #     param_m.requires_grad = False  # not update by gradient
        # self.projection_m = self.projection
        # for param_b, param_m in zip(
        #     self.projection.parameters(), self.projection_m.parameters()
        # ):
        #     param_m.data.copy_(param_b.data)  # initialize
        #     param_m.requires_grad = False  # not update by gradient
    
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_b, param_m in zip(
                self.convnet.parameters(), self.encoder_m.parameters()
        ):
            param_m.data = param_m.data * self.m + param_b.data * (1.0 - self.m)
        # for param_b, param_m in zip(
        #         self.projection.parameters(), self.projection_m.parameters()
        # ):
        #     param_m.data = param_m.data * self.m + param_b.data * (1.0 - self.m)

    def forward(self, x, y=None, k=None, t=None, out='z'):
        # y_pre = None
        y_pro = None
        # k_pre = None
        k_pro = None
        # t_pre = None
        t_pro = None
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
        h = self.convnet(x)
        # h = self.encoder_b(x)
        if out == 'h':
            return h
        if y is not None:
            # y = self.convnet(y)
            y = self.encoder_m(y)
            y_pro = self.projection(y)
            # y_pre = self.predictor(y_pro)
        if k is not None:
            k = self.convnet(k)
            # k = self.encoder_b(k)
            k_pro = self.projection(k)
            # k_pre = self.predictor(k_pro)
        if t is not None:
            # t = self.convnet(t)
            t = self.encoder_m(t)
            t_pro = self.projection(t)
            # t_pre = self.predictor(t_pro)
        # return self.projection(h)
        # pro = self.projection(h)
        # pre = self.predictor(pro)
        # return pre, self.projection(h), y_pre, y_pro, k_pre, k_pro, t_pre, t_pro
        return self.projection(h), y_pro,  k_pro,  t_pro
