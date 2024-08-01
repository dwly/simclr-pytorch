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

        dim = 128
        pred_dim = 64
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                       nn.BatchNorm1d(pred_dim),
                                       nn.ReLU(inplace=True),  # hidden layer
                                       nn.Linear(pred_dim, dim))  # output layer

    def forward(self, x, y=None, k=None, t=None, out='z'):
        y_pre = None
        y_pro = None
        k_pre = None
        k_pro = None
        t_pre = None
        t_pro = None
        h = self.convnet(x)
        if out == 'h':
            return h
        if y is not None:
            y = self.convnet(y)
            y_pro = self.projection(y)
            y_pre = self.predictor(y_pro)
        if k is not None:
            k = self.convnet(k)
            k_pro = self.projection(k)
            k_pre = self.predictor(k_pro)
        if t is not None:
            t = self.convnet(t)
            t_pro = self.projection(t)
            t_pre = self.predictor(t_pro)
        # return self.projection(h)
        pro = self.projection(h)
        pre = self.predictor(pro)
        return pre, self.projection(h), y_pre, y_pro, k_pre, k_pro, t_pre, t_pro
