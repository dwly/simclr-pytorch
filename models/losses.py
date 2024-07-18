import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import diffdist
import torch.distributed as dist


def gather(z):
    gather_z = [torch.zeros_like(z) for _ in range(torch.distributed.get_world_size())]
    gather_z = diffdist.functional.all_gather(gather_z, z)
    gather_z = torch.cat(gather_z)

    return gather_z


def accuracy(logits, labels, k):
    topk = torch.sort(logits.topk(k, dim=1)[1], 1)[0]
    labels = torch.sort(labels, 1)[0]
    acc = (topk == labels).all(1).float()
    return acc


def mean_cumulative_gain(logits, labels, k):
    topk = torch.sort(logits.topk(k, dim=1)[1], 1)[0]
    labels = torch.sort(labels, 1)[0]
    mcg = (topk == labels).float().mean(1)
    return mcg


def mean_average_precision(logits, labels, k):
    # TODO: not the fastest solution but looks fine
    argsort = torch.argsort(logits, dim=1, descending=True)
    labels_to_sorted_idx = torch.sort(torch.gather(torch.argsort(argsort, dim=1), 1, labels), dim=1)[0] + 1
    precision = (1 + torch.arange(k, device=logits.device).float()) / labels_to_sorted_idx
    return precision.sum(1) / k


class NTXent(nn.Module):
    """
    Contrastive loss with distributed data parallel support
    """
    LARGE_NUMBER = 1e9

    def __init__(self, tau=1., gpu=None, multiplier=2, distributed=False):
        super().__init__()
        self.tau = tau
        self.multiplier = multiplier
        self.distributed = distributed
        self.norm = 1.

    def forward(self, z, get_map=False):
        n = z.shape[0]
        assert n % self.multiplier == 0

        z = F.normalize(z, p=2, dim=1) / np.sqrt(self.tau)

        if self.distributed:
            z_list = [torch.zeros_like(z) for _ in range(dist.get_world_size())]
            # all_gather fills the list as [<proc0>, <proc1>, ...]
            # TODO: try to rewrite it with pytorch official tools
            z_list = diffdist.functional.all_gather(z_list, z)
            # split it into [<proc0_aug0>, <proc0_aug1>, ..., <proc0_aug(m-1)>, <proc1_aug(m-1)>, ...]
            z_list = [chunk for x in z_list for chunk in x.chunk(self.multiplier)]
            # sort it to [<proc0_aug0>, <proc1_aug0>, ...] that simply means [<batch_aug0>, <batch_aug1>, ...] as expected below
            z_sorted = []
            for m in range(self.multiplier):
                for i in range(dist.get_world_size()):
                    z_sorted.append(z_list[i * self.multiplier + m])
            z = torch.cat(z_sorted, dim=0)
            n = z.shape[0]

        logits = z @ z.t()
        logits[np.arange(n), np.arange(n)] = -self.LARGE_NUMBER
        logprob = F.log_softmax(logits, dim=1)

        # 选择每行（除自己）最大的100个相似度
        # topk_val, topk_idx = torch.topk(logits, k=200, dim=1, sorted=False)
        topk_val, topk_idx = torch.topk(logits, k=100, dim=1, largest=True, sorted=False)
        # logits[np.arange(n), np.arange(n)] = -self.LARGE_NUMBER
        # 去掉自己（最大的那个值，因为设置了-LARGE_NUMBER所以是第一个）
        # topk_val = topk_val[:, 1:]
        # topk_idx = topk_idx[:, 1:]
        # # 利用gather来重建相似度矩阵，只包含topk相似度
        # logits_topk = torch.zeros_like(logits)
        # logits_topk.scatter_(1, topk_idx, topk_val)

        # 初始化logits_topk为非常大的负数，确保softmax时未覆盖值的影响接近于零
        logits_topk = torch.full_like(logits, -self.LARGE_NUMBER)
        # 利用scatter_填入topk相似度值
        logits_topk.scatter_(1, topk_idx, topk_val)
        # 用上面的logits_topk计算softmax
        logprob_topk = F.log_softmax(logits_topk, dim=1)

        # choose all positive objects for an example, for i it would be (i + k * n/m), where k=0...(m-1)
        m = self.multiplier
        labels = (np.repeat(np.arange(n), m) + np.tile(np.arange(m) * n//m, n)) % n
        # remove labels pointet to itself, i.e. (i, i)
        labels = labels.reshape(n, m)[:, 1:].reshape(-1)

        # TODO: maybe different terms for each process should only be computed here...
        loss = -logprob_topk[np.repeat(np.arange(n), m-1), labels].sum() / n / (m-1) / self.norm

        # zero the probability of identical pairs
        pred = logprob.data.clone()
        pred[np.arange(n), np.arange(n)] = -self.LARGE_NUMBER
        acc = accuracy(pred, torch.LongTensor(labels.reshape(n, m-1)).to(logprob.device), m-1)

        if get_map:
            _map = mean_average_precision(pred, torch.LongTensor(labels.reshape(n, m-1)).to(logprob.device), m-1)
            return loss, acc, _map

        return loss, acc

class NTXentWithMargin(nn.Module):
    """
    Contrastive loss with distributed data parallel support
    """
    LARGE_NUMBER = 1e9

    def __init__(self, tau=1., gpu=None, multiplier=2, distributed=False):
        super().__init__()
        self.tau = tau
        self.multiplier = multiplier
        self.distributed = distributed
        self.norm = 1.
        # self.temperature = 1.

    def forward(self, z, get_map=False):
        n = z.shape[0]
        assert n % self.multiplier == 0

        z = F.normalize(z, p=2, dim=1) / np.sqrt(self.tau)

        if self.distributed:
            z_list = [torch.zeros_like(z) for _ in range(dist.get_world_size())]
            # all_gather fills the list as [<proc0>, <proc1>, ...]
            # TODO: try to rewrite it with pytorch official tools
            z_list = diffdist.functional.all_gather(z_list, z)
            # split it into [<proc0_aug0>, <proc0_aug1>, ..., <proc0_aug(m-1)>, <proc1_aug(m-1)>, ...]
            z_list = [chunk for x in z_list for chunk in x.chunk(self.multiplier)]
            # sort it to [<proc0_aug0>, <proc1_aug0>, ...] that simply means [<batch_aug0>, <batch_aug1>, ...] as expected below
            z_sorted = []
            for m in range(self.multiplier):
                for i in range(dist.get_world_size()):
                    z_sorted.append(z_list[i * self.multiplier + m])
            z = torch.cat(z_sorted, dim=0)
            n = z.shape[0]

        logits = z @ z.t()
        logits[np.arange(n), np.arange(n)] = -self.LARGE_NUMBER
        logprob = F.log_softmax(logits, dim=1)

        # choose all positive objects for an example, for i it would be (i + k * n/m), where k=0...(m-1)
        m = self.multiplier
        labels = (np.repeat(np.arange(n), m) + np.tile(np.arange(m) * n//m, n)) % n
        # remove labels pointet to itself, i.e. (i, i)
        labels = labels.reshape(n, m)[:, 1:].reshape(-1)
        margin = 0.5
        # calculate the new logits with margin
        logits_with_margin = logits - margin  # subtract margin value from all logits
        # assign a LARGE_NUMBER to original positive samples to avoid them contributing to the loss
        logits_with_margin[np.arange(n), np.arange(n)] = -self.LARGE_NUMBER
        # 使用带margin的logits重新计算log probability
        # logprob_with_margin = F.log_softmax(logits_with_margin / self.tau, dim=1)
        logprob_with_margin = F.log_softmax(logits_with_margin, dim=1)
        # 使用带margin的logprob计算损失
        loss = -logprob_with_margin[np.repeat(np.arange(n), m - 1), labels].sum() / n / (m - 1) / self.norm


        # zero the probability of identical pairs
        pred = logprob.data.clone()
        pred[np.arange(n), np.arange(n)] = -self.LARGE_NUMBER
        acc = accuracy(pred, torch.LongTensor(labels.reshape(n, m-1)).to(logprob.device), m-1)

        if get_map:
            _map = mean_average_precision(pred, torch.LongTensor(labels.reshape(n, m-1)).to(logprob.device), m-1)
            return loss, acc, _map

        return loss, acc
class NTXentWithSemiHard(nn.Module):
    """
    Contrastive loss with distributed data parallel support
    """
    LARGE_NUMBER = 1e9

    def __init__(self, tau=1., gpu=None, multiplier=2, distributed=False):
        super().__init__()
        self.tau = tau
        self.multiplier = multiplier
        self.distributed = distributed
        self.norm = 1.

    def forward(self, z, get_map=False):
        n = z.shape[0]
        assert n % self.multiplier == 0

        z = F.normalize(z, p=2, dim=1) / np.sqrt(self.tau)

        if self.distributed:
            z_list = [torch.zeros_like(z) for _ in range(dist.get_world_size())]
            # all_gather fills the list as [<proc0>, <proc1>, ...]
            # TODO: try to rewrite it with pytorch official tools
            z_list = diffdist.functional.all_gather(z_list, z)
            # split it into [<proc0_aug0>, <proc0_aug1>, ..., <proc0_aug(m-1)>, <proc1_aug(m-1)>, ...]
            z_list = [chunk for x in z_list for chunk in x.chunk(self.multiplier)]
            # sort it to [<proc0_aug0>, <proc1_aug0>, ...] that simply means [<batch_aug0>, <batch_aug1>, ...] as expected below
            z_sorted = []
            for m in range(self.multiplier):
                for i in range(dist.get_world_size()):
                    z_sorted.append(z_list[i * self.multiplier + m])
            z = torch.cat(z_sorted, dim=0)
            n = z.shape[0]

        logits = z @ z.t()
        logits[np.arange(n), np.arange(n)] = -self.LARGE_NUMBER
        logprob = F.log_softmax(logits, dim=1)

        # 选择每行最大的K个相似度
        topk_val, topk_idx = torch.topk(logits, k=n/4, dim=1, largest=True, sorted=False)
        # 选择每行最小的K个相似度
        bottk_val, bottk_idx = torch.topk(logits, k=n/4, dim=1, largest=False, sorted=False)

        logits_semik = logits.data.clone()
        # logits_semik.scatter_(1,topk_idx,0)
        # logits_semik.scatter_(1,bottk_idx,0)

        logits_semik.scatter_(1, topk_idx, -self.LARGE_NUMBER)
        logits_semik.scatter_(1, bottk_idx, -self.LARGE_NUMBER)

        # 用上面的logits_semik计算softmax
        logprob_semik = F.log_softmax(logits_semik, dim=1)

        # choose all positive objects for an example, for i it would be (i + k * n/m), where k=0...(m-1)
        m = self.multiplier
        labels = (np.repeat(np.arange(n), m) + np.tile(np.arange(m) * n//m, n)) % n
        # remove labels pointet to itself, i.e. (i, i)
        labels = labels.reshape(n, m)[:, 1:].reshape(-1)

        # TODO: maybe different terms for each process should only be computed here...
        loss = -logprob_semik[np.repeat(np.arange(n), m-1), labels].sum() / n / (m-1) / self.norm

        # zero the probability of identical pairs
        pred = logprob.data.clone()
        pred[np.arange(n), np.arange(n)] = -self.LARGE_NUMBER
        acc = accuracy(pred, torch.LongTensor(labels.reshape(n, m-1)).to(logprob.device), m-1)

        if get_map:
            _map = mean_average_precision(pred, torch.LongTensor(labels.reshape(n, m-1)).to(logprob.device), m-1)
            return loss, acc, _map

        return loss, acc

class Probability(nn.Module):
    """
    Contrastive loss with distributed data parallel support
    """
    LARGE_NUMBER = 1e9

    def __init__(self, tau=1., gpu=None, multiplier=2, distributed=False):
        super().__init__()
        self.tau = tau
        self.multiplier = multiplier
        self.distributed = distributed
        self.norm = 1.

    def forward(self, z, get_map=False):
        n = z.shape[0]
        assert n % self.multiplier == 0

        z = F.normalize(z, p=2, dim=1) / np.sqrt(self.tau)

        if self.distributed:
            z_list = [torch.zeros_like(z) for _ in range(dist.get_world_size())]
            # all_gather fills the list as [<proc0>, <proc1>, ...]
            # TODO: try to rewrite it with pytorch official tools
            z_list = diffdist.functional.all_gather(z_list, z)
            # split it into [<proc0_aug0>, <proc0_aug1>, ..., <proc0_aug(m-1)>, <proc1_aug(m-1)>, ...]
            z_list = [chunk for x in z_list for chunk in x.chunk(self.multiplier)]
            # sort it to [<proc0_aug0>, <proc1_aug0>, ...] that simply means [<batch_aug0>, <batch_aug1>, ...] as expected below
            z_sorted = []
            for m in range(self.multiplier):
                for i in range(dist.get_world_size()):
                    z_sorted.append(z_list[i * self.multiplier + m])
            z = torch.cat(z_sorted, dim=0)
            n = z.shape[0]

        logits = z @ z.t()
        logits[np.arange(n), np.arange(n)] = -self.LARGE_NUMBER
        logits /= 0.1
        logprob = F.softmax(logits, dim=1)
        return logprob
class Pearso(nn.Module):
    """
    Contrastive loss with distributed data parallel support
    """
    LARGE_NUMBER = 1e9

    def __init__(self, tau=1., gpu=None, multiplier=2, distributed=False):
        super().__init__()
        self.tau = tau
        self.multiplier = multiplier
        self.distributed = distributed
        self.norm = 1.

    def forward(self, z, get_map=False):
        n = z.shape[0]
        assert n % self.multiplier == 0

        # z = F.normalize(z, p=2, dim=1) / np.sqrt(self.tau)

        # 计算特征向量的均值

        mean_z = torch.mean(z, 1, keepdim=True)

        # 减去均值，进行中心化处理

        z_centered = z - mean_z

        # 计算每个特征向量的L2范数

        stddev_z = z_centered.norm(p=2, dim=1, keepdim=True) / np.sqrt(self.tau)

        # 归一化特征向量，这样皮尔逊相关系数只需要计算中心化后的向量乘积

        z = z_centered / (stddev_z + 1e-8)  # 避免除以零

        if self.distributed:
            z_list = [torch.zeros_like(z) for _ in range(dist.get_world_size())]
            # all_gather fills the list as [<proc0>, <proc1>, ...]
            # TODO: try to rewrite it with pytorch official tools
            z_list = diffdist.functional.all_gather(z_list, z)
            # split it into [<proc0_aug0>, <proc0_aug1>, ..., <proc0_aug(m-1)>, <proc1_aug(m-1)>, ...]
            z_list = [chunk for x in z_list for chunk in x.chunk(self.multiplier)]
            # sort it to [<proc0_aug0>, <proc1_aug0>, ...] that simply means [<batch_aug0>, <batch_aug1>, ...] as expected below
            z_sorted = []
            for m in range(self.multiplier):
                for i in range(dist.get_world_size()):
                    z_sorted.append(z_list[i * self.multiplier + m])
            z = torch.cat(z_sorted, dim=0)
            n = z.shape[0]

        logits = 1-(z @ z.t())
        logits[np.arange(n), np.arange(n)] = -self.LARGE_NUMBER

        logprob = F.log_softmax(logits, dim=1)

        # choose all positive objects for an example, for i it would be (i + k * n/m), where k=0...(m-1)
        m = self.multiplier
        labels = (np.repeat(np.arange(n), m) + np.tile(np.arange(m) * n//m, n)) % n
        # remove labels pointet to itself, i.e. (i, i)
        labels = labels.reshape(n, m)[:, 1:].reshape(-1)

        # TODO: maybe different terms for each process should only be computed here...
        loss = -logprob[np.repeat(np.arange(n), m-1), labels].sum() / n / (m-1) / self.norm

        # zero the probability of identical pairs
        pred = logprob.data.clone()
        pred[np.arange(n), np.arange(n)] = -self.LARGE_NUMBER
        acc = accuracy(pred, torch.LongTensor(labels.reshape(n, m-1)).to(logprob.device), m-1)

        if get_map:
            _map = mean_average_precision(pred, torch.LongTensor(labels.reshape(n, m-1)).to(logprob.device), m-1)
            return loss, acc, _map

        return loss, acc

class Prediction_loss(nn.Module):
    """
    Contrastive loss with distributed data parallel support
    """
    LARGE_NUMBER = 1e9

    def __init__(self, tau=1., gpu=None, multiplier=2, distributed=False):
        super().__init__()
        self.tau = tau
        self.multiplier = multiplier
        self.distributed = distributed
        self.norm = 1.

    def forward(self, preds, targets, get_map=False):
        n = preds.shape[0]
        assert n % self.multiplier == 0
        if self.distributed:
            preds = self.method_distribute(preds)
            targets = self.method_distribute(targets)
        # 均值中心化
        # preds_mean = preds.mean(dim=1, keepdim=True)
        # targets_mean = targets.mean(dim=1, keepdim=True)
        #
        # preds_centered = preds - preds_mean
        # targets_centered = targets - targets_mean
        # # 方差归一化
        # preds_std = preds_centered.std(dim=1, keepdim=True)
        # targets_std = targets_centered.std(dim=1, keepdim=True)
        # preds_normalized = preds_centered / (preds_std + 1e-6)  # 添加一个小的值以避免除以零
        # targets_normalized = targets_centered / (targets_std + 1e-6)
        criterion = nn.MSELoss()
        # return criterion(preds_normalized, targets_normalized.detach())
        return criterion(preds, targets.detach())

    def method_distribute(self, z):
        z_list = [torch.zeros_like(z) for _ in range(dist.get_world_size())]
        # all_gather fills the list as [<proc0>, <proc1>, ...]
        # TODO: try to rewrite it with pytorch official tools
        z_list = diffdist.functional.all_gather(z_list, z)
        # split it into [<proc0_aug0>, <proc0_aug1>, ..., <proc0_aug(m-1)>, <proc1_aug(m-1)>, ...]
        z_list = [chunk for x in z_list for chunk in x.chunk(self.multiplier)]
        # sort it to [<proc0_aug0>, <proc1_aug0>, ...] that simply means [<batch_aug0>, <batch_aug1>, ...] as expected below
        z_sorted = []
        for m in range(self.multiplier):
            for i in range(dist.get_world_size()):
                z_sorted.append(z_list[i * self.multiplier + m])
        return torch.cat(z_sorted, dim=0)