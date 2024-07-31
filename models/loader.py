# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from PIL import ImageFilter
import random

from torchvision.transforms import transforms
# import RandomMask
from utils import datautils


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform
        self.mask_transform = RandomMask(mask_size=(1, 1), mask_value=0, num_masks=5)
        self.clip = datautils.Clip()
    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        #增加掩码增强
        q_m = self.mask_transform(q)
        k_m = self.mask_transform(k)
        q = self.clip(q)
        k = self.clip(k)
        q_m = self.clip(q_m)
        k_m = self.clip(k_m)

        return [q, k, q_m, k_m]
        # return [q, k_m]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
class RandomMask:
    def __init__(self, mask_size=(16, 16), mask_value=0, num_masks=1):
        super().__init__()
        self.mask_size = mask_size
        self.mask_value = mask_value
        self.num_masks = num_masks
    def __call__(self, img):
        _, H, W = img.size()
        for _ in range(self.num_masks):
            top = random.randint(0, H - self.mask_size[0])
            left = random.randint(0, W - self.mask_size[1])
            img[:, top:top + self.mask_size[0], left:left + self.mask_size[1]] = self.mask_value
        return img

        # """Apply random square masks to the image."""
        # draw = ImageDraw.Draw(img)
        # width, height = img.size
        # for _ in range(num_patches):
        #     upper_left_x = np.random.randint(0, width - patch_size)
        #     upper_left_y = np.random.randint(0, height - patch_size)
        #     draw.rectangle(
        #         (upper_left_x, upper_left_y, upper_left_x + patch_size, upper_left_y + patch_size),
        #         fill='black'
        #     )
        #
        # return img