import numpy as np
import torch
import torch.nn as nn
from pysemseg.losses import JaccardLoss, FocalLoss
from pysemseg.utils import tensor_to_numpy
import cv2


def _dilate_tensor(tensor, kernel_size=(3, 3), iterations=3):
    np_images = tensor_to_numpy(tensor)
    np_images = np.split(np_images, np_images.shape[0])
    kernel = np.ones(kernel_size, dtype=np.uint8)
    dilated_images = [
        cv2.dilate(img[0].astype(np.uint8), kernel=kernel, iterations=iterations)
        for img in np_images
    ]
    dilated_images = np.stack(dilated_images)
    dilated_tensor = torch.LongTensor(dilated_images).to(tensor.device)
    return dilated_tensor


class BorderCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=-1):
        super().__init__()
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(
                ignore_index=ignore_index,reduction='none')

    def forward(self, inputs, targets):
        dilated_targets = _dilate_tensor(
            targets, kernel_size=(3, 3), iterations=3)
        border_pixels = dilated_targets - targets
        loss = border_pixels.float() * self.ce(inputs, targets)
        num_targets = (targets != self.ignore_index).sum().float()
        return loss.sum()


class HybridLoss(nn.Module):
    def __init__(
            self, weights=None, ignore_index=-1, ce_coef=0.5,
            jaccard_coef=0.25, focal_coef=0.25):
        super().__init__()
        self.ce_coef = ce_coef
        self.jaccard_coef = jaccard_coef
        self.focal_coef = focal_coef
        self.weights = weights
        if weights is not None and isinstance(weights, list):
            self.weights = torch.FloatTensor(self.weights)
        self.ce = nn.CrossEntropyLoss(
            weight=self.weights, ignore_index=ignore_index)
        self.log_jaccard = JaccardLoss(ignore_index=ignore_index)
        self.focal_loss = FocalLoss(
            ignore_index=ignore_index, weights=self.weights)

    def forward(self, inputs, targets):
        return  (
            self.ce_coef * self.ce(inputs, targets) +
            self.jaccard_coef * self.log_jaccard(inputs, targets) +
            self.focal_coef * self.focal_loss(inputs, targets)
        )


class Hybrid3ChanLoss(nn.Module):
    def __init__(
            self, weights=None, ignore_index=-1, ce_coef=0.3,
            fore_jac_coef=0.2, border_jac_coef=0.2, focal_coef=0.3):
        super().__init__()
        self.ce_coef = ce_coef
        self.fore_jac_coef = fore_jac_coef
        self.border_jac_coef = border_jac_coef
        self.focal_coef = focal_coef

        self.weights = weights
        if weights is not None and isinstance(weights, list):
            self.weights = torch.FloatTensor(self.weights)

        self.ce = nn.CrossEntropyLoss(
            weight=self.weights, ignore_index=ignore_index)
        self.fore_jac = JaccardLoss(ignore_index=ignore_index, pos_index=1)
        self.border_jac = JaccardLoss(ignore_index=ignore_index, pos_index=2)
        self.focal = FocalLoss(
            ignore_index=ignore_index, weights=self.weights)

    def forward(self, inputs, targets):
        return  (
            self.ce_coef * self.ce(inputs, targets) +
            self.fore_jac_coef * self.fore_jac(inputs, targets) +
            self.border_jac_coef * self.border_jac(inputs, targets) +
            self.focal_coef * self.focal(inputs, targets)
        )
