import torch
import torch.nn as nn
from pysemseg.losses import JaccardLoss, FocalLoss



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
        return (
            self.ce_coef * self.ce(inputs, targets) +
            self.jaccard_coef * self.log_jaccard(inputs, targets) +
            self.focal_coef * self.focal_loss(inputs, targets)
        )
