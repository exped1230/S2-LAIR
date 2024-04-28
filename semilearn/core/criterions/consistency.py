
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import torch 
import torch.nn as nn 
from torch.nn import functional as F
import numpy as np

from .cross_entropy import ce_loss



def consistency_loss(logits, targets, name='ce', mask=None, similarity_matrix=None):
    """
    consistency regularization loss in semi-supervised learning.

    Args:
        logits: logit to calculate the loss on and back-propagion, usually being the strong-augmented unlabeled samples
        targets: pseudo-labels (either hard label or soft label)
        name: use cross-entropy ('ce') or mean-squared-error ('mse') to calculate loss
        mask: masks to mask-out samples when calculating the loss, usually being used as confidence-masking-out
    """

    assert name in ['ce', 'mse', 'soft_cos']
    # logits_w = logits_w.detach()
    if name == 'mse':
        probs = torch.softmax(logits, dim=-1)
        loss = F.mse_loss(probs, targets, reduction='none').mean(dim=1)
    if name == 'soft_cos':
        assert similarity_matrix != None
        assert logits.shape == targets.shape
        probs = torch.softmax(logits, dim=-1)
        norm_a = torch.sqrt(torch.sum(similarity_matrix*torch.matmul(probs.unsqueeze(-1),probs.unsqueeze(-1).permute(0, 2, 1)),dim=(1, 2), keepdim=True))
        norm_b = torch.sqrt(torch.sum(similarity_matrix*torch.matmul(targets.unsqueeze(-1),targets.unsqueeze(-1).permute(0, 2, 1)),dim=(1, 2), keepdim=True))
        # 计算相似度矩阵的分子部分
        numerator = torch.sum(similarity_matrix*torch.matmul(probs.unsqueeze(-1),targets.unsqueeze(-1).permute(0, 2, 1)),dim=(1, 2), keepdim=True)
        similarity = (numerator / (norm_a * norm_b))
        loss = 1 - similarity.squeeze()
    else:
        loss = ce_loss(logits, targets, reduction='none')

    if mask is not None:
        # mask must not be boolean type
        loss = loss * mask

    return loss.mean()



class ConsistencyLoss(nn.Module):
    """
    Wrapper for consistency loss
    """
    def forward(self, logits, targets, name='ce', mask=None, similarity_matrix=None):
        return consistency_loss(logits, targets, name, mask, similarity_matrix)