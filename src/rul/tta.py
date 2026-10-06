"""Test-time adaptation without target labels."""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn


@torch.no_grad()
def adabn(model: nn.Module, x_target: np.ndarray, batch_size: int = 512) -> nn.Module:
    """AdaBN (Li et al., 2016): re-estimate every BatchNorm's statistics on unlabeled target data.

    Returns an adapted copy; the source model is left untouched. Only BN running
    statistics change; no weights are updated and no target labels are used.
    """
    adapted = copy.deepcopy(model)
    bns = [m for m in adapted.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    for bn in bns:
        bn.reset_running_stats()
        bn.momentum = None  # cumulative moving average over all target batches
    adapted.eval()
    for bn in bns:
        bn.train()
    for i in range(0, len(x_target), batch_size):
        batch = torch.from_numpy(x_target[i:i + batch_size])
        if len(batch) > 1:
            adapted(batch)
    adapted.eval()
    return adapted
