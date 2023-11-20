#  Copyright (c) 2020 Preferred Networks, Inc.
#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from torch import nn


class EmpiricalNormalization(nn.Module):
    """Normalize mean and variance of values based on empirical values."""

    def __init__(self, shape, eps=1e-2, decay=0.995):
        """Initialize EmpiricalNormalization module.

        Args:
            shape (int or tuple of int): Shape of input values except batch axis.
            eps (float): Small value for stability.
            until (int or None): If this arg is specified, the link learns input values until the sum of batch sizes
            exceeds it.
        """
        super().__init__()
        self.eps = eps
        self.decay = decay
        self.register_buffer("_mean", torch.zeros(shape).unsqueeze(0))
        self.register_buffer("_var", torch.ones(shape).unsqueeze(0))
        self.register_buffer("_std", torch.ones(shape).unsqueeze(0))

    @property
    def mean(self):
        return self._mean.squeeze(0).clone()

    @property
    def std(self):
        return self._std.squeeze(0).clone()

    def forward(self, x):
        """Normalize mean and variance of values based on empirical values.

        Args:
            x (ndarray or Variable): Input values

        Returns:
            ndarray or Variable: Normalized output values
        """

        if self.training:
            self.update(x)
        return (x - self._mean) / (self._std + self.eps)

    @torch.jit.unused
    def update(self, x):
        """Learn input values without computing the output values of them"""
        
        mean_x = torch.mean(x, dim=0, keepdim=True)
        self._mean = self._mean * self.decay + (1.0 - self.decay) * mean_x
        var_x = torch.mean(x**2, dim=0, keepdim=True)
        self._var = self._var * self.decay + (1.0 - self.decay) * var_x 
        self._std = torch.sqrt(self._var - self._mean**2 + self.eps)

    @torch.jit.unused
    def inverse(self, y):
        return y * (self._std + self.eps) + self._mean
