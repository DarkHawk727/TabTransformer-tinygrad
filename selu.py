import sys
import os

sys.path.append(os.path.abspath("."))

import math
from typing import Callable, List

import numpy as np

from tinygrad import Tensor, nn

from tinygrad.engine.lazy import LazyBuffer
from tinygrad.tensor import Function


# https://github.com/pytorch/pytorch/blob/96aaa311c0251d24decb9dc5da4957b7c590af6f/torch/nn/modules/activation.py#L507
class Selu(Function):
    _alpha: float = 1.6732632423543772848170429916717
    _lambda: float = 1.0507009873554804934193349852946

    def forward(self, x: LazyBuffer) -> LazyBuffer:
        self.ret = self._lambda * LazyBuffer.where(
            x >= 0, x, self._alpha * ((x * (1 / math.log(2))).exp2() - 1)
        )
        return self.ret

    def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
        dx = LazyBuffer.where(
            self.ret >= 0,
            self._lambda,
            self._lambda * self._alpha * (self.ret * (1 / math.log(2))).exp2(),
        )
        return dx * grad_output
