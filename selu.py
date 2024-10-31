import math

from tinygrad.engine.lazy import LazyBuffer
from tinygrad.tensor import Function


# https://github.com/pytorch/pytorch/blob/96aaa311c0251d24decb9dc5da4957b7c590af6f/torch/nn/modules/activation.py#L507
class Selu(Function):
    _alpha: float = 1.6732632423543772848170429916717
    _lambda: float = 1.0507009873554804934193349852946

    def forward(self, x: LazyBuffer) -> LazyBuffer:
        alpha_buf = x.const_like(self._alpha)
        lambda_buf = x.const_like(self._lambda)
        self.ret = lambda_buf * LazyBuffer.where(
            x >= 0, x, alpha_buf * ((x * (1 / math.log(2))).exp2() - 1)
        )
        return self.ret

    def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
        alpha_buf = self.ret.const_like(self._alpha)
        lambda_buf = self.ret.const_like(self._lambda)
        dx = LazyBuffer.where(
            self.ret >= 0,
            lambda_buf,
            lambda_buf * alpha_buf * (self.ret * (1 / math.log(2))).exp2(),
        )
        return dx * grad_output
