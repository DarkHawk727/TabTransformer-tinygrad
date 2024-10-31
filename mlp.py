from typing import Callable, List

from tinygrad import Tensor, nn

from selu import Selu


class MLP:
    def __init__(self, l: int, m1: int = 4, m2: int = 2) -> None:
        self.layers: List[Callable[[Tensor], Tensor]] = [
            nn.Linear(l, m1 * l),
            nn.BatchNorm(m1 * l),
            Selu.apply,
            nn.Linear(m1 * l, m2 * l),
            nn.BatchNorm(m2 * l),
            Selu.apply,
            nn.Linear(m2 * l, 1),
            Tensor.sigmoid,
        ]

    def __call__(self, x: Tensor) -> Tensor:
        return x.sequential(self.layers)
