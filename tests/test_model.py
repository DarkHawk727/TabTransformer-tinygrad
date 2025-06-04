import numpy as np
from tinygrad import Tensor
import sys
from pathlib import Path
import types

sys.path.append(str(Path(__file__).resolve().parents[1]))

# Stub for tinygrad.engine.lazy.LazyBuffer if missing
if 'tinygrad.engine.lazy' not in sys.modules:
    dummy_lazy = types.ModuleType('tinygrad.engine.lazy')
    class LazyBuffer: pass
    dummy_lazy.LazyBuffer = LazyBuffer
    sys.modules['tinygrad.engine.lazy'] = dummy_lazy

# Stub selu module to avoid dependency on tinygrad internals
if 'selu' not in sys.modules:
    selu_stub = types.ModuleType('selu')
    class Selu:
        @staticmethod
        def apply(x):
            return x
    selu_stub.Selu = Selu
    sys.modules['selu'] = selu_stub

from model import split_features


def test_split_features_order():
    data = Tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    cat_indices = [0, 2]
    x_cat, x_cont = split_features(data, cat_indices)

    expected_cat = np.array([[1, 3], [5, 7]])
    expected_cont = np.array([[2, 4], [6, 8]])

    assert np.array_equal(x_cat.numpy(), expected_cat)
    assert np.array_equal(x_cont.numpy(), expected_cont)
