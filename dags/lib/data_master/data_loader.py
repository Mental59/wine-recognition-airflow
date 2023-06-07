import numpy as np

__all__ = [
    'test_numpy'
]

def test_numpy():
    arr = np.random.randn(4, 4)
    print('numpy array:', arr)
