from ..utils import get_data_dir

__all__ = [
    'test_numpy',
    'test_saving_numpy_array',
    'test_loading_numpy_array'
]


def test_numpy():
    import numpy as np

    arr = np.random.randn(4, 4)
    print('numpy array:', arr)


def test_saving_numpy_array():
    import os
    import numpy as np

    x = np.arange(10)
    np.save(os.path.join(get_data_dir(), 'arr'), x)
    print('Numpy array has been saved:', x)


def test_loading_numpy_array():
    import os
    import numpy as np

    x = np.load(os.path.join(get_data_dir(), 'arr.npy'))
    print('Loaded numpy array:', x)
