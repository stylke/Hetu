import hetu
import numpy as np

def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    if isinstance(a, hetu.Tensor):
        a = a.numpy(force=True)
    if isinstance(b, hetu.Tensor):
        b = b.numpy(force=True)
    return np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)
