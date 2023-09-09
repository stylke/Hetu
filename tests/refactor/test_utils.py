import hetu
import numpy as np

def allclose(a, b, rtol=1e-05, atol=3e-05, equal_nan=False):
    if isinstance(a, hetu.Tensor) or isinstance(a, hetu.NDArray):
        a = a.numpy(force=True)
    if isinstance(b, hetu.Tensor) or isinstance(b, hetu.NDArray):
        b = b.numpy(force=True)
    a_f = a.reshape(a.size)
    b_f = b.reshape(b.size)
    for i in range(0, min(1000, a_f.size)):
        if ((a_f[i] - b_f[i]) > 1e-1) or ((a_f[i] - b_f[i]) < -1e-1):
            # print(a.shape," ", b.shape)
            print(i,":", "x:", a_f[i], "y:", b_f[i], "delta:",  a_f[i] - b_f[i], " ", a.size, " ", b.size)
            break
    return np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)
