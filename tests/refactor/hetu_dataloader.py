import hetu
import numpy as np

if __name__ == "__main__":
    a = np.random.rand(7, 2).astype(np.float32)
    b = np.random.rand(2, 4).astype(np.float32)
    a_nd = hetu.NDArray(a, dtype=hetu.float32, device="cpu")
    x = None
    with hetu.graph("eager"):
        with hetu.context(eager_device="cpu"):
            x = hetu.from_numpy(b)
            mm = hetu.utils.data.DataLoader(a_nd, batch_size = 2, shuffle = True)
    print(a_nd)
    import time
    print(len(mm))
    with hetu.graph("eager"):
        with hetu.context(eager_device="cpu"):
            for u in mm:
                print(u.shape)
                out = hetu.matmul(u, x)
                print(out.numpy(force=True))
    
    print("Thank you")