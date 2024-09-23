import hetu
import hetu.nn as nn
import torch.optim as optim
import numpy as np
import torch
import unittest
from test_utils import allclose
import os
import sys

GRAD_TEST = True

class TestNonContigOps(unittest.TestCase):
    _transpose_shapes = [
        (3, 4, 5),
        (16, 4, 16),
        (4, 8, 16, 32)
    ]

    _asstrided_test_shapes = [
        ((8, 8), (2, 2), (2, 2), 1),
        ((3,), (3, 3), (1, 0), 0),
        ((8, 8), (4, 4), (1, 2), 0),
        ((8, 8), (2, 2), (1, 1), 0),
        ((256, 256), (32, 32), (1, 1), 0),
        ((512, 512), (64, 64), (2, 2), 1),
        ((512, 1024), (16, 16), (1, 1), 0),
        ((512, 1024), (16, 16), (1, 1), 1),
        ((512, 1024), (128, 128), (1, 1), 0),
        ((512, 1024), (128, 128), (1, 1), 1),
        ((6, 4, 6, 8), (2, 3, 4, 5), (1, 2, 1, 1), 0)
    ]

    _diagonal_test_args = [
        ((10, 3, 4), 1, 0, 1),
        ((10, 5, 2), 0, 0, 1),
        ((100, 100), 0, 0, 1),
        ((100, 100), 1, 1, 0),
        ((10, 3, 4), 0, 2, 1),
        ((10, 3, 4), 0, 1, 2),
        ((10, 3, 4), 1, 0, 1),
        ((10, 3, 4), 0, 2, 1),
        ((10, 3, 4), 0, 1, 2),
        ((4, 2, 4, 4), -2, 3, 0),
        ((4, 2, 4, 4), -2, 0, 3)
    ]

    _slice_shapes = [
        (64, 256),
        (512, 128)
    ]

    def test_transpose_op(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestNonContigOps._transpose_shapes:
            x_np = np.random.randn(*shape)
            perm = np.arange(x_np.ndim)
            np.random.shuffle(perm)
            perm = list(perm)
            gt = np.transpose(x_np, perm)
            x = hetu.from_numpy(x_np)
            y = hetu.transpose(x, perm)
            self.assertTrue(allclose(y, gt))

            if GRAD_TEST:
              torch_in = torch.tensor(x_np, requires_grad=True)
              torch_out = torch_in.permute(perm).contiguous()
              torch_loss = torch_out.sum()
              torch_optimizer = optim.SGD([torch_in], lr = 0.5)
              hetu_in = hetu.Tensor(x_np, requires_grad=True)
              hetu_out = hetu.transpose(hetu_in, perm)
              hetu_loss = hetu_out.sum()
              hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
              torch_loss.backward()
              torch_optimizer.step()
              hetu_optimizer.minimize(hetu_loss)
              self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)
    
    def test_as_strided_op(self):
        print(sys._getframe().f_code.co_name)
        for shape_x, shape_y, stride, storage_offset in TestNonContigOps._asstrided_test_shapes:
            x_np = np.random.randn(*shape_x)
            gt = torch.as_strided(torch.from_numpy(x_np), shape_y, stride, storage_offset).numpy()
            x = hetu.from_numpy(x_np)
            y = hetu.as_strided(x, list(shape_y), list(stride), storage_offset)
            self.assertTrue(allclose(y, gt))

            if GRAD_TEST:
                torch_in = torch.tensor(x_np, requires_grad=True)
                torch_out = torch.as_strided(torch_in, shape_y, stride, storage_offset)
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                hetu_in = hetu.Tensor(x_np, requires_grad=True)
                hetu_out = hetu.as_strided(hetu_in, list(shape_y), list(stride), storage_offset)
                hetu_loss = hetu_out.sum()
                hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
                torch_loss.backward()
                torch_optimizer.step()
                hetu_optimizer.minimize(hetu_loss)
                self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)
        
    def test_diagonal_op(self):
        print(sys._getframe().f_code.co_name)
        for shape, offset, dim1, dim2 in TestNonContigOps._diagonal_test_args:
            x_np = np.random.randn(*shape)
            gt = np.diagonal(x_np, offset, dim1, dim2)
            x = hetu.from_numpy(x_np)
            y = hetu.diagonal(x, offset, dim1, dim2)
            self.assertTrue(allclose(y, gt))

            if GRAD_TEST:
                torch_in = torch.tensor(x_np, requires_grad=True)
                torch_out = torch.diagonal(torch_in, offset, dim1, dim2)
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                hetu_in = hetu.Tensor(x_np, requires_grad=True)
                hetu_out = hetu.diagonal(hetu_in, offset, dim1, dim2)
                hetu_loss = hetu_out.sum()
                hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
                torch_loss.backward()
                torch_optimizer.step()
                hetu_optimizer.minimize(hetu_loss)
                self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)
    
    def test_slice_op(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestNonContigOps._slice_shapes:
            x_np = np.random.randn(*shape)
            begin_pos = list(np.random.randint(0, 16, size = [2]))
            out_size = list(np.random.randint(16, 32, size = [2]))
            gt = x_np[begin_pos[0]:begin_pos[0]+out_size[0], begin_pos[1]:begin_pos[1]+out_size[1]]
            x = hetu.from_numpy(x_np)
            self.assertTrue(allclose(hetu.slice(x, begin_pos, out_size), gt))
        print(sys._getframe().f_code.co_name)

if __name__ == "__main__":
    os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
    with hetu.graph("eager"):
        with hetu.context(eager_device="cuda:0"):
            unittest.main()