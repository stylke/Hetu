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

class TestAbsOps(unittest.TestCase):
    _test_shapes = [
        (8, 8),
        (128, 128),
        (256, 256)
    ]
    
    def test_abs_op(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestAbsOps._test_shapes:
            x_np = np.random.randn(*shape).astype(np.float32)
            x = hetu.from_numpy(x_np)
            # transpose to non-contiguous input
            perm = list(range(len(shape)))
            perm.reverse()
            trans_x = x.transpose(perm)
            trans_x_np = x_np.transpose(tuple(perm))
            gt = np.abs(trans_x_np)
            self.assertTrue(allclose(trans_x.abs(), gt))
            self.assertTrue(allclose(hetu.abs(trans_x), gt))
        print(sys._getframe().f_code.co_name)

class TestNegOps(unittest.TestCase):
    _test_shapes = [
        (8, 8),
        (128, 128),
        (256, 256)
    ]
    
    def test_neg_op(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestNegOps._test_shapes:
            x_np = np.random.randn(*shape).astype(np.float32)
            x = hetu.from_numpy(x_np)
            # transpose to non-contiguous input
            perm = list(range(len(shape)))
            perm.reverse()
            trans_x = x.transpose(perm)
            trans_x_np = x_np.transpose(tuple(perm))
            gt = np.negative(trans_x_np)
            self.assertTrue(allclose(trans_x.neg(), gt))
            self.assertTrue(allclose(hetu.neg(trans_x), gt))

            if GRAD_TEST:
                torch_in = torch.tensor(x_np, requires_grad=True)
                trans_torch_in = torch.permute(torch_in, tuple(perm))
                torch_out = torch.neg(trans_torch_in)
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                hetu_in = hetu.Tensor(x_np, requires_grad=True)
                trans_hetu_in = hetu_in.transpose(perm)
                hetu_out = hetu.neg(trans_hetu_in)
                hetu_loss = hetu_out.sum()
                hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
                torch_loss.backward()
                torch_optimizer.step()
                hetu_optimizer.minimize(hetu_loss)
                self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)

class TestReciprocalOps(unittest.TestCase):
    _test_shapes = [
        (8, 8),
        (128, 128),
        (256, 256)
    ]
    
    def test_reciprocal_op(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestReciprocalOps._test_shapes:
            x_np = np.random.randn(*shape).astype(np.float32)
            x = hetu.from_numpy(x_np)
            # transpose to non-contiguous input
            perm = list(range(len(shape)))
            perm.reverse()
            trans_x = x.transpose(perm)
            trans_x_np = x_np.transpose(tuple(perm))
            gt = np.reciprocal(trans_x_np)
            self.assertTrue(allclose(trans_x.reciprocal(), gt))
            self.assertTrue(allclose(hetu.reciprocal(trans_x), gt))

            if GRAD_TEST:
                torch_in = torch.tensor(x_np, requires_grad=True)
                trans_torch_in = torch.permute(torch_in, tuple(perm))
                torch_out = torch.reciprocal(trans_torch_in)
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                hetu_in = hetu.Tensor(x_np, requires_grad=True)
                trans_hetu_in = hetu_in.transpose(perm)
                hetu_out = hetu.reciprocal(trans_hetu_in)
                hetu_loss = hetu_out.sum()
                hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
                torch_loss.backward()
                torch_optimizer.step()
                hetu_optimizer.minimize(hetu_loss)
                self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)


class TestArithmeticOps(unittest.TestCase):

    _test_elementwise_shapes = [
        (256, 4), 
        (64, 256), 
        (64, 32, 16), 
    ]

    _test_broadcast_shapes = [
        ((64, 256), (64, 1)), 
        ((64, 256), (1, 256)), 
        ((64, 256), (64,)), 
    ]

    _test_pow_exponents = [
        0.0,
        -1.0,
        -2.0,
        4.0
    ]

    def test_elementwise_add(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestArithmeticOps._test_elementwise_shapes:
            x_np = np.random.randn(*shape).astype(np.float32)
            y_np = np.random.randn(*shape).astype(np.float32)
            x = hetu.from_numpy(x_np)
            y = hetu.from_numpy(y_np)
            # transpose to non-contiguous input
            perm = list(range(len(shape)))
            perm.reverse()
            trans_x = x.transpose(perm)
            trans_x_np = x_np.transpose(tuple(perm))
            trans_y = y.transpose(perm)
            trans_y_np = y_np.transpose(tuple(perm))
            c = np.random.randn()

            # tensor + tensor
            gt = trans_x_np + trans_y_np
            self.assertTrue(allclose(trans_x + trans_y, gt))
            self.assertTrue(allclose(trans_x.add(trans_y), gt))
            self.assertTrue(allclose(hetu.add(trans_x, trans_y), gt))
            # tensor + constant & constant + tensor
            gt = trans_x_np + c
            self.assertTrue(allclose(trans_x + c, gt))
            self.assertTrue(allclose(c + trans_x, gt))
            self.assertTrue(allclose(trans_x.add(c), gt))
            self.assertTrue(allclose(hetu.add(trans_x, c), gt))
            self.assertTrue(allclose(hetu.add(c, trans_x), gt))
        print(sys._getframe().f_code.co_name)

    def test_broadcast_add(self):
        print(sys._getframe().f_code.co_name)
        for shape_x, shape_y in TestArithmeticOps._test_broadcast_shapes:
            x_np = np.random.randn(*shape_x).astype(np.float32)
            y_np = np.random.randn(*shape_y).astype(np.float32)
            x = hetu.from_numpy(x_np)
            y = hetu.from_numpy(y_np)
            # transpose to non-contiguous input
            perm_x = list(range(len(shape_x)))
            perm_x.reverse()
            perm_y = list(range(len(shape_y)))
            perm_y.reverse()
            trans_x = x.transpose(perm_x)
            trans_x_np = x_np.transpose(tuple(perm_x))
            trans_y = y.transpose(perm_y)
            trans_y_np = y_np.transpose(tuple(perm_y))
            gt = trans_x_np + trans_y_np
            self.assertTrue(allclose(trans_x + trans_y, gt))
            self.assertTrue(allclose(trans_y + trans_x, gt))
            self.assertTrue(allclose(trans_x.add(trans_y), gt))
            self.assertTrue(allclose(trans_y.add(trans_x), gt))
            self.assertTrue(allclose(hetu.add(trans_x, trans_y), gt))

            if GRAD_TEST:
              torch_in = torch.tensor(y_np, requires_grad=True)
              trans_torch_in = torch.permute(torch_in, tuple(perm_y))
              torch_out = torch.add(trans_torch_in, torch.from_numpy(trans_x_np))
              torch_loss = torch_out.sum()
              torch_optimizer = optim.SGD([torch_in], lr = 0.5)
              hetu_in = hetu.Tensor(y_np, requires_grad=True)
              trans_hetu_in = hetu_in.transpose(perm_y)
              hetu_out = hetu.add(trans_hetu_in, trans_x)
              hetu_loss = hetu_out.sum()
              hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
              torch_loss.backward()
              torch_optimizer.step()
              hetu_optimizer.minimize(hetu_loss)
              self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)

    def test_sqrt(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestArithmeticOps._test_elementwise_shapes:
            x_np = np.abs(np.random.randn(*shape)).astype(np.float64)
            x = hetu.from_numpy(x_np)
            # transpose to non-contiguous input
            perm = list(range(len(shape)))
            perm.reverse()
            trans_x = x.transpose(perm)
            trans_x_np = x_np.transpose(tuple(perm))
            gt = np.sqrt(trans_x_np)
            self.assertTrue(allclose(trans_x.sqrt(), gt))
            self.assertTrue(allclose(hetu.sqrt(trans_x), gt))

            if GRAD_TEST:
                torch_in = torch.tensor(x_np, requires_grad=True)
                trans_torch_in = torch.permute(torch_in, tuple(perm))
                torch_out = torch.sqrt(trans_torch_in)
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                hetu_in = hetu.Tensor(x_np, requires_grad=True)
                trans_hetu_in = hetu_in.transpose(perm)
                hetu_out = hetu.sqrt(trans_hetu_in)
                hetu_loss = hetu_out.sum()
                hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
                torch_loss.backward()
                torch_optimizer.step()
                hetu_optimizer.minimize(hetu_loss)
                self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)

    def test_rsqrt(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestArithmeticOps._test_elementwise_shapes:
            x_np = np.abs(np.random.randn(*shape)).astype(np.float32)
            x = hetu.from_numpy(x_np)
            # transpose to non-contiguous input
            perm = list(range(len(shape)))
            perm.reverse()
            trans_x = x.transpose(perm)
            trans_x_np = x_np.transpose(tuple(perm))
            gt = np.reciprocal(np.sqrt(trans_x_np))
            self.assertTrue(allclose(trans_x.rsqrt(), gt))
            self.assertTrue(allclose(hetu.rsqrt(trans_x), gt))

            if GRAD_TEST:
                torch_in = torch.tensor(x_np, requires_grad=True)
                trans_torch_in = torch.permute(torch_in, tuple(perm))
                torch_out = torch.rsqrt(trans_torch_in)
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                hetu_in = hetu.Tensor(x_np, requires_grad=True)
                trans_hetu_in = hetu_in.transpose(perm)
                hetu_out = hetu.rsqrt(trans_hetu_in)
                hetu_loss = hetu_out.sum()
                hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
                torch_loss.backward()
                torch_optimizer.step()
                hetu_optimizer.minimize(hetu_loss)
                self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)

    def test_pow(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestArithmeticOps._test_elementwise_shapes:
            x_np = np.random.randn(*shape).astype(np.float64)
            x = hetu.from_numpy(x_np)
            # transpose to non-contiguous input
            perm = list(range(len(shape)))
            perm.reverse()
            trans_x = x.transpose(perm)
            trans_x_np = x_np.transpose(tuple(perm))
            for exponent in TestArithmeticOps._test_pow_exponents:
                gt = np.power(trans_x_np, exponent)
                self.assertTrue(allclose(trans_x.pow(exponent), gt))
                self.assertTrue(allclose(hetu.pow(trans_x, exponent), gt))

                if GRAD_TEST:
                    torch_in = torch.tensor(x_np, requires_grad=True)
                    trans_torch_in = torch.permute(torch_in, tuple(perm))
                    torch_out = torch.pow(trans_torch_in, exponent)
                    torch_loss = torch_out.sum()
                    torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                    hetu_in = hetu.Tensor(x_np, requires_grad=True)
                    trans_hetu_in = hetu_in.transpose(perm)
                    hetu_out = hetu.pow(trans_hetu_in, exponent)
                    hetu_loss = hetu_out.sum()
                    hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
                    torch_loss.backward()
                    torch_optimizer.step()
                    hetu_optimizer.minimize(hetu_loss)
                    self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)

    def test_ceil(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestArithmeticOps._test_elementwise_shapes:
            x_np = np.random.randn(*shape).astype(np.float32)
            x = hetu.from_numpy(x_np)
            # transpose to non-contiguous input
            perm = list(range(len(shape)))
            perm.reverse()
            trans_x = x.transpose(perm)
            trans_x_np = x_np.transpose(tuple(perm))
            gt = np.ceil(trans_x_np)
            self.assertTrue(allclose(trans_x.ceil(), gt))
            self.assertTrue(allclose(hetu.ceil(trans_x), gt))

            if GRAD_TEST:
                torch_in = torch.tensor(x_np, requires_grad=True)
                trans_torch_in = torch.permute(torch_in, tuple(perm))
                torch_out = torch.ceil(trans_torch_in)
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                hetu_in = hetu.Tensor(x_np, requires_grad=True)
                trans_hetu_in = hetu_in.transpose(perm)
                hetu_out = hetu.ceil(trans_hetu_in)
                hetu_loss = hetu_out.sum()
                hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
                torch_loss.backward()
                torch_optimizer.step()
                hetu_optimizer.minimize(hetu_loss)
                self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)

    def test_floor(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestArithmeticOps._test_elementwise_shapes:
            x_np = np.random.randn(*shape).astype(np.float32)
            x = hetu.from_numpy(x_np)
            # transpose to non-contiguous input
            perm = list(range(len(shape)))
            perm.reverse()
            trans_x = x.transpose(perm)
            trans_x_np = x_np.transpose(tuple(perm))
            gt = np.floor(trans_x_np)
            self.assertTrue(allclose(trans_x.floor(), gt))
            self.assertTrue(allclose(hetu.floor(trans_x), gt))

            if GRAD_TEST:
                torch_in = torch.tensor(x_np, requires_grad=True)
                trans_torch_in = torch.permute(torch_in, tuple(perm))
                torch_out = torch.floor(trans_torch_in)
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                hetu_in = hetu.Tensor(x_np, requires_grad=True)
                trans_hetu_in = hetu_in.transpose(perm)
                hetu_out = hetu.floor(trans_hetu_in)
                hetu_loss = hetu_out.sum()
                hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
                torch_loss.backward()
                torch_optimizer.step()
                hetu_optimizer.minimize(hetu_loss)
                self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)

    def test_round(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestArithmeticOps._test_elementwise_shapes:
            x_np = np.random.randn(*shape).astype(np.float32)
            x = hetu.from_numpy(x_np)
            # transpose to non-contiguous input
            perm = list(range(len(shape)))
            perm.reverse()
            trans_x = x.transpose(perm)
            trans_x_np = x_np.transpose(tuple(perm))
            gt = np.round(trans_x_np)
            self.assertTrue(allclose(trans_x.round(), gt))
            self.assertTrue(allclose(hetu.round(trans_x), gt))

            if GRAD_TEST:
                torch_in = torch.tensor(x_np, requires_grad=True)
                trans_torch_in = torch.permute(torch_in, tuple(perm))
                torch_out = torch.round(trans_torch_in)
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                hetu_in = hetu.Tensor(x_np, requires_grad=True)
                trans_hetu_in = hetu_in.transpose(perm)
                hetu_out = hetu.round(trans_hetu_in)
                hetu_loss = hetu_out.sum()
                hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
                torch_loss.backward()
                torch_optimizer.step()
                hetu_optimizer.minimize(hetu_loss)
                self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)

class TestDotOps(unittest.TestCase):

    _test_shapes = [
        ((64,), (32,), (2,)),
        ((128,), (64,), (2,))
    ]
    
    def test_dot_op(self):
        print(sys._getframe().f_code.co_name)
        for shape_x, strided_shape, strided_stride in TestDotOps._test_shapes:
            x_np = np.random.randn(*shape_x).astype(np.float32)
            y_np = np.random.randn(*shape_x).astype(np.float32)
            x = hetu.from_numpy(x_np)
            y = hetu.from_numpy(y_np)
            strided_hetu_x = hetu.as_strided(x, list(strided_shape), list(strided_stride), 0)
            strided_hetu_y = hetu.as_strided(y, list(strided_shape), list(strided_stride), 0)
            strided_torch_x = torch.as_strided(torch.from_numpy(x_np), strided_shape, strided_stride, 0)
            strided_torch_y = torch.as_strided(torch.from_numpy(y_np), strided_shape, strided_stride, 0)
            gt = torch.matmul(strided_torch_x, strided_torch_y).cpu().numpy()
            self.assertTrue(allclose(hetu.matmul(strided_hetu_x, strided_hetu_y), gt))
            self.assertTrue(allclose(strided_hetu_x.matmul(strided_hetu_y), gt))

            if GRAD_TEST:
              torch_in = torch.tensor(x_np, requires_grad=True)
              strided_torch_in = torch.as_strided(torch_in, strided_shape, strided_stride, 0)
              torch_out = torch.matmul(strided_torch_in, strided_torch_y)
              torch_loss = torch_out.sum()
              torch_optimizer = optim.SGD([torch_in], lr = 0.5)
              hetu_in = hetu.Tensor(x_np, requires_grad=True)
              strided_hetu_in = hetu.as_strided(hetu_in, list(strided_shape), list(strided_stride), 0)
              hetu_out = hetu.matmul(strided_hetu_in, strided_hetu_y)
              hetu_loss = hetu_out.sum()
              hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
              torch_loss.backward()
              torch_optimizer.step()
              hetu_optimizer.minimize(hetu_loss)
              self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)

class TestMatMulOps(unittest.TestCase):

    _matmul_shapes = [
        # 2D x 1D
        ((128, 64), (128,)),
        # 1D x 2D
        ((128,), (64, 128)),
        # 2D x 2D
        ((128, 64), (512, 128)),
        # ND x 1D
        ((128, 64, 8), (128,)),
        # 1D x ND
        ((128,), (64, 128, 8)),
        # ND x 2D
        ((128, 64, 8), (512, 128)),
        # 2D x ND
        ((128, 512), (64, 128, 2)),
        # ND x ND
        ((256, 64, 8), (8, 256, 8)),
        ((256, 64, 8), (64, 256, 8, 8)),
        ((64, 8, 16, 8), (256, 64, 16, 8))
    ]

    _linear_shapes = [
        ((128, 64), (512, 128))
    ]
    
    def test_matmul_op(self):
        print(sys._getframe().f_code.co_name)
        for shape_x, shape_y in TestMatMulOps._matmul_shapes:
            x_np = np.random.randn(*shape_x).astype(np.float32)
            y_np = np.random.randn(*shape_y).astype(np.float32)
            x = hetu.from_numpy(x_np)
            y = hetu.from_numpy(y_np)
            # transpose to non-contiguous input
            perm_x = list(range(len(shape_x)))
            perm_y = list(range(len(shape_y)))
            perm_x.reverse()
            perm_y.reverse()
            trans_x = x.transpose(perm_x)
            trans_y = y.transpose(perm_y)
            trans_x_np = x_np.transpose(tuple(perm_x))
            trans_y_np = y_np.transpose(tuple(perm_y))
            gt = np.matmul(trans_x_np, trans_y_np)
            self.assertTrue(allclose(hetu.matmul(trans_x, trans_y), gt))
            self.assertTrue(allclose(trans_x.matmul(trans_y), gt))

            if GRAD_TEST:
              torch_in = torch.tensor(x_np, requires_grad=True)
              trans_torch_in = torch.permute(torch_in, tuple(perm_x))
              torch_out = torch.matmul(trans_torch_in, torch.from_numpy(trans_y_np))
              torch_loss = torch_out.sum()
              torch_optimizer = optim.SGD([torch_in], lr = 0.5)
              hetu_in = hetu.Tensor(x_np, requires_grad=True)
              trans_hetu_in = hetu_in.transpose(perm_x)
              hetu_out = hetu.matmul(trans_hetu_in, trans_y)
              hetu_loss = hetu_out.sum()
              hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
              torch_loss.backward()
              torch_optimizer.step()
              hetu_optimizer.minimize(hetu_loss)
              self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)

    def test_linear_op(self):
        print(sys._getframe().f_code.co_name)
        for shape_x, shape_y in TestMatMulOps._linear_shapes:
            x_np = np.random.randn(*shape_x).astype(np.float32)
            w_np = np.random.randn(*shape_y).astype(np.float32)
            bias_np = np.random.randn(shape_y[0]).astype(np.float32)
            x = hetu.from_numpy(x_np)
            w = hetu.from_numpy(w_np)
            bias = hetu.from_numpy(bias_np)
            # transpose to non-contiguous input
            perm_x = list(range(len(shape_x)))
            perm_w = list(range(len(shape_y)))
            perm_x.reverse()
            perm_w.reverse()
            trans_x = x.transpose(perm_x)
            trans_w = w.transpose(perm_w)
            trans_x_np = x_np.transpose(tuple(perm_x))
            trans_w_np = w_np.transpose(tuple(perm_w))
            gt = np.matmul(trans_x_np, trans_w_np) + bias_np
            self.assertTrue(allclose(hetu.linear(trans_x, trans_w, bias), gt))

            if GRAD_TEST:
              torch_in = torch.tensor(x_np, requires_grad=True)
              trans_torch_in = torch.permute(torch_in, tuple(perm_x))
              torch_out = torch.matmul(trans_torch_in, torch.from_numpy(trans_w_np)) + torch.from_numpy(bias_np)
              torch_loss = torch_out.sum()
              torch_optimizer = optim.SGD([torch_in], lr = 0.5)
              hetu_in = hetu.Tensor(x_np, requires_grad=True)
              trans_hetu_in = hetu_in.transpose(perm_x)
              hetu_out = hetu.linear(trans_hetu_in, trans_w, bias)
              hetu_loss = hetu_out.sum()
              hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
              torch_loss.backward()
              torch_optimizer.step()
              hetu_optimizer.minimize(hetu_loss)
              self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)

class TestBatchMatMulOps(unittest.TestCase):

    _test_shapes = [
        ((128, 64, 1), (32, 128, 1)),
        ((256, 64, 16), (128, 256, 16))
    ]
    
    def test_batch_matmul_op(self):
        print(sys._getframe().f_code.co_name)
        for shape_x, shape_y in TestBatchMatMulOps._test_shapes:
            x_np = np.random.randn(*shape_x).astype(np.float32)
            y_np = np.random.randn(*shape_y).astype(np.float32)
            x = hetu.from_numpy(x_np)
            y = hetu.from_numpy(y_np)
            # transpose to non-contiguous input
            perm_x = list(range(len(shape_x)))
            perm_y = list(range(len(shape_y)))
            perm_x.reverse()
            perm_y.reverse()
            trans_x = x.transpose(perm_x)
            trans_x_np = x_np.transpose(tuple(perm_x))
            trans_y = y.transpose(perm_y)
            trans_y_np = y_np.transpose(tuple(perm_y))
            gt = torch.bmm(torch.from_numpy(trans_x_np), torch.from_numpy(trans_y_np)).numpy()
            self.assertTrue(allclose(hetu.bmm(trans_x, trans_y), gt))
            self.assertTrue(allclose(trans_x.bmm(trans_y), gt))

            if GRAD_TEST:
              torch_in = torch.tensor(x_np, requires_grad=True)
              trans_torch_in = torch.permute(torch_in, tuple(perm_x))
              torch_out = torch.bmm(trans_torch_in, torch.from_numpy(trans_y_np))
              torch_loss = torch_out.sum()
              torch_optimizer = optim.SGD([torch_in], lr = 0.5)
              hetu_in = hetu.Tensor(x_np, requires_grad=True)
              trans_hetu_in = hetu_in.transpose(perm_x)
              hetu_out = hetu.bmm(trans_hetu_in, trans_y)
              hetu_loss = hetu_out.sum()
              hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
              torch_loss.backward()
              torch_optimizer.step()
              hetu_optimizer.minimize(hetu_loss)
              self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)

class TestActivationOps(unittest.TestCase):

    _test_shapes = [
        (2, 2),
        (64, 256),
        (1024, 16)
    ]

    _test_softmax_shapes = [
        ((2, 3, 4, 5), 0),
        ((17, 8, 25, 7), 0),
        ((17, 8, 25, 7), 1),
        ((17, 8, 25, 7), 2),
        ((17, 8, 25, 7), 3)
    ]

    def test_sigmoid_op(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestActivationOps._test_shapes:
            x_np = np.random.randn(*shape).astype(np.float32)
            x = hetu.from_numpy(x_np)
            # transpose to non-contiguous input
            perm = list(range(len(shape)))
            perm.reverse()
            trans_x = x.transpose(perm)
            trans_x_np = x_np.transpose(tuple(perm))
            gt = 1 / (1 + np.exp(-trans_x_np))
            self.assertTrue(allclose(hetu.sigmoid(trans_x), gt))
            self.assertTrue(allclose(trans_x.sigmoid(), gt))

            if GRAD_TEST:
              torch_in = torch.tensor(x_np, requires_grad=True)
              trans_torch_in = torch.permute(torch_in, tuple(perm))
              torch_out = torch.sigmoid(trans_torch_in)
              torch_loss = torch_out.sum()
              torch_optimizer = optim.SGD([torch_in], lr = 0.01)
              hetu_in = hetu.Tensor(x_np, requires_grad=True)
              trans_hetu_in = hetu_in.transpose(perm)
              hetu_out = hetu.sigmoid(trans_hetu_in)
              hetu_loss = hetu_out.sum()
              hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.01)
              torch_loss.backward()
              torch_optimizer.step()
              hetu_optimizer.minimize(hetu_loss).get_or_compute()
              self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)
    
    def test_sin_op(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestActivationOps._test_shapes:
            x_np = np.random.randn(*shape).astype(np.float32)
            x = hetu.from_numpy(x_np)
            # transpose to non-contiguous input
            perm = list(range(len(shape)))
            perm.reverse()
            trans_x = x.transpose(perm)
            trans_x_np = x_np.transpose(tuple(perm))
            gt = np.sin(trans_x_np)
            self.assertTrue(allclose(hetu.sin(trans_x), gt))
            self.assertTrue(allclose(trans_x.sin(), gt))

            if GRAD_TEST:
              torch_in = torch.tensor(x_np, requires_grad=True)
              trans_torch_in = torch.permute(torch_in, tuple(perm))
              torch_out = torch.sin(trans_torch_in)
              torch_loss = torch_out.sum()
              torch_optimizer = optim.SGD([torch_in], lr = 0.01)
              hetu_in = hetu.Tensor(x_np, requires_grad=True)
              trans_hetu_in = hetu_in.transpose(perm)
              hetu_out = hetu.sin(trans_hetu_in)
              hetu_loss = hetu_out.sum()
              hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.01)
              torch_loss.backward()
              torch_optimizer.step()
              hetu_optimizer.minimize(hetu_loss).get_or_compute()
              self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)
    
    def test_relu_op(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestActivationOps._test_shapes:
            x_np = np.random.randn(*shape).astype(np.float32) - 0.5
            x = hetu.from_numpy(x_np)
            # transpose to non-contiguous input
            perm = list(range(len(shape)))
            perm.reverse()
            trans_x = x.transpose(perm)
            trans_x_np = x_np.transpose(tuple(perm))
            gt = trans_x_np * (trans_x_np > 0).astype(trans_x_np.dtype)
            self.assertTrue(allclose(hetu.relu(trans_x), gt))
            self.assertTrue(allclose(trans_x.relu(), gt))

            if GRAD_TEST:
              torch_in = torch.tensor(x_np, requires_grad=True)
              trans_torch_in = torch.permute(torch_in, tuple(perm))
              torch_out = torch.relu(trans_torch_in)
              torch_loss = torch_out.sum()
              torch_optimizer = optim.SGD([torch_in], lr = 0.01)
              hetu_in = hetu.Tensor(x_np, requires_grad=True)
              trans_hetu_in = hetu_in.transpose(perm)
              hetu_out = hetu.relu(trans_hetu_in)
              hetu_loss = hetu_out.sum()
              hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.01)
              torch_loss.backward()
              torch_optimizer.step()
              hetu_optimizer.minimize(hetu_loss).get_or_compute()
              self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)
            
    
    def test_leaky_relu_op(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestActivationOps._test_shapes:
            x_np = np.random.randn(*shape).astype(np.float32)
            x = hetu.from_numpy(x_np)
            # transpose to non-contiguous input
            perm = list(range(len(shape)))
            perm.reverse()
            trans_x = x.transpose(perm)
            trans_x_np = x_np.transpose(tuple(perm))
            alphas = [0.1, 0.2, 0.5]
            for alpha in alphas:
                gt = np.where(trans_x_np > 0, trans_x_np, alpha * trans_x_np)
                self.assertTrue(allclose(hetu.leakyrelu(trans_x, alpha), gt))
                self.assertTrue(allclose(trans_x.leakyrelu(alpha), gt))

                if GRAD_TEST:
                    torch_in = torch.tensor(x_np, requires_grad=True)
                    trans_torch_in = torch.permute(torch_in, tuple(perm))
                    torch_out = torch.nn.functional.leaky_relu(trans_torch_in, alpha)
                    torch_loss = torch_out.sum()
                    torch_optimizer = optim.SGD([torch_in], lr = 0.01)
                    hetu_in = hetu.Tensor(x_np, requires_grad=True)
                    trans_hetu_in = hetu_in.transpose(perm)
                    hetu_out = hetu.leakyrelu(trans_hetu_in, alpha)
                    hetu_loss = hetu_out.sum()
                    hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.01)
                    torch_loss.backward()
                    torch_optimizer.step()
                    hetu_optimizer.minimize(hetu_loss).get_or_compute()
                    self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)

    def test_tanh_op(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestActivationOps._test_shapes:
            x_np = np.random.randn(*shape).astype(np.float32)
            x = hetu.from_numpy(x_np)
            # transpose to non-contiguous input
            perm = list(range(len(shape)))
            perm.reverse()
            trans_x = x.transpose(perm)
            trans_x_np = x_np.transpose(tuple(perm))
            gt = np.tanh(trans_x_np)
            self.assertTrue(allclose(hetu.tanh(trans_x), gt))
            self.assertTrue(allclose(trans_x.tanh(), gt))

            if GRAD_TEST:
              torch_in = torch.tensor(x_np, requires_grad=True)
              trans_torch_in = torch.permute(torch_in, tuple(perm))
              torch_out = torch.tanh(trans_torch_in)
              torch_loss = torch_out.sum()
              torch_optimizer = optim.SGD([torch_in], lr = 0.01)
              hetu_in = hetu.Tensor(x_np, requires_grad=True)
              trans_hetu_in = hetu_in.transpose(perm)
              hetu_out = hetu.tanh(trans_hetu_in)
              hetu_loss = hetu_out.sum()
              hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.01)
              torch_loss.backward()
              torch_optimizer.step()
              hetu_optimizer.minimize(hetu_loss).get_or_compute()
              self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)

    def test_triu_op(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestActivationOps._test_shapes:
            x_np = np.random.randn(*shape)
            x = hetu.from_numpy(x_np)
            # transpose to non-contiguous input
            perm = list(range(len(shape)))
            perm.reverse()
            trans_x = x.transpose(perm)
            trans_x_np = x_np.transpose(tuple(perm))
            gt = torch.triu(torch.from_numpy(trans_x_np), 0).numpy()
            self.assertTrue(allclose(hetu.triu(trans_x, False, 0), gt))
            self.assertTrue(allclose(trans_x.triu(False, 0), gt))

            if GRAD_TEST:
              torch_in = torch.tensor(x_np, requires_grad=True)
              trans_torch_in = torch.permute(torch_in, tuple(perm))
              torch_out = torch.triu(trans_torch_in, 0)
              torch_loss = torch_out.sum()
              torch_optimizer = optim.SGD([torch_in], lr = 0.01)
              hetu_in = hetu.Tensor(x_np, requires_grad=True)
              trans_hetu_in = hetu_in.transpose(perm)
              hetu_out = hetu.triu(trans_hetu_in, False, 0)
              hetu_loss = hetu_out.sum()
              hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.01)
              torch_loss.backward()
              torch_optimizer.step()
              hetu_optimizer.minimize(hetu_loss).get_or_compute()
              self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)

    def test_tril_op(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestActivationOps._test_shapes:
            x_np = np.random.randn(*shape)
            x = hetu.from_numpy(x_np)
            # transpose to non-contiguous input
            perm = list(range(len(shape)))
            perm.reverse()
            trans_x = x.transpose(perm)
            trans_x_np = x_np.transpose(tuple(perm))
            gt = torch.tril(torch.from_numpy(trans_x_np), 0).numpy()
            self.assertTrue(allclose(hetu.triu(trans_x, True, 0), gt))
            self.assertTrue(allclose(trans_x.triu(True, 0), gt))

            if GRAD_TEST:
              torch_in = torch.tensor(x_np, requires_grad=True)
              trans_torch_in = torch.permute(torch_in, tuple(perm))
              torch_out = torch.tril(trans_torch_in, 0)
              torch_loss = torch_out.sum()
              torch_optimizer = optim.SGD([torch_in], lr = 0.01)
              hetu_in = hetu.Tensor(x_np, requires_grad=True)
              trans_hetu_in = hetu_in.transpose(perm)
              hetu_out = hetu.triu(trans_hetu_in, True, 0)
              hetu_loss = hetu_out.sum()
              hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.01)
              torch_loss.backward()
              torch_optimizer.step()
              hetu_optimizer.minimize(hetu_loss).get_or_compute()
              self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)
            
    
    def test_softmax_op(self):
        print(sys._getframe().f_code.co_name)
        for shape, dim in TestActivationOps._test_softmax_shapes:
            x_np = np.random.randn(*shape).astype(np.float32)
            x = hetu.from_numpy(x_np)
            # transpose to non-contiguous input
            perm = list(range(len(shape)))
            perm.reverse()
            trans_x = x.transpose(perm)
            trans_x_np = x_np.transpose(tuple(perm))
            gt = torch.softmax(torch.from_numpy(trans_x_np), dim).numpy()

            self.assertTrue(allclose(hetu.softmax(trans_x, dim), gt))
            self.assertTrue(allclose(trans_x.softmax(dim), gt))

            if GRAD_TEST:
              torch_in = torch.tensor(x_np, requires_grad=True)
              trans_torch_in = torch.permute(torch_in, tuple(perm))
              torch_out = torch.softmax(trans_torch_in, 0)
              torch_loss = torch_out.sum()
              torch_optimizer = optim.SGD([torch_in], lr = 0.01)
              hetu_in = hetu.Tensor(x_np, requires_grad=True)
              trans_hetu_in = hetu_in.transpose(perm)
              hetu_out = hetu.softmax(trans_hetu_in, 0)
              hetu_loss = hetu_out.sum()
              hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.01)
              torch_loss.backward()
              torch_optimizer.step()
              hetu_optimizer.minimize(hetu_loss).get_or_compute()
              self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)


class TestTransformOps(unittest.TestCase):

    _test_shapes = [
        (64, 256),
        (128, 128),
        (2, 2)
    ]

    _pad_shapes = [
        (8, 4, 32, 32),
        (16, 4, 16, 16)
    ]

    _transposeshapes = [
        (16, 4, 16),
        (4, 8, 16, 32)
    ]

    def test_reshape_op(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestTransformOps._test_shapes:
            x_np = np.random.randn(*shape)
            x = hetu.from_numpy(x_np)
            # transpose to non-contiguous input
            perm = list(range(len(shape)))
            perm.reverse()
            trans_x = x.transpose(perm)
            trans_x_np = x_np.transpose(tuple(perm))
            shape_to = list(shape)
            shape_to[0] = int(shape_to[0] / 2)
            shape_to[1] *= 2
            gt = np.reshape(trans_x_np, tuple(shape_to))
            self.assertTrue(allclose(hetu.reshape(trans_x, shape_to), gt))
            self.assertTrue(allclose(trans_x.reshape(shape_to), gt))

            if GRAD_TEST:
              torch_in = torch.tensor(x_np, requires_grad=True)
              trans_torch_in = torch.permute(torch_in, tuple(perm))
              torch_out = torch.reshape(trans_torch_in, tuple(shape_to)).contiguous()
              torch_loss = torch_out.sum()
              torch_optimizer = optim.SGD([torch_in], lr = 0.5)
              hetu_in = hetu.Tensor(x_np, requires_grad=True)
              trans_hetu_in = hetu_in.transpose(perm)
              hetu_out = hetu.reshape(trans_hetu_in, shape_to)
              hetu_loss = hetu_out.sum()
              hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
              torch_loss.backward()
              torch_optimizer.step()
              hetu_optimizer.minimize(hetu_loss).get_or_compute()
              self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)

    def test_broadcast_op(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestTransformOps._test_shapes:
            x_np = np.random.randn(*shape).astype(np.float32)
            x = hetu.from_numpy(x_np)
            # transpose to non-contiguous input
            perm = list(range(len(shape)))
            perm.reverse()
            trans_x = x.transpose(perm)
            trans_x_np = x_np.transpose(tuple(perm))
            reverse_shape = list(shape)
            reverse_shape.reverse()
            shape_to = reverse_shape
            shape_to = [16] + reverse_shape
            gt = np.broadcast_to(trans_x_np, tuple(shape_to))
            self.assertTrue(allclose(hetu.broadcast(trans_x, shape_to, []), gt))
            self.assertTrue(allclose(trans_x.broadcast(shape_to, []), gt))

            if GRAD_TEST:
              torch_in = torch.tensor(x_np, requires_grad=True)
              trans_torch_in = torch.permute(torch_in, tuple(perm))
              torch_out = torch.broadcast_to(trans_torch_in, tuple(shape_to))
              torch_loss = torch_out.sum()
              torch_optimizer = optim.SGD([torch_in], lr = 0.01)
              hetu_in = hetu.Tensor(x_np, requires_grad=True)
              trans_hetu_in = hetu_in.transpose(perm)
              hetu_out = hetu.broadcast(trans_hetu_in, shape_to, [])
              hetu_loss = hetu_out.sum()
              hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.01)
              torch_loss.backward()
              torch_optimizer.step()
              hetu_optimizer.minimize(hetu_loss).get_or_compute()
              self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)

    def test_concat_op(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestTransformOps._test_shapes:
            x_np = np.random.randn(*shape).astype(np.float32)
            y_np = np.random.randn(*shape).astype(np.float32)
            z_np = np.random.randn(*shape).astype(np.float32)
            x = hetu.from_numpy(x_np)
            y = hetu.from_numpy(y_np)
            z = hetu.from_numpy(z_np)
            # transpose to non-contiguous input
            perm = list(range(len(shape)))
            perm.reverse()
            trans_x = x.transpose(perm)
            trans_x_np = x_np.transpose(tuple(perm))
            trans_y = y.transpose(perm)
            trans_y_np = y_np.transpose(tuple(perm))
            trans_z = z.transpose(perm)
            trans_z_np = z_np.transpose(tuple(perm))
            gt = np.concatenate((trans_x_np, trans_y_np), 0)
            self.assertTrue(allclose(hetu.concat(trans_x, trans_y, 0), gt))
            self.assertTrue(allclose(trans_x.concat(trans_y, 0), gt))
            self.assertTrue(allclose(hetu.concat([trans_x, trans_y], 0), gt))
            gt = np.concatenate((trans_x_np, trans_y_np, trans_z_np), 0)
            self.assertTrue(allclose(hetu.concat([trans_x, trans_y, trans_z], 0), gt))
        print(sys._getframe().f_code.co_name)
    
    def test_pad_op(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestTransformOps._pad_shapes:
            x_np = np.random.randn(*shape)
            x = hetu.from_numpy(x_np)
            # transpose to non-contiguous input
            perm = list(range(len(shape)))
            perm.reverse()
            trans_x = x.transpose(perm)
            trans_x_np = x_np.transpose(tuple(perm))
            gt = np.pad(trans_x_np, ((0,0),(0,0),(1,1),(2,2)), "constant", constant_values = 0.1)
            self.assertTrue(allclose(hetu.pad(trans_x, [1,1,2,2], "constant", 0.1), gt))
            self.assertTrue(allclose(trans_x.pad([1,1,2,2], "constant", 0.1), gt))

            if GRAD_TEST:
              torch_in = torch.tensor(x_np, requires_grad=True)
              trans_torch_in = torch.permute(torch_in, tuple(perm))
              torch_out = torch.nn.functional.pad(trans_torch_in, (0,0,0,0,1,1,2,2), "constant", 0.1)
              torch_loss = torch_out.sum()
              torch_optimizer = optim.SGD([torch_in], lr = 0.01)
              hetu_in = hetu.Tensor(x_np, requires_grad=True)
              trans_hetu_in = hetu_in.transpose(perm)
              hetu_out = hetu.pad(trans_hetu_in, [1,1,2,2], "constant", 0.1)
              hetu_loss = hetu_out.sum()
              hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.01)
              torch_loss.backward()
              torch_optimizer.step()
              hetu_optimizer.minimize(hetu_loss).get_or_compute()
              self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)

class TestNormOps(unittest.TestCase):

    _test_shapes = [
        (16, 16, 3, 4),      
        (16, 16, 8, 5)  
    ]

    def test_layernorm_op(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestNormOps._test_shapes:
            reverse_shape = list(shape)
            reverse_shape.reverse()
            for i in range(1, 4):
                norm_shape = reverse_shape[i:]
                x_np = np.random.randn(*shape).astype(np.float32)
                x = hetu.from_numpy(x_np)
                scale_np = np.ones(tuple(norm_shape)).astype(np.float32)
                scale = hetu.from_numpy(scale_np)
                bias_np = np.zeros(tuple(norm_shape)).astype(np.float32)
                bias = hetu.from_numpy(bias_np)
                # transpose to non-contiguous input
                perm = list(range(len(shape)))
                perm.reverse()
                trans_x = x.transpose(perm)
                trans_x_np = x_np.transpose(tuple(perm))
                layernorm = torch.nn.LayerNorm(tuple(norm_shape), 1e-5)
                gt = layernorm(torch.from_numpy(trans_x_np)).detach().numpy()
                gt2 = torch.layer_norm(torch.from_numpy(trans_x_np), normalized_shape=tuple(norm_shape), weight = torch.from_numpy(scale_np), bias = torch.from_numpy(bias_np),
                                    eps=1e-5).numpy()
                self.assertTrue(allclose(gt2, gt))
                self.assertTrue(allclose(hetu.layer_norm(trans_x, scale, bias, list(norm_shape), 1e-5)[0], gt))
                self.assertTrue(allclose(trans_x.layer_norm(scale, bias, list(norm_shape), 1e-5)[0], gt))

                if GRAD_TEST:
                    torch_in = torch.tensor(x_np, requires_grad=True)
                    trans_torch_in = torch.permute(torch_in, tuple(perm))
                    torch_out = layernorm(trans_torch_in)
                    torch_loss = torch_out.sum()
                    torch_optimizer = optim.SGD([torch_in], lr = 0.01)
                    hetu_in = hetu.Tensor(x_np, requires_grad=True)
                    trans_hetu_in = hetu_in.transpose(perm)
                    hetu_out = hetu.layer_norm(trans_hetu_in, scale, bias, list(norm_shape), 1e-5)[0]
                    hetu_loss = hetu_out.sum()
                    hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.01)
                    torch_loss.backward()
                    torch_optimizer.step()
                    hetu_optimizer.minimize(hetu_loss).get_or_compute()
                    self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)
    
    def test_instancenorm_op(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestNormOps._test_shapes:
            x_np = np.random.randn(*shape).astype(np.float32)
            x = hetu.from_numpy(x_np)
            # transpose to non-contiguous input
            perm = list(range(len(shape)))
            perm.reverse()
            trans_x = x.transpose(perm)
            trans_x_np = x_np.transpose(tuple(perm))
            instancenorm = torch.nn.InstanceNorm2d(num_features=shape[1], eps=1e-5)
            gt = instancenorm(torch.from_numpy(trans_x_np)).detach().numpy()
            self.assertTrue(allclose(hetu.instance_norm(trans_x, 1e-5)[0], gt))
            self.assertTrue(allclose(trans_x.instance_norm(1e-5)[0], gt))

            if GRAD_TEST:
                torch_in = torch.tensor(x_np, requires_grad=True)
                trans_torch_in = torch.permute(torch_in, tuple(perm))
                torch_out = instancenorm(trans_torch_in)
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_in], lr = 0.01)
                hetu_in = hetu.Tensor(x_np, requires_grad=True)
                trans_hetu_in = hetu_in.transpose(perm)
                hetu_out = hetu.instance_norm(trans_hetu_in, 1e-5)[0]
                hetu_loss = hetu_out.sum()
                hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.01)
                torch_loss.backward()
                torch_optimizer.step()
                hetu_optimizer.minimize(hetu_loss).get_or_compute()
                self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)

class TestReduceOps(unittest.TestCase):

    _test_shapes = [
        (17, 8, 25, 7),
        (16, 4, 16, 16),
        (1, 8, 32, 32),
    ]
    
    def test_reduce_sum_op(self):
        print(sys._getframe().f_code.co_name)
        for shape_x in TestReduceOps._test_shapes:
            x_np = np.random.randn(*shape_x).astype(np.float32)
            x = hetu.from_numpy(x_np)
            # transpose to non-contiguous input
            perm = list(range(len(shape_x)))
            perm.reverse()
            trans_x = x.transpose(perm)
            trans_x_np = x_np.transpose(tuple(perm))
            gt = np.sum(trans_x_np)
            self.assertTrue(allclose(hetu.reduce(trans_x, "sum"), gt))
            self.assertTrue(allclose(trans_x.reduce("sum"), gt))
            self.assertTrue(allclose(hetu.sum(trans_x), gt))
            self.assertTrue(allclose(trans_x.sum(), gt))

            def get_axes(i):
                axes = []
                ins = 0
                while i > 0:
                    if (i % 2):
                        axes.append(ins)
                    i = i // 2
                    ins += 1
                return axes
            for i in range(1, pow(2, len(shape_x))):
                axes = get_axes(i)
                torch_x = torch.from_numpy(trans_x_np).to("cuda:0")
                gt = torch.sum(torch_x, axes)
                self.assertTrue(allclose(hetu.sum(trans_x, axes), gt.cpu().numpy()))
                self.assertTrue(allclose(trans_x.sum(axes), gt.cpu().numpy()))
                # keepdim test
                gt = torch.sum(torch_x, axes, keepdims=True)
                self.assertTrue(allclose(hetu.sum(trans_x, axes, [True]), gt.cpu().numpy()))
        print(sys._getframe().f_code.co_name)
                

    def test_reduce_mean_op(self):
        print(sys._getframe().f_code.co_name)
        for shape_x in TestReduceOps._test_shapes:
            x_np = np.random.randn(*shape_x).astype(np.float32)
            x = hetu.from_numpy(x_np)
            # transpose to non-contiguous input
            perm = list(range(len(shape_x)))
            perm.reverse()
            trans_x = x.transpose(perm)
            trans_x_np = x_np.transpose(tuple(perm))
            gt = np.average(trans_x_np)
            self.assertTrue(allclose(hetu.reduce(trans_x, "mean"), gt))
            self.assertTrue(allclose(trans_x.reduce("mean"), gt))
            self.assertTrue(allclose(hetu.mean(trans_x), gt))
            self.assertTrue(allclose(trans_x.mean(), gt))

            def get_axes(i):
                axes = []
                ins = 0
                while i > 0:
                    if (i % 2):
                        axes.append(ins)
                    i = i // 2
                    ins += 1
                return axes
            for i in range(1, pow(2, len(shape_x))):
                axes = get_axes(i)
                torch_x = torch.from_numpy(trans_x_np).to("cuda:0")
                gt = torch.mean(torch_x, axes)
                self.assertTrue(allclose(hetu.mean(trans_x, axes), gt.cpu().numpy()))
                self.assertTrue(allclose(trans_x.mean(axes), gt.cpu().numpy()))
                # keepdim test
                gt = torch.mean(torch_x, axes, keepdims=True)
                self.assertTrue(allclose(hetu.mean(trans_x, axes, [True]), gt.cpu().numpy()))
        print(sys._getframe().f_code.co_name)

class TestLossOps(unittest.TestCase):
    _test_binary_label_shapes = [
        (1, 64)
    ]

    _test_nllloss_label_shapes = [
        ((64, 16), (16, ))
    ]

    _test_cross_entropy_label_shapes = [
        (64, 16)
    ]

    def test_bce_op(self):
        print(sys._getframe().f_code.co_name)
        MIN_VALUE = -100.0
        for shape in TestLossOps._test_binary_label_shapes:
            probs_np = np.random.uniform(1e-10, 1, size=shape).astype(np.float32)
            labels_np = np.random.choice([0, 1], size=shape).astype(np.float32)
            probs = hetu.from_numpy(probs_np)
            labels = hetu.from_numpy(labels_np)
            # transpose to non-contiguous input
            perm = list(range(len(shape)))
            perm.reverse()
            trans_probs = probs.transpose(perm)
            trans_labels = labels.transpose(perm)
            trans_probs_np = probs_np.transpose(tuple(perm))
            trans_labels_np = labels_np.transpose(tuple(perm)).copy()
            loss = hetu.binary_cross_entropy(trans_probs, trans_labels)
            bce = torch.nn.BCELoss(reduction="mean")
            gt = bce(torch.from_numpy(trans_probs_np), torch.from_numpy(trans_labels_np)).numpy()
            self.assertTrue(allclose(loss, gt))

            if GRAD_TEST:
                torch_probs = torch.tensor(probs_np, requires_grad=True)
                trans_torch_probs = torch.permute(torch_probs, tuple(perm))
                torch_out = bce(trans_torch_probs, torch.from_numpy(trans_labels_np))
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_probs], lr = 0.01)
                hetu_in = hetu.Tensor(probs_np, requires_grad=True)
                trans_hetu_in = hetu_in.transpose(perm)
                hetu_out = hetu.binary_cross_entropy(trans_hetu_in, hetu.from_numpy(trans_labels_np))
                hetu_loss = hetu_out.sum()
                hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.01)
                torch_loss.backward()
                torch_optimizer.step()
                hetu_optimizer.minimize(hetu_loss).get_or_compute()
                self.assertTrue(allclose(hetu_in, torch_probs.detach().numpy()))
        print(sys._getframe().f_code.co_name)

    def test_nllloss_op(self):
        print(sys._getframe().f_code.co_name)
        for shape, lshape in TestLossOps._test_nllloss_label_shapes:
            probs_np = np.random.uniform(1e-10, 1, size=shape).astype(np.float32)
            labels_np = np.random.choice(range(16), size=lshape).astype(np.int64)
            probs = hetu.from_numpy(probs_np)
            labels = hetu.from_numpy(labels_np)
            # transpose to non-contiguous input
            perm = list(range(len(shape)))
            perm.reverse()
            trans_probs = probs.transpose(perm)
            trans_probs_np = probs_np.transpose(tuple(perm))
            gt = torch.nn.functional.nll_loss(torch.from_numpy(trans_probs_np), torch.from_numpy(labels_np)).numpy()
            loss = hetu.nll_loss(trans_probs, labels)
            self.assertTrue(allclose(loss, gt))

            if GRAD_TEST:
                torch_in = torch.tensor(probs_np, requires_grad=True)
                trans_torch_in = torch.permute(torch_in, tuple(perm))
                torch_out = torch.nn.functional.nll_loss(trans_torch_in, torch.from_numpy(labels_np))
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                hetu_in = hetu.Tensor(probs_np, requires_grad=True)
                trans_hetu_in = hetu_in.transpose(perm)
                hetu_out = hetu.nll_loss(trans_hetu_in, labels)
                hetu_loss = hetu_out.sum()
                hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
                torch_loss.backward()
                torch_optimizer.step()
                hetu_optimizer.minimize(hetu_loss)
                self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)
    
    def test_kldivloss_op(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestLossOps._test_binary_label_shapes:
            probs_np = np.random.uniform(1e-10, 1, size=shape).astype(np.float32)
            labels_np = np.random.choice([0, 1], size=shape).astype(np.float32)
            probs = hetu.from_numpy(probs_np)
            labels = hetu.from_numpy(labels_np)
            # transpose to non-contiguous input
            perm = list(range(len(shape)))
            perm.reverse()
            trans_probs = probs.transpose(perm)
            trans_labels = labels.transpose(perm)
            trans_probs_np = probs_np.transpose(tuple(perm))
            trans_labels_np = labels_np.transpose(tuple(perm)).copy()
            gt = torch.nn.functional.kl_div(torch.from_numpy(trans_probs_np), torch.from_numpy(trans_labels_np)).numpy()
            loss = hetu.kl_div(trans_probs, trans_labels)
            self.assertTrue(allclose(loss, gt))

            if GRAD_TEST:
                torch_in = torch.tensor(probs_np, requires_grad=True)
                trans_torch_in = torch.permute(torch_in, tuple(perm))
                torch_out = torch.nn.functional.kl_div(trans_torch_in, torch.from_numpy(trans_labels_np))
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                hetu_in = hetu.Tensor(probs_np, requires_grad=True)
                trans_hetu_in = hetu_in.transpose(perm)
                hetu_out = hetu.kl_div(trans_hetu_in, hetu.from_numpy(trans_labels_np))
                hetu_loss = hetu_out.sum()
                hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
                torch_loss.backward()
                torch_optimizer.step()
                hetu_optimizer.minimize(hetu_loss)
                self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)
    
    def test_mseloss_op(self):
        print(sys._getframe().f_code.co_name)
        MIN_VALUE = -100.0
        for shape in TestLossOps._test_binary_label_shapes:
            probs_np = np.random.uniform(1e-10, 1, size=shape).astype(np.float32)
            labels_np = np.random.choice([0, 1], size=shape).astype(np.float32)
            probs = hetu.from_numpy(probs_np)
            labels = hetu.from_numpy(labels_np)
            # transpose to non-contiguous input
            perm = list(range(len(shape)))
            perm.reverse()
            trans_probs = probs.transpose(perm)
            trans_labels = labels.transpose(perm)
            trans_probs_np = probs_np.transpose(tuple(perm))
            trans_labels_np = labels_np.transpose(tuple(perm)).copy()
            gt = torch.nn.functional.mse_loss(torch.from_numpy(trans_probs_np), torch.from_numpy(trans_labels_np)).numpy()
            loss = hetu.mse_loss(trans_probs, trans_labels)
            self.assertTrue(allclose(loss, gt))

            if GRAD_TEST:
                torch_in = torch.tensor(probs_np, requires_grad=True)
                trans_torch_in = torch.permute(torch_in, tuple(perm))
                torch_out = torch.nn.functional.mse_loss(trans_torch_in, torch.from_numpy(trans_labels_np))
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_in], lr = 0.01)
                hetu_in = hetu.Tensor(probs_np, requires_grad=True)
                trans_hetu_in = hetu_in.transpose(perm)
                hetu_out = hetu.mse_loss(trans_hetu_in, hetu.from_numpy(trans_labels_np))
                hetu_loss = hetu_out.sum()
                hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.01)
                torch_loss.backward()
                torch_optimizer.step()
                hetu_optimizer.minimize(hetu_loss).get_or_compute()
                self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)

class TestOtherOps(unittest.TestCase):

    _embedding_test_shapes = [
        ((4, 4), (5)),
        ((16, 32), (16))
    ]

    _maskedfill_test_shapes = [
        ((3, 4, 5, 6),),
        ((1, 9, 1, 10),)
    ]

    _norm_test_shapes = [
        ((4, 5, 2, 3), 2, 2),
        ((3, 4, 5, 5), 0, 1)
    ]

    _repeat_test_shapes = [
        ((3, 5, 7), (2, 2, 3, 4)),
        ((2, 4, 6, 8), (2, 3, 4, 5) )
    ]

    _roll_test_shapes = [
        ((2, 2), (1,), (0,)),
        ((3, 6, 7, 9), (2, 4, 6), (0, 1, 3)),
        ((2, 4, 6, 8), (1, 7), (2, 3) )
    ]

    _gather_test_shapes = [
        ((2, 2), (2, 1), 1),
        ((32, 16, 5), (1, 16, 32), 0)
    ]

    _onehot_test_shapes = [
        (32, 4),
        (64, 2)
    ]

    def test_gatherop(self):
        print(sys._getframe().f_code.co_name)
        for shape_x, shape_id, dim in TestOtherOps._gather_test_shapes:
            x_np = np.random.randn(*shape_x)
            id_np = np.random.randint(0, shape_x[len(shape_x) - dim - 1], size=shape_id)
            x = hetu.from_numpy(x_np)
            id = hetu.from_numpy(id_np)
            # transpose to non-contiguous input
            perm = list(range(len(shape_x)))
            perm.reverse()
            trans_x = x.transpose(perm)
            trans_x_np = x_np.transpose(tuple(perm))
            gt = torch.gather(torch.from_numpy(trans_x_np), dim, torch.from_numpy(id_np)).numpy()
            self.assertTrue(allclose(hetu.gather(trans_x, dim, id), gt))
            self.assertTrue(allclose(trans_x.gather(dim, id), gt))

            if GRAD_TEST:
                torch_in = torch.tensor(x_np, requires_grad=True)
                trans_torch_in = torch.permute(torch_in, tuple(perm))
                torch_out = torch.gather(trans_torch_in, dim, torch.from_numpy(id_np))
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_in], lr = 0.01)
                hetu_in = hetu.Tensor(x_np, requires_grad=True)
                trans_hetu_in = hetu_in.transpose(perm)
                hetu_out = hetu.gather(trans_hetu_in, dim, id)
                hetu_loss = hetu_out.sum()
                hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.01)
                torch_loss.backward()
                torch_optimizer.step()
                hetu_optimizer.minimize(hetu_loss).get_or_compute()
                self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)
    
    def test_maskedfillop(self):
        print(sys._getframe().f_code.co_name)
        for shape_x in TestOtherOps._maskedfill_test_shapes:
            shape_x = shape_x[0]
            x_np = np.random.randn(*shape_x)
            mask_np = np.random.choice([0, 1], size=shape_x).astype(np.int64)
            x = hetu.from_numpy(x_np)
            mask = hetu.from_numpy(mask_np)
            # transpose to non-contiguous input
            perm = list(range(len(shape_x)))
            perm.reverse()
            trans_x = x.transpose(perm)
            trans_x_np = x_np.transpose(tuple(perm))
            trans_mask = mask.transpose(perm)
            trans_mask_np = mask_np.transpose(tuple(perm))
            val = np.random.random()
            gt = torch.masked_fill(torch.from_numpy(trans_x_np), torch.from_numpy(trans_mask_np), val).numpy()
            self.assertTrue(allclose(hetu.masked_fill(trans_x, trans_mask, val), gt))
            self.assertTrue(allclose(trans_x.masked_fill(trans_mask, val), gt))

            if GRAD_TEST:
                torch_in = torch.tensor(x_np, requires_grad=True)
                # trans_torch_in = torch.permute(torch_in, tuple(perm))
                torch_out = torch.masked_fill(torch_in, torch.from_numpy(mask_np), val)
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_in], lr = 0.01)
                hetu_in = hetu.Tensor(x_np, requires_grad=True)
                # trans_hetu_in = hetu_in.transpose(perm)
                hetu_out = hetu.masked_fill(hetu_in, mask)
                hetu_loss = hetu_out.sum()
                hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.01)
                torch_loss.backward()
                torch_optimizer.step()
                hetu_optimizer.minimize(hetu_loss).get_or_compute()
                self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)

    def test_normop(self):
        print(sys._getframe().f_code.co_name)
        for shape_x, dim0, p0 in TestOtherOps._norm_test_shapes:
            x_np = np.random.randn(*shape_x).astype(np.float32)
            x = hetu.from_numpy(x_np)
            # transpose to non-contiguous input
            perm = list(range(len(shape_x)))
            perm.reverse()
            trans_x = x.transpose(perm)
            trans_x_np = x_np.transpose(tuple(perm))
            gt = torch.norm(torch.from_numpy(trans_x_np), p=p0, dim=dim0).numpy()
            self.assertTrue(allclose(hetu.norm(trans_x, p0, dim0), gt))
            self.assertTrue(allclose(trans_x.norm(p0, dim0), gt))

            if GRAD_TEST:
                torch_in = torch.tensor(x_np, requires_grad=True)
                trans_torch_in = torch.permute(torch_in, tuple(perm))
                torch_out = torch.norm(trans_torch_in, p=p0, dim=dim0)
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_in], lr = 0.01)
                hetu_in = hetu.Tensor(x_np, requires_grad=True)
                trans_hetu_in = hetu_in.transpose(perm)
                hetu_out = hetu.norm(trans_hetu_in, p0, dim0)
                hetu_loss = hetu_out.sum()
                hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.01)
                torch_loss.backward()
                torch_optimizer.step()
                hetu_optimizer.minimize(hetu_loss).get_or_compute()
                self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)
    
    def test_repeatop(self):
        print(sys._getframe().f_code.co_name)
        for shape_x, repeats in TestOtherOps._repeat_test_shapes:
            x_np = np.random.randn(*shape_x)
            x = hetu.from_numpy(x_np)
            # transpose to non-contiguous input
            perm = list(range(len(shape_x)))
            perm.reverse()
            trans_x = x.transpose(perm)
            trans_x_np = x_np.transpose(tuple(perm))
            gt = torch.from_numpy(trans_x_np).repeat(*repeats).numpy()
            self.assertTrue(allclose(hetu.repeat(trans_x, list(repeats)), gt))
            self.assertTrue(allclose(trans_x.repeat(list(repeats)), gt))

            if GRAD_TEST:
                torch_in = torch.tensor(x_np, requires_grad=True)
                trans_torch_in = torch.permute(torch_in, tuple(perm))
                torch_out = trans_torch_in.repeat(*repeats)
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_in], lr = 0.01)
                hetu_in = hetu.Tensor(x_np, requires_grad=True)
                trans_hetu_in = hetu_in.transpose(perm)
                hetu_out = hetu.repeat(trans_hetu_in, list(repeats))
                hetu_loss = hetu_out.sum()
                hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.01)
                torch_loss.backward()
                torch_optimizer.step()
                hetu_optimizer.minimize(hetu_loss).get_or_compute()
                self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)

    def test_rollop(self):
        print(sys._getframe().f_code.co_name)
        for shape_x, shifts, dims in TestOtherOps._roll_test_shapes:
            x_np = np.random.randn(*shape_x)
            x = hetu.from_numpy(x_np)
            # transpose to non-contiguous input
            perm = list(range(len(shape_x)))
            perm.reverse()
            trans_x = x.transpose(perm)
            trans_x_np = x_np.transpose(tuple(perm))
            gt = torch.roll(torch.from_numpy(trans_x_np), shifts=shifts, dims=dims).numpy()
            # print(hetu.roll(x, list(shifts), list(dims)).numpy(force=True), "\n", gt)
            self.assertTrue(allclose(hetu.roll(trans_x, list(shifts), list(dims)), gt))
            self.assertTrue(allclose(trans_x.roll(list(shifts), list(dims)), gt))

            if GRAD_TEST:
                torch_in = torch.tensor(x_np, requires_grad=True)
                trans_torch_in = torch.permute(torch_in, tuple(perm))
                torch_out = torch.roll(trans_torch_in, shifts, dims)
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_in], lr = 0.01)
                hetu_in = hetu.Tensor(x_np, requires_grad=True)
                trans_hetu_in = hetu_in.transpose(perm)
                hetu_out = hetu.roll(trans_hetu_in, list(shifts), list(dims))
                hetu_loss = hetu_out.sum()
                hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.01)
                torch_loss.backward()
                torch_optimizer.step()
                hetu_optimizer.minimize(hetu_loss).get_or_compute()
                self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)
    
    def test_embedding_lookupop(self):
        print(sys._getframe().f_code.co_name)
        for shape_x, shape_id in TestOtherOps._embedding_test_shapes:
            x_np = np.random.randn(*shape_x)
            id_np = np.random.randint(0, shape_x[0], size=shape_id)
            x = hetu.from_numpy(x_np)
            id = hetu.from_numpy(id_np)
            # transpose to non-contiguous input
            perm = list(range(len(shape_x)))
            perm.reverse()
            trans_x = x.transpose(perm)
            trans_x_np = x_np.transpose(tuple(perm))
            gt = torch.embedding(torch.from_numpy(trans_x_np), torch.from_numpy(id_np)).numpy()
            self.assertTrue(allclose(hetu.embedding_lookup(trans_x, id), gt))
            self.assertTrue(allclose(trans_x.embedding_lookup(id), gt))

            if GRAD_TEST:
                torch_in = torch.tensor(x_np, requires_grad=True)
                trans_torch_in = torch.permute(torch_in, tuple(perm))
                torch_out = torch.embedding(trans_torch_in, torch.from_numpy(id_np))
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_in], lr = 0.01)
                hetu_in = hetu.Tensor(x_np, requires_grad=True)
                trans_hetu_in = hetu_in.transpose(perm)
                hetu_out = hetu.embedding_lookup(trans_hetu_in, id)
                hetu_loss = hetu_out.sum()
                hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.01)
                torch_loss.backward()
                torch_optimizer.step()
                hetu_optimizer.minimize(hetu_loss).get_or_compute()
                self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)

    def test_onehotop(self):
        print(sys._getframe().f_code.co_name)
        for shape_x in TestOtherOps._onehot_test_shapes:
            x_np = np.random.randint(0, 16, size=shape_x)
            x = hetu.from_numpy(x_np)
            # transpose to non-contiguous input
            perm = list(range(len(shape_x)))
            perm.reverse()
            trans_x = x.transpose(perm)
            trans_x_np = x_np.transpose(tuple(perm))
            gt = torch.nn.functional.one_hot(torch.from_numpy(trans_x_np), num_classes = 16).numpy()
            self.assertTrue(allclose(hetu.onehot(trans_x, 16), gt))
            self.assertTrue(allclose(trans_x.onehot(16), gt))
        print(sys._getframe().f_code.co_name)

    def test_whereop(self):
        print(sys._getframe().f_code.co_name)
        for shape_x in TestOtherOps._onehot_test_shapes:
            reverse_shape_x = list(shape_x)
            reverse_shape_x.reverse()
            cond_np = np.random.choice([0, 1], size=reverse_shape_x).astype(np.int64)
            x_np = np.random.randn(*shape_x)
            y_np = np.random.randn(*shape_x)
            cond = hetu.from_numpy(cond_np)
            x = hetu.from_numpy(x_np)
            y = hetu.from_numpy(y_np)
            # transpose to non-contiguous input
            perm = list(range(len(shape_x)))
            perm.reverse()
            trans_x = x.transpose(perm)
            trans_x_np = x_np.transpose(tuple(perm))
            trans_y = y.transpose(perm)
            trans_y_np = y_np.transpose(tuple(perm))
            gt = np.where(cond_np, trans_x_np, trans_y_np) 
            self.assertTrue(allclose(hetu.where(cond, trans_x, trans_y), gt))
        print(sys._getframe().f_code.co_name)

if __name__ == "__main__":
    os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
    with hetu.graph("eager"):
        with hetu.context(eager_device="cuda:0"):
            unittest.main()
