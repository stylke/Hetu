import hetu
import hetu.nn as nn
import torch.optim as optim
import numpy as np
import torch
import unittest
from test_utils import allclose
import os
import sys

# Warning: Remember to set rtol = 1e-05, atol = 3e-05 in `test_utils.py`

GRAD_TEST = True

class TestCeilOps(unittest.TestCase):

    _test_shapes = [
        (2, 2),
        (64, 256),
        (1024, 16)
    ]

    def test_ceil_op(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestCeilOps._test_shapes:
            x_np = np.random.randn(*shape).astype(np.float32)
            gt = np.ceil(x_np)
            x = hetu.from_numpy(x_np)
            res_out = hetu.ceil(x)
            x.ceil_()
            self.assertTrue(allclose(res_out, gt))
            self.assertTrue(allclose(x, gt))

            if GRAD_TEST:
                torch_in = torch.tensor(x_np, requires_grad=True)
                torch_out = torch.ceil(torch_in)
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                hetu_in = hetu.Tensor(x_np, requires_grad=True)
                hetu_out = hetu.ceil(hetu_in)
                hetu_loss = hetu_out.sum()
                hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
                torch_loss.backward()
                torch_optimizer.step()
                hetu_optimizer.minimize(hetu_loss)
                self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))

                # in-place
                torch_in = torch.tensor(x_np, requires_grad=True)
                torch_out = torch.ceil(torch_in)
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                hetu_in = hetu.Tensor(x_np, requires_grad=True)
                hetu_out = hetu_in.add(0)
                # hetu_out.ceil_()
                hetu_out = hetu.ceil_(hetu_out)
                hetu_loss = hetu_out.sum()
                hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
                torch_loss.backward()
                torch_optimizer.step()
                hetu_optimizer.minimize(hetu_loss)
                self.assertTrue(allclose(hetu_loss, torch_loss.detach().numpy()))
                self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)

class TestFloorOps(unittest.TestCase):

    _test_shapes = [
        (2, 2),
        (64, 256),
        (1024, 16)
    ]

    def test_floor_op(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestFloorOps._test_shapes:
            x_np = np.random.randn(*shape).astype(np.float32)
            gt = np.floor(x_np)
            x = hetu.from_numpy(x_np)
            res_out = hetu.floor(x)
            x.floor_()
            self.assertTrue(allclose(res_out, gt))
            self.assertTrue(allclose(x, gt))

            if GRAD_TEST:
                torch_in = torch.tensor(x_np, requires_grad=True)
                torch_out = torch.floor(torch_in)
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                hetu_in = hetu.Tensor(x_np, requires_grad=True)
                hetu_out = hetu.floor(hetu_in)
                hetu_loss = hetu_out.sum()
                hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
                torch_loss.backward()
                torch_optimizer.step()
                hetu_optimizer.minimize(hetu_loss)
                self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))

                # in-place
                torch_in = torch.tensor(x_np, requires_grad=True)
                torch_out = torch.floor(torch_in)
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                hetu_in = hetu.Tensor(x_np, requires_grad=True)
                hetu_out = hetu_in.add(0)
                # hetu_out.floor_()
                hetu_out = hetu.floor_(hetu_out)
                hetu_loss = hetu_out.sum()
                hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
                torch_loss.backward()
                torch_optimizer.step()
                hetu_optimizer.minimize(hetu_loss)
                self.assertTrue(allclose(hetu_loss, torch_loss.detach().numpy()))
                self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)

class TestRoundOps(unittest.TestCase):

    _test_shapes = [
        (2, 2),
        (64, 256),
        (1024, 16)
    ]

    def test_round_op(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestRoundOps._test_shapes:
            x_np = np.random.randn(*shape).astype(np.float32)
            gt = np.round(x_np)
            x = hetu.from_numpy(x_np)
            res_out = hetu.round(x)
            x.round_()
            self.assertTrue(allclose(res_out, gt))
            self.assertTrue(allclose(x, gt))

            if GRAD_TEST:
                torch_in = torch.tensor(x_np, requires_grad=True)
                torch_out = torch.round(torch_in)
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                hetu_in = hetu.Tensor(x_np, requires_grad=True)
                hetu_out = hetu.round(hetu_in)
                hetu_loss = hetu_out.sum()
                hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
                torch_loss.backward()
                torch_optimizer.step()
                hetu_optimizer.minimize(hetu_loss)
                self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))

                # in-place
                torch_in = torch.tensor(x_np, requires_grad=True)
                torch_out = torch.round(torch_in)
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                hetu_in = hetu.Tensor(x_np, requires_grad=True)
                hetu_out = hetu_in.add(0)
                # hetu_out.round_()
                hetu_out = hetu.round_(hetu_out)
                hetu_loss = hetu_out.sum()
                hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
                torch_loss.backward()
                torch_optimizer.step()
                hetu_optimizer.minimize(hetu_loss)
                self.assertTrue(allclose(hetu_loss, torch_loss.detach().numpy()))
                self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)

class TestNegOps(unittest.TestCase):

    _test_shapes = [
        (2, 2),
        (64, 256),
        (1024, 16)
    ]

    def test_neg_op(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestNegOps._test_shapes:
            x_np = np.random.randn(*shape).astype(np.float32)
            gt = np.negative(x_np)
            x = hetu.from_numpy(x_np)
            res_out = hetu.neg(x)
            x.neg_()
            self.assertTrue(allclose(res_out, gt))
            self.assertTrue(allclose(x, gt))

            if GRAD_TEST:
                torch_in = torch.tensor(x_np, requires_grad=True)
                torch_out = torch.neg(torch_in)
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                hetu_in = hetu.Tensor(x_np, requires_grad=True)
                hetu_out = hetu.neg(hetu_in)
                hetu_loss = hetu_out.sum()
                hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
                torch_loss.backward()
                torch_optimizer.step()
                hetu_optimizer.minimize(hetu_loss)
                self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))

                # in-place
                torch_in = torch.tensor(x_np, requires_grad=True)
                torch_out = torch.neg(torch_in)
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                hetu_in = hetu.Tensor(x_np, requires_grad=True)
                hetu_out = hetu_in.add(0)
                # hetu_out.neg_()
                hetu_out = hetu.neg_(hetu_out)
                hetu_loss = hetu_out.sum()
                hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
                torch_loss.backward()
                torch_optimizer.step()
                hetu_optimizer.minimize(hetu_loss)
                self.assertTrue(allclose(hetu_loss, torch_loss.detach().numpy()))
                self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)

class TestPowOps(unittest.TestCase):

    _test_shapes = [
        (2, 2),
        (64, 256),
        (1024, 16)
    ]

    _exponent = [
        0.0,
        -1.0,
        -2.0, 
        4.0
    ]

    def test_pow_op(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestPowOps._test_shapes:
            for exponent in TestPowOps._exponent:
                x_np = np.random.randn(*shape).astype(np.float64)
                gt = np.power(x_np, exponent)
                x = hetu.from_numpy(x_np)
                res_out = hetu.pow(x, exponent)
                x.pow_(exponent)
                self.assertTrue(allclose(res_out, gt))
                self.assertTrue(allclose(x, gt))
        # in-place pow backward is not supported by Hetu
        print(sys._getframe().f_code.co_name)

class TestReciprocalOps(unittest.TestCase):

    _test_shapes = [
        (2, 2),
        (64, 256),
        (1024, 16)
    ]

    def test_reciprocal_op(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestReciprocalOps._test_shapes:
            x_np = np.random.randn(*shape).astype(np.float32)
            gt = np.reciprocal(x_np)
            x = hetu.from_numpy(x_np)
            res_out = hetu.reciprocal(x)
            x.reciprocal_()
            self.assertTrue(allclose(res_out, gt))
            self.assertTrue(allclose(x, gt))

            if GRAD_TEST:
                torch_in = torch.tensor(x_np, requires_grad=True)
                torch_out = torch.reciprocal(torch_in)
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                hetu_in = hetu.Tensor(x_np, requires_grad=True)
                hetu_out = hetu.reciprocal(hetu_in)
                hetu_loss = hetu_out.sum()
                hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
                torch_loss.backward()
                torch_optimizer.step()
                hetu_optimizer.minimize(hetu_loss)
                self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))

                # in-place
                torch_in = torch.tensor(x_np, requires_grad=True)
                torch_out = torch.reciprocal(torch_in)
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                hetu_in = hetu.Tensor(x_np, requires_grad=True)
                hetu_out = hetu_in.add(0)
                # hetu_out.reciprocal_()
                hetu_out = hetu.reciprocal_(hetu_out)
                hetu_loss = hetu_out.sum()
                hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
                torch_loss.backward()
                torch_optimizer.step()
                hetu_optimizer.minimize(hetu_loss)
                self.assertTrue(allclose(hetu_loss, torch_loss.detach().numpy()))
                self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)

class TestReluOps(unittest.TestCase):

    _test_shapes = [
        (2, 2),
        (64, 256),
        (1024, 16)
    ]

    def test_relu_op(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestReluOps._test_shapes:
            x_np = np.random.randn(*shape).astype(np.float32)
            gt = x_np * (x_np > 0).astype(x_np.dtype)
            x = hetu.from_numpy(x_np)
            res_out = hetu.relu(x)
            x.relu_()
            self.assertTrue(allclose(res_out, gt))
            self.assertTrue(allclose(x, gt))

            if GRAD_TEST:
                torch_in = torch.tensor(x_np, requires_grad=True)
                torch_out = torch.relu(torch_in)
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                hetu_in = hetu.Tensor(x_np, requires_grad=True)
                hetu_out = hetu.relu(hetu_in)
                hetu_loss = hetu_out.sum()
                hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
                torch_loss.backward()
                torch_optimizer.step()
                hetu_optimizer.minimize(hetu_loss)
                self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))

                # in-place
                torch_in = torch.tensor(x_np, requires_grad=True)
                torch_out = torch.relu(torch_in)
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                hetu_in = hetu.Tensor(x_np, requires_grad=True)
                hetu_out = hetu_in.add(0)
                # hetu_out.relu_()
                hetu_out = hetu.relu_(hetu_out)
                hetu_loss = hetu_out.sum()
                hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
                torch_loss.backward()
                torch_optimizer.step()
                hetu_optimizer.minimize(hetu_loss)
                self.assertTrue(allclose(hetu_loss, torch_loss.detach().numpy()))
                self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)

class TestTanhOps(unittest.TestCase):

    _test_shapes = [
        (2, 2),
        (64, 256),
        (1024, 16)
    ]

    def test_tanh_op(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestTanhOps._test_shapes:
            x_np = np.random.randn(*shape).astype(np.float32)
            gt = np.tanh(x_np)
            x = hetu.from_numpy(x_np)
            res_out = hetu.tanh(x)
            x.tanh_()
            self.assertTrue(allclose(res_out, gt))
            self.assertTrue(allclose(x, gt))

            if GRAD_TEST:
                torch_in = torch.tensor(x_np, requires_grad=True)
                torch_out = torch.tanh(torch_in)
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                hetu_in = hetu.Tensor(x_np, requires_grad=True)
                hetu_out = hetu.tanh(hetu_in)
                hetu_loss = hetu_out.sum()
                hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
                torch_loss.backward()
                torch_optimizer.step()
                hetu_optimizer.minimize(hetu_loss)
                self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))

                # in-place
                torch_in = torch.tensor(x_np, requires_grad=True)
                torch_out = torch.tanh(torch_in)
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                hetu_in = hetu.Tensor(x_np, requires_grad=True)
                hetu_out = hetu_in.add(0)
                # hetu_out.tanh_()
                hetu_out = hetu.tanh_(hetu_out)
                hetu_loss = hetu_out.sum()
                hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
                torch_loss.backward()
                torch_optimizer.step()
                hetu_optimizer.minimize(hetu_loss)
                self.assertTrue(allclose(hetu_loss, torch_loss.detach().numpy()))
                self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)

class TestLeakyReluOps(unittest.TestCase):

    _test_shapes = [
        (2, 2),
        (64, 256),
        (1024, 16)
    ]

    def test_leaky_relu_op(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestLeakyReluOps._test_shapes:
            x_np = np.random.randn(*shape).astype(np.float32)
            alphas = [0.1, 0.2, 0.5]
            for alpha in alphas:
                gt = np.where(x_np > 0, x_np, alpha * x_np)
                x = hetu.from_numpy(x_np)
                res_out = hetu.leakyrelu(x, alpha)
                x.leakyrelu_(alpha)
                self.assertTrue(allclose(res_out, gt))
                self.assertTrue(allclose(x, gt))

                if GRAD_TEST:
                    torch_in = torch.tensor(x_np, requires_grad=True)
                    torch_out = torch.nn.functional.leaky_relu(torch_in, alpha)
                    torch_loss = torch_out.sum()
                    torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                    hetu_in = hetu.Tensor(x_np, requires_grad=True)
                    hetu_out = hetu.leakyrelu(hetu_in, alpha)
                    hetu_loss = hetu_out.sum()
                    hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
                    torch_loss.backward()
                    torch_optimizer.step()
                    hetu_optimizer.minimize(hetu_loss)
                    self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))

                    # in-place
                    torch_in = torch.tensor(x_np, requires_grad=True)
                    torch_out = torch.nn.functional.leaky_relu(torch_in, alpha)
                    torch_loss = torch_out.sum()
                    torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                    hetu_in = hetu.Tensor(x_np, requires_grad=True)
                    hetu_out = hetu_in.add(0)
                    # hetu_out.leakyrelu_(alpha)
                    hetu_out = hetu.leakyrelu_(hetu_out, alpha)
                    hetu_loss = hetu_out.sum()
                    hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
                    torch_loss.backward()
                    torch_optimizer.step()
                    hetu_optimizer.minimize(hetu_loss)
                    self.assertTrue(allclose(hetu_loss, torch_loss.detach().numpy()))
                    self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)

class TestSigmoidOps(unittest.TestCase):

    _test_shapes = [
        (2, 2),
        (64, 256),
        (1024, 16)
    ]

    def test_sigmoid_op(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestSigmoidOps._test_shapes:
            x_np = np.random.randn(*shape).astype(np.float32)
            gt = 1 / (1 + np.exp(-x_np))
            x = hetu.from_numpy(x_np)
            res_out = hetu.sigmoid(x)
            x.sigmoid_()
            self.assertTrue(allclose(res_out, gt))
            self.assertTrue(allclose(x, gt))

            if GRAD_TEST:
                torch_in = torch.tensor(x_np, requires_grad=True)
                torch_out = torch.sigmoid(torch_in)
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                hetu_in = hetu.Tensor(x_np, requires_grad=True)
                hetu_out = hetu.sigmoid(hetu_in)
                hetu_loss = hetu_out.sum()
                hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
                torch_loss.backward()
                torch_optimizer.step()
                hetu_optimizer.minimize(hetu_loss)
                self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))

                # in-place
                torch_in = torch.tensor(x_np, requires_grad=True)
                torch_out = torch.sigmoid(torch_in)
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                hetu_in = hetu.Tensor(x_np, requires_grad=True)
                hetu_out = hetu_in.add(0)
                # hetu_out.sigmoid_()
                hetu_out = hetu.sigmoid_(hetu_out)
                hetu_loss = hetu_out.sum()
                hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
                torch_loss.backward()
                torch_optimizer.step()
                hetu_optimizer.minimize(hetu_loss)
                self.assertTrue(allclose(hetu_loss, torch_loss.detach().numpy()))
                self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)

class TestSqrtOps(unittest.TestCase):

    _test_shapes = [
        (2, 2),
        (64, 256),
        (1024, 16)
    ]

    def test_sqrt_op(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestSqrtOps._test_shapes:
            x_np = np.abs(np.random.randn(*shape)).astype(np.float32)
            gt = np.sqrt(x_np)
            x = hetu.from_numpy(x_np)
            res_out = hetu.sqrt(x)
            x.sqrt_()
            self.assertTrue(allclose(res_out, gt))
            self.assertTrue(allclose(x, gt))

            if GRAD_TEST:
                torch_in = torch.tensor(x_np, requires_grad=True)
                torch_out = torch.sqrt(torch_in)
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                hetu_in = hetu.Tensor(x_np, requires_grad=True)
                hetu_out = hetu.sqrt(hetu_in)
                hetu_loss = hetu_out.sum()
                hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
                torch_loss.backward()
                torch_optimizer.step()
                hetu_optimizer.minimize(hetu_loss)
                self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))

                # in-place
                torch_in = torch.tensor(x_np, requires_grad=True)
                torch_out = torch.sqrt(torch_in)
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                hetu_in = hetu.Tensor(x_np, requires_grad=True)
                hetu_out = hetu_in.add(0)
                # hetu_out.sqrt_()
                hetu_out = hetu.sqrt_(hetu_out)
                hetu_loss = hetu_out.sum()
                hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
                torch_loss.backward()
                torch_optimizer.step()
                hetu_optimizer.minimize(hetu_loss)
                self.assertTrue(allclose(hetu_loss, torch_loss.detach().numpy()))
                self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)

class TestRSqrtOps(unittest.TestCase):

    _test_shapes = [
        (2, 2),
        (64, 256),
        (1024, 16)
    ]

    def test_rsqrt_op(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestRSqrtOps._test_shapes:
            x_np = np.abs(np.random.randn(*shape)).astype(np.float32)
            gt = np.reciprocal(np.sqrt(x_np))
            x = hetu.from_numpy(x_np)
            res_out = hetu.rsqrt(x)
            x.rsqrt_()
            self.assertTrue(allclose(res_out, gt))
            self.assertTrue(allclose(x, gt))

            if GRAD_TEST:
                torch_in = torch.tensor(x_np, requires_grad=True)
                torch_out = torch.rsqrt(torch_in)
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                hetu_in = hetu.Tensor(x_np, requires_grad=True)
                hetu_out = hetu.rsqrt(hetu_in)
                hetu_loss = hetu_out.sum()
                hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
                torch_loss.backward()
                torch_optimizer.step()
                hetu_optimizer.minimize(hetu_loss)
                self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))

                # in-place
                torch_in = torch.tensor(x_np, requires_grad=True)
                torch_out = torch.rsqrt(torch_in)
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                hetu_in = hetu.Tensor(x_np, requires_grad=True)
                hetu_out = hetu_in.add(0)
                # hetu_out.rsqrt_()
                hetu_out = hetu.rsqrt_(hetu_out)
                hetu_loss = hetu_out.sum()
                hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
                torch_loss.backward()
                torch_optimizer.step()
                hetu_optimizer.minimize(hetu_loss)
                self.assertTrue(allclose(hetu_loss, torch_loss.detach().numpy()))
                self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)

class TestWhereOps(unittest.TestCase):

    _test_shapes = [
        (2, 2),
        (64, 256),
        (1024, 16)
    ]

    def test_where_op(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestWhereOps._test_shapes:
            cond_np = np.random.choice([0, 1], size=shape).astype(np.int64)
            x_np = np.random.randn(*shape)
            y_np = np.random.randn(*shape)
            gt = np.where(cond_np, x_np, y_np)
            cond = hetu.from_numpy(cond_np)
            x = hetu.from_numpy(x_np)
            y = hetu.from_numpy(y_np)
            res_out = hetu.where(cond, x, y)
            hetu.where_(cond, x, y)
            self.assertTrue(allclose(res_out, gt))
            self.assertTrue(allclose(x, gt))

            if GRAD_TEST:
                torch_cond = torch.tensor(cond_np, dtype=torch.bool)
                torch_x = torch.tensor(x_np, requires_grad=True)
                torch_y = torch.tensor(y_np)
                torch_out = torch.where(torch_cond, torch_x, torch_y)
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_x], lr = 0.5)
                hetu_cond = hetu.Tensor(cond_np)
                hetu_x = hetu.Tensor(x_np, requires_grad=True)
                hetu_y = hetu.Tensor(y_np)
                hetu_out = hetu.where(hetu_cond, hetu_x, hetu_y)
                hetu_loss = hetu_out.sum()
                hetu_optimizer = hetu.SGDOptimizer([hetu_x], lr = 0.5)
                torch_loss.backward()
                torch_optimizer.step()
                hetu_optimizer.minimize(hetu_loss)
                self.assertTrue(allclose(hetu_x, torch_x.detach().numpy()))

                # in-place
                torch_cond = torch.tensor(cond_np, dtype=torch.bool)
                torch_x = torch.tensor(x_np, requires_grad=True)
                torch_y = torch.tensor(y_np)
                torch_out = torch.where(torch_cond, torch_x, torch_y)
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_x], lr = 0.5)
                hetu_cond = hetu.Tensor(cond_np)
                hetu_x = hetu.Tensor(x_np, requires_grad=True)
                hetu_y = hetu.Tensor(y_np)
                hetu_out = hetu_x.add(0)
                # hetu_out.where_(hetu_cond, hetu_y)
                hetu_out = hetu.where_(hetu_cond, hetu_out, hetu_y)
                hetu_loss = hetu_out.sum()
                hetu_optimizer = hetu.SGDOptimizer([hetu_x], lr = 0.5)
                torch_loss.backward()
                torch_optimizer.step()
                hetu_optimizer.minimize(hetu_loss)
                self.assertTrue(allclose(hetu_loss, torch_loss.detach().numpy()))
                self.assertTrue(allclose(hetu_x, torch_x.detach().numpy()))
        print(sys._getframe().f_code.co_name)

if __name__ == "__main__":
    os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
    with hetu.graph("eager"):
        with hetu.context(eager_device="cuda:0"):
            unittest.main()
