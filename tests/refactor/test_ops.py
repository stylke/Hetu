import hetu
import numpy as np
import unittest
from test_utils import allclose

class TestArithmeticOps(unittest.TestCase):

    _test_elementwise_shapes = [
        (1024,), 
        (64, 256), 
        (64, 32, 16), 
    ]

    _test_broadcast_shapes = [
        ((1024,), (1,)), 
        ((1024,), (1024,)), 
        ((64, 256), (64, 1)), 
        ((64, 256), (1, 256)), 
        ((64, 256), (256,)), 
    ]

    def test_elementwise_add(self):
        for shape in TestArithmeticOps._test_elementwise_shapes:
            x_np = np.random.randn(*shape)
            y_np = np.random.randn(*shape)
            x = hetu.from_numpy(x_np)
            y = hetu.from_numpy(y_np)
            c = np.random.randn()
            # tensor + tensor
            gt = x_np + y_np
            self.assertTrue(allclose(x + y, gt))
            self.assertTrue(allclose(x.add(y), gt))
            self.assertTrue(allclose(hetu.add(x, y), gt))
            # tensor + constant & constant + tensor
            gt = x_np + c
            self.assertTrue(allclose(x + c, gt))
            self.assertTrue(allclose(c + x, gt))
            self.assertTrue(allclose(x.add(c), gt))
            self.assertTrue(allclose(hetu.add(x, c), gt))
            self.assertTrue(allclose(hetu.add(c, x), gt))
    
    def test_broadcast_add(self):
        for shape_x, shape_y in TestArithmeticOps._test_broadcast_shapes:
            x_np = np.random.randn(*shape_x)
            y_np = np.random.randn(*shape_y)
            x = hetu.from_numpy(x_np)
            y = hetu.from_numpy(y_np)
            gt = x_np + y_np
            self.assertTrue(allclose(x + y, gt))
            self.assertTrue(allclose(y + x, gt))
            self.assertTrue(allclose(x.add(y), gt))
            self.assertTrue(allclose(y.add(x), gt))
            self.assertTrue(allclose(hetu.add(x, y), gt))

    def test_elementwise_sub(self):
        for shape in TestArithmeticOps._test_elementwise_shapes:
            x_np = np.random.randn(*shape)
            y_np = np.random.randn(*shape)
            x = hetu.from_numpy(x_np)
            y = hetu.from_numpy(y_np)
            c = np.random.randn()
            # tensor - tensor
            gt = x_np - y_np
            self.assertTrue(allclose(x - y, gt))
            self.assertTrue(allclose(x.sub(y), gt))
            self.assertTrue(allclose(hetu.sub(x, y), gt))
            gt = y_np - x_np
            self.assertTrue(allclose(y - x, gt))
            self.assertTrue(allclose(y.sub(x), gt))
            self.assertTrue(allclose(hetu.sub(y, x), gt))
            # tensor - constant
            gt = x_np - c
            self.assertTrue(allclose(x - c, gt))
            self.assertTrue(allclose(x.sub(c), gt))
            self.assertTrue(allclose(hetu.sub(x, c), gt))
            # constant - tensor
            gt = c - x_np
            self.assertTrue(allclose(c - x, gt))
            self.assertTrue(allclose(hetu.sub(c, x), gt))
    
    def test_broadcast_sub(self):
        for shape_x, shape_y in TestArithmeticOps._test_broadcast_shapes:
            x_np = np.random.randn(*shape_x)
            y_np = np.random.randn(*shape_y)
            x = hetu.from_numpy(x_np)
            y = hetu.from_numpy(y_np)
            gt = x_np - y_np
            self.assertTrue(allclose(x - y, gt))
            self.assertTrue(allclose(x.sub(y), gt))
            self.assertTrue(allclose(hetu.sub(x, y), gt))
            gt = y_np - x_np
            self.assertTrue(allclose(y - x, gt))
            self.assertTrue(allclose(y.sub(x), gt))
            self.assertTrue(allclose(hetu.sub(y, x), gt))
    
    def test_neg(self):
        for shape in TestArithmeticOps._test_elementwise_shapes:
            x_np = np.random.randn(*shape)
            x = hetu.from_numpy(x_np)
            gt = np.negative(x_np)
            self.assertTrue(allclose(x.neg(), gt))
            self.assertTrue(allclose(hetu.neg(x), gt))

    def test_elementwise_mul(self):
        for shape in TestArithmeticOps._test_elementwise_shapes:
            x_np = np.random.randn(*shape)
            y_np = np.random.randn(*shape)
            x = hetu.from_numpy(x_np)
            y = hetu.from_numpy(y_np)
            c = np.random.randn()
            # tensor * tensor
            gt = x_np * y_np
            self.assertTrue(allclose(x * y, gt))
            self.assertTrue(allclose(y * x, gt))
            self.assertTrue(allclose(x.mul(y), gt))
            self.assertTrue(allclose(y.mul(x), gt))
            self.assertTrue(allclose(hetu.mul(x, y), gt))
            # tensor * constant & constant * tensor
            gt = x_np * c
            self.assertTrue(allclose(x * c, gt))
            self.assertTrue(allclose(c * x, gt))
            self.assertTrue(allclose(x.mul(c), gt))
            self.assertTrue(allclose(hetu.mul(x, c), gt))
            self.assertTrue(allclose(hetu.mul(c, x), gt))
    
    def test_broadcast_mul(self):
        for shape_x, shape_y in TestArithmeticOps._test_broadcast_shapes:
            x_np = np.random.randn(*shape_x)
            y_np = np.random.randn(*shape_y)
            x = hetu.from_numpy(x_np)
            y = hetu.from_numpy(y_np)
            gt = x_np * y_np
            self.assertTrue(allclose(x * y, gt))
            self.assertTrue(allclose(y * x, gt))
            self.assertTrue(allclose(x.mul(y), gt))
            self.assertTrue(allclose(y.mul(x), gt))
            self.assertTrue(allclose(hetu.mul(x, y), gt))
    
    def test_elementwise_div(self):
        for shape in TestArithmeticOps._test_elementwise_shapes:
            x_np = np.random.randn(*shape)
            y_np = np.random.randn(*shape)
            x = hetu.from_numpy(x_np)
            y = hetu.from_numpy(y_np)
            c = np.random.randn()
            # tensor / tensor
            gt = x_np / y_np
            self.assertTrue(allclose(x / y, gt))
            self.assertTrue(allclose(x.div(y), gt))
            self.assertTrue(allclose(hetu.div(x, y), gt))
            gt = y_np / x_np
            self.assertTrue(allclose(y / x, gt))
            self.assertTrue(allclose(y.div(x), gt))
            self.assertTrue(allclose(hetu.div(y, x), gt))
            # tensor - constant
            gt = x_np / c
            self.assertTrue(allclose(x / c, gt))
            self.assertTrue(allclose(x.div(c), gt))
            self.assertTrue(allclose(hetu.div(x, c), gt))
            # constant - tensor
            gt = c / x_np
            self.assertTrue(allclose(c / x, gt))
            self.assertTrue(allclose(hetu.div(c, x), gt))
    
    def test_broadcast_div(self):
        for shape_x, shape_y in TestArithmeticOps._test_broadcast_shapes:
            x_np = np.random.randn(*shape_x)
            y_np = np.random.randn(*shape_y)
            x = hetu.from_numpy(x_np)
            y = hetu.from_numpy(y_np)
            gt = x_np / y_np
            self.assertTrue(allclose(x / y, gt))
            self.assertTrue(allclose(x.div(y), gt))
            self.assertTrue(allclose(hetu.div(x, y), gt))
            gt = y_np / x_np
            self.assertTrue(allclose(y / x, gt))
            self.assertTrue(allclose(y.div(x), gt))
            self.assertTrue(allclose(hetu.div(y, x), gt))

    def test_reciprocal(self):
        for shape in TestArithmeticOps._test_elementwise_shapes:
            x_np = np.random.randn(*shape)
            x = hetu.from_numpy(x_np)
            gt = np.reciprocal(x_np)
            self.assertTrue(allclose(x.reciprocal(), gt))
            self.assertTrue(allclose(hetu.reciprocal(x), gt))

class TestMatMulOps(unittest.TestCase):

    _test_shapes = [
        ((64, 256), (256, 128))
    ]
    
    def test_matmul_op(self):
        for shape_x, shape_y in TestMatMulOps._test_shapes:
            x_np = np.random.randn(*shape_x)
            y_np = np.random.randn(*shape_y)
            gt = np.matmul(x_np, y_np)
            x = hetu.from_numpy(x_np)
            y = hetu.from_numpy(y_np)
            self.assertTrue(allclose(hetu.matmul(x, y), gt))
            self.assertTrue(allclose(x.matmul(y), gt))
    
    def test_linear_op(self):
        for shape_x, shape_y in TestMatMulOps._test_shapes:
            x_np = np.random.randn(*shape_x)
            w_np = np.random.randn(*shape_y[::-1])
            bias_np = np.random.randn(shape_y[-1])
            gt = np.matmul(x_np, w_np.transpose()) + bias_np
            x = hetu.from_numpy(x_np)
            w = hetu.from_numpy(w_np)
            bias = hetu.from_numpy(bias_np)
            self.assertTrue(allclose(hetu.linear(x, w, bias), gt))

class TestActivationOps(unittest.TestCase):

    _test_shapes = [
        (64, 256)
    ]

    def test_sigmoid_op(self):
        for shape in TestActivationOps._test_shapes:
            x_np = np.random.randn(*shape)
            gt = 1 / (1 + np.exp(-x_np))
            x = hetu.from_numpy(x_np)
            self.assertTrue(allclose(hetu.sigmoid(x), gt))
            self.assertTrue(allclose(x.sigmoid(), gt))
    
    def test_relu_op(self):
        for shape in TestActivationOps._test_shapes:
            x_np = np.random.randn(*shape)
            gt = x_np * (x_np > 0).astype(x_np.dtype)
            x = hetu.from_numpy(x_np)
            self.assertTrue(allclose(hetu.relu(x), gt))
            self.assertTrue(allclose(x.relu(), gt))

class TestLossOps(unittest.TestCase):
    _test_binary_label_shapes = [
        (64, 1)
    ]

    def test_bce_op(self):
        MIN_VALUE = -100.0
        for shape in TestLossOps._test_binary_label_shapes:
            probs_np = np.random.uniform(1e-10, 1, size=shape).astype(np.float32)
            labels_np = np.random.choice([0, 1], size=shape).astype(np.float32)
            t1_np = np.maximum(np.log(probs_np), MIN_VALUE)
            t2_np = np.maximum(np.log(1 - probs_np), MIN_VALUE)
            gt = -(labels_np * t1_np + (1 - labels_np) * t2_np)
            probs = hetu.from_numpy(probs_np)
            labels = hetu.from_numpy(labels_np)
            loss = hetu.binary_cross_entroy(probs, labels)
            self.assertTrue(allclose(loss, gt))

if __name__ == "__main__":
    unittest.main()
