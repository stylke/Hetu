import hetu
import hetu.nn as nn
import numpy as np
import torch
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

    def test_sqrt(self):
        for shape in TestArithmeticOps._test_elementwise_shapes:
            x_np = np.abs(np.random.randn(*shape))
            x = hetu.from_numpy(x_np)
            gt = np.sqrt(x_np)
            self.assertTrue(allclose(x.sqrt(), gt))
            self.assertTrue(allclose(hetu.sqrt(x), gt))

    def test_sum(self):
        for shape in TestArithmeticOps._test_elementwise_shapes:
            x_np = np.random.randn(*shape)
            y_np = np.random.randn(*shape)
            z_np = np.random.randn(*shape)
            x = hetu.from_numpy(x_np)
            y = hetu.from_numpy(y_np)
            z = hetu.from_numpy(z_np)
            gt = x_np + y_np + z_np
            self.assertTrue(allclose(hetu.sum([x,y,z]), gt))

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

class TestBatchMatMulOps(unittest.TestCase):

    _test_shapes = [
        ((1, 64, 128), (1, 128, 32)),
        ((16, 64, 256), (16, 256, 128))
    ]
    
    def test_batch_matmul_op(self):
        for shape_x, shape_y in TestBatchMatMulOps._test_shapes:
            x_np = np.random.randn(*shape_x)
            y_np = np.random.randn(*shape_y)
            gt = torch.bmm(torch.from_numpy(x_np), torch.from_numpy(y_np)).numpy()
            x = hetu.from_numpy(x_np)
            y = hetu.from_numpy(y_np)
            self.assertTrue(allclose(hetu.bmm(x, y), gt))
            self.assertTrue(allclose(x.bmm(y), gt))

# class TestMatDotOps(unittest.TestCase):

#     _test_shapes = [
#         ((128, 64), (128, 64)),
#         ((256, 64), (256, 16))
#     ]
    
#     def test_batch_matmul_op(self):
#         for shape_x, shape_y in TestMatDotOps._test_shapes:
#             x_np = np.random.randn(*shape_x)
#             y_np = np.random.randn(*shape_y)
#             x = hetu.from_numpy(x_np)
#             y = hetu.from_numpy(y_np)
#             gt = np.dot(x_np,y_np)
#             self.assertTrue(allclose(hetu.dot(x, y), gt))
#             self.assertTrue(allclose(x.dot(y), gt))
    

class TestActivationOps(unittest.TestCase):

    _test_shapes = [
        (64, 256),
        (1024, 16)
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
            
    
    def test_leaky_relu_op(self):
        for shape in TestActivationOps._test_shapes:
            x_np = np.random.randn(*shape)
            alphas = [0.1, 0.2, 0.5]
            for alpha in alphas:
                gt = np.where(x_np > 0, x_np, alpha * x_np)
                x = hetu.from_numpy(x_np)
                self.assertTrue(allclose(hetu.leakyrelu(x, alpha), gt))
                self.assertTrue(allclose(x.leakyrelu(alpha), gt))

    def test_tanh_op(self):
        for shape in TestActivationOps._test_shapes:
            x_np = np.random.randn(*shape)
            gt = np.tanh(x_np)
            x = hetu.from_numpy(x_np)
            self.assertTrue(allclose(hetu.tanh(x), gt))
            self.assertTrue(allclose(x.tanh(), gt))
    
    def test_softmax_op(self):
        for shape in TestActivationOps._test_shapes:
            x_np = np.random.randn(*shape)
            gt = torch.softmax(torch.from_numpy(x_np), 1).numpy()
            x = hetu.from_numpy(x_np)
            self.assertTrue(allclose(hetu.softmax(x), gt))
            self.assertTrue(allclose(x.softmax(), gt))


class TestTransformOps(unittest.TestCase):

    _test_shapes = [
        (64, 256),
        (128, 128)
    ]

    _pad_shapes = [
        (8, 4, 32, 32),
        (16, 4, 16, 16)
    ]

    _transpose_shapes = [
        (16, 4, 16),
        (4, 8, 16, 32)
    ]

    def test_reshape_op(self):
        for shape in TestTransformOps._test_shapes:
            x_np = np.random.randn(*shape)
            shape_to = list(shape)
            shape_to[0] = int(shape_to[0] / 2)
            shape_to[1] *= 2
            gt = np.reshape(x_np, tuple(shape_to))
            x = hetu.from_numpy(x_np)
            self.assertTrue(allclose(hetu.reshape(x, shape_to), gt))
            self.assertTrue(allclose(x.reshape(shape_to), gt))

    def test_broadcast_op(self):
        for shape in TestTransformOps._test_shapes:
            x_np = np.random.randn(*shape)
            shape_to = list(shape)
            shape_to = [16] + shape_to
            gt = np.broadcast_to(x_np, tuple(shape_to))
            x = hetu.from_numpy(x_np)
            self.assertTrue(allclose(hetu.broadcast(x, shape_to, []), gt))
            self.assertTrue(allclose(x.broadcast(shape_to, []), gt))

    def test_concat_op(self):
        for shape in TestTransformOps._test_shapes:
            x_np = np.random.randn(*shape)
            y_np = np.random.randn(*shape)
            z_np = np.random.randn(*shape)
            gt = np.concatenate((x_np, y_np), 0)
            x = hetu.from_numpy(x_np)
            y = hetu.from_numpy(y_np)
            z = hetu.from_numpy(z_np)
            self.assertTrue(allclose(hetu.concat(x, y, 0), gt))
            self.assertTrue(allclose(x.concat(y, 0), gt))
            self.assertTrue(allclose(hetu.concat([x, y], 0), gt))
            gt = np.concatenate((x_np, y_np, z_np), 0)
            self.assertTrue(allclose(hetu.concat([x, y, z], 0), gt))
    
    def test_pad_op(self):
        for shape in TestTransformOps._pad_shapes:
            x_np = np.random.randn(*shape)
            x = hetu.from_numpy(x_np)
            gt = np.pad(x_np, ((0,0),(0,0),(1,1),(2,2)), "constant", constant_values = 0.1)
            self.assertTrue(allclose(hetu.pad(x, [1,1,2,2], "constant", 0.1), gt))
            self.assertTrue(allclose(x.pad([1,1,2,2], "constant", 0.1), gt))

    def test_slice_op(self):
        for shape in TestTransformOps._test_shapes:
            x_np = np.random.randn(*shape)
            begin_pos = list(np.random.randint(0, 16 ,size = [2]))
            out_size = list(np.random.randint(16, 32 ,size = [2]))
            gt = x_np[begin_pos[0]:begin_pos[0]+out_size[0], begin_pos[1]:begin_pos[1]+out_size[1]]
            x = hetu.from_numpy(x_np)
            self.assertTrue(allclose(hetu.slice(x, begin_pos, out_size), gt))
            self.assertTrue(allclose(x.slice(begin_pos, out_size), gt))

    def test_split_op(self):
        for shape in TestTransformOps._test_shapes:
            x_np = np.random.randn(*shape)
            idx = list(np.random.randint(0, 8 ,size = [1]))
            gt = np.split(x_np, 8, 0)[idx[0]]
            x = hetu.from_numpy(x_np)
            self.assertTrue(allclose(hetu.split(x, [0], idx, [8]), gt))
            self.assertTrue(allclose(x.split([0], idx, [8]), gt))
    
    def test_transpose_op(self):
        for shape in TestTransformOps._test_shapes:
            x_np = np.random.randn(*shape)
            gt = np.transpose(x_np, (1, 0))
            x = hetu.from_numpy(x_np)
            self.assertTrue(allclose(hetu.transpose(x, [1, 0]), gt))
            self.assertTrue(allclose(x.transpose([1, 0]), gt))

        for shape in TestTransformOps._transpose_shapes:
            x_np = np.random.randn(*shape)
            perm = np.arange(x_np.ndim)
            np.random.shuffle(perm)
            perm = list(perm)
            gt = np.transpose(x_np, perm)
            x = hetu.from_numpy(x_np)
            self.assertTrue(allclose(hetu.transpose(x, perm), gt))
            self.assertTrue(allclose(x.transpose(perm), gt))

class TestConv2dOps(unittest.TestCase):

    _data_shapes = [
        (4, 3, 16, 16),        
    ]

    _filter_shapes = [
        (3, 3, 2, 2),
        (4, 3, 4, 4)        
    ]

    def test_conv2d_op(self):
        for shape in TestConv2dOps._data_shapes:
            x_np = np.random.randn(*shape)
            x = hetu.from_numpy(x_np)
            for f_shape in TestConv2dOps._filter_shapes:
                f_np = np.random.randn(*f_shape)
                f = hetu.from_numpy(f_np)
                bias_np = np.random.randn()
                gt = torch.conv2d(torch.from_numpy(x_np), torch.from_numpy(f_np), stride = 1, padding = 0).numpy()
                bias_shape = [f_shape[0]]
                self.assertTrue(allclose(hetu.conv2d(x, f, 0, 1), gt))
                self.assertTrue(allclose(x.conv2d(f, 0, 1), gt))
                # test conv2d add bias
                bias_np = np.random.randn(*bias_shape)
                bias = hetu.from_numpy(bias_np)
                gt = torch.conv2d(torch.from_numpy(x_np), torch.from_numpy(f_np), torch.from_numpy(bias_np), stride = 1, padding = 0).numpy()
                self.assertTrue(allclose(hetu.conv2d(x, f, bias, 0, 1), gt))
                self.assertTrue(allclose(x.conv2d(f, bias, 0, 1), gt))


class TestPoolOps(unittest.TestCase):

    _test_shapes = [
        (4, 3, 16, 16),      
        (5, 8, 16, 16)  
    ]


    def test_maxpool_op(self):
        for shape in TestPoolOps._test_shapes:
            x_np = np.random.randn(*shape)
            x = hetu.from_numpy(x_np)
            maxpool2d = torch.nn.MaxPool2d(2, 1, 0)
            gt = maxpool2d(torch.from_numpy(x_np)).numpy()
            self.assertTrue(allclose(hetu.maxpool(x, 2, 2, 0, 1), gt))
            self.assertTrue(allclose(x.maxpool(2, 2, 0, 1), gt))

    def test_avgpool_op(self):
        for shape in TestPoolOps._test_shapes:
            x_np = np.random.randn(*shape)
            x = hetu.from_numpy(x_np)
            avgpool2d = torch.nn.AvgPool2d(2, 1, 0)
            gt = avgpool2d(torch.from_numpy(x_np)).numpy()
            self.assertTrue(allclose(hetu.avgpool(x, 2, 2, 0, 1), gt))
            self.assertTrue(allclose(x.avgpool(2, 2, 0, 1), gt))

class TestNormOps(unittest.TestCase):

    _test_shapes = [
        (4, 3, 16, 16),      
        (5, 8, 16, 16)  
    ]


    def test_batchnorm_op(self):
        for shape in TestPoolOps._test_shapes:
            x_np = np.random.randn(*shape).astype(np.float32)
            x = hetu.from_numpy(x_np)
            scale_np = np.ones(shape[1]).astype(np.float32)
            scale = hetu.from_numpy(scale_np)
            bias_np = np.zeros(shape[1]).astype(np.float32)
            bias = hetu.from_numpy(bias_np)
            gt = torch.batch_norm(torch.from_numpy(x_np), weight = torch.from_numpy(scale_np), bias = torch.from_numpy(bias_np),
                                 running_mean=None, running_var=None, training=True, momentum=0.1, eps=1e-5, cudnn_enabled=True).numpy()
            self.assertTrue(allclose(hetu.batch_norm(x, scale, bias, 0.1 ,1e-5), gt))
            self.assertTrue(allclose(x.batch_norm(scale, bias, 0.1 ,1e-5), gt))

    def test_layernorm_op(self):
        for shape in TestPoolOps._test_shapes:
            norm_shape = shape[3:]
            x_np = np.random.randn(*shape).astype(np.float32)
            x = hetu.from_numpy(x_np)
            scale_np = np.ones(norm_shape).astype(np.float32)
            scale = hetu.from_numpy(scale_np)
            bias_np = np.zeros(norm_shape).astype(np.float32)
            bias = hetu.from_numpy(bias_np)
            layernorm = torch.nn.LayerNorm(norm_shape, 1e-5)
            gt = layernorm(torch.from_numpy(x_np)).detach().numpy()
            gt2 = torch.layer_norm(torch.from_numpy(x_np), normalized_shape=tuple(norm_shape), weight = torch.from_numpy(scale_np), bias = torch.from_numpy(bias_np),
                                  eps=1e-5).numpy()
            self.assertTrue(allclose(gt2, gt))
            self.assertTrue(allclose(hetu.layer_norm(x, scale, bias, 1e-5), gt))
            self.assertTrue(allclose(x.layer_norm(scale, bias, 1e-5), gt))
    
    def test_instancenorm_op(self):
        for shape in TestPoolOps._test_shapes:
            x_np = np.random.randn(*shape).astype(np.float32)
            x = hetu.from_numpy(x_np)
            instancenorm = torch.nn.InstanceNorm2d(num_features=shape[1], eps=1e-5)
            gt = instancenorm(torch.from_numpy(x_np)).detach().numpy()
            self.assertTrue(allclose(hetu.instance_norm(x, 1e-5), gt))
            self.assertTrue(allclose(x.instance_norm(1e-5), gt))

class TestReduceOps(unittest.TestCase):

    _test_shapes = [
        (16, 4, 16, 16),
        (1, 8, 32, 32)
    ]
    
    def test_reduce_sum_op(self):
        for shape_x in TestReduceOps._test_shapes:
            x_np = np.random.randn(*shape_x)
            gt = np.sum(x_np)
            x = hetu.from_numpy(x_np)
            self.assertTrue(allclose(hetu.reduce(x, "sum"), gt))
            self.assertTrue(allclose(x.reduce("sum"), gt))
            self.assertTrue(allclose(hetu.sum(x), gt))
            self.assertTrue(allclose(x.sum(), gt))
            for i in range(1, pow(2, len(shape_x))):
                tmp = i
                ins = 0
                axes = []
                while tmp > 0:
                    if (tmp % 2 == 1):
                        axes.append(ins)
                    tmp //= 2
                    ins += 1
                gt = np.sum(x_np, tuple(axes))
                x = hetu.from_numpy(x_np)
                self.assertTrue(allclose(hetu.sum(x, axes), gt))
                self.assertTrue(allclose(x.sum(axes), gt))
                #keepdim test
                gt = np.sum(x_np, tuple(axes), keepdims=True)
                x = hetu.from_numpy(x_np)
                self.assertTrue(allclose(hetu.sum(x, axes, [True]), gt))
                

    def test_reduce_mean_op(self):
        for shape_x in TestReduceOps._test_shapes:
            x_np = np.random.randn(*shape_x)
            gt = np.average(x_np)
            x = hetu.from_numpy(x_np)
            self.assertTrue(allclose(hetu.reduce(x, "mean"), gt))
            self.assertTrue(allclose(x.reduce("mean"), gt))
            self.assertTrue(allclose(hetu.mean(x), gt))
            self.assertTrue(allclose(x.mean(), gt))
            for i in range(1, pow(2, len(shape_x))):
                tmp = i
                ins = 0
                axes = []
                while tmp > 0:
                    if (tmp % 2 == 1):
                        axes.append(ins)
                    tmp //= 2
                    ins += 1
                gt = np.average(x_np, tuple(axes))
                x = hetu.from_numpy(x_np)
                self.assertTrue(allclose(hetu.mean(x, axes), gt))
                self.assertTrue(allclose(x.mean(axes), gt))
                #keepdim test
                gt = np.mean(x_np, tuple(axes), keepdims=True)
                x = hetu.from_numpy(x_np)
                self.assertTrue(allclose(hetu.mean(x, axes, [True]), gt))

class TestLossOps(unittest.TestCase):
    _test_binary_label_shapes = [
        (64, 1)
    ]

    _test_nllloss_label_shapes = [
        ((64, 16), (64, ))
    ]

    _test_cross_entropy_label_shapes = [
        (64, 16)
    ]

    def test_bce_op(self):
        MIN_VALUE = -100.0
        for shape in TestLossOps._test_binary_label_shapes:
            probs_np = np.random.uniform(1e-10, 1, size=shape).astype(np.float32)
            labels_np = np.random.choice([0, 1], size=shape).astype(np.float32)
            # t1_np = np.maximum(np.log(probs_np), MIN_VALUE)
            # t2_np = np.maximum(np.log(1 - probs_np), MIN_VALUE)
            # gt = -(labels_np * t1_np + (1 - labels_np) * t2_np)
            gt = torch.nn.functional.binary_cross_entropy(torch.from_numpy(probs_np), torch.from_numpy(labels_np)).numpy()
            probs = hetu.from_numpy(probs_np)
            labels = hetu.from_numpy(labels_np)
            loss = hetu.binary_cross_entropy(probs, labels)
            self.assertTrue(allclose(loss, gt))
    
    def test_nllloss_op(self):
        for shape, lshape in TestLossOps._test_nllloss_label_shapes:
            probs_np = np.random.uniform(1e-10, 1, size=shape).astype(np.float32)
            labels_np = np.random.choice(range(16), size=lshape).astype(np.int64)
            # t1_np = np.maximum(np.log(probs_np), MIN_VALUE)
            # t2_np = np.maximum(np.log(1 - probs_np), MIN_VALUE)
            # gt = -(labels_np * t1_np + (1 - labels_np) * t2_np)
            gt = torch.nn.functional.nll_loss(torch.from_numpy(probs_np), torch.from_numpy(labels_np)).numpy()
            #gt = torch.nn.functional.nll_loss(torch.from_numpy(probs_np), torch.from_numpy(labels_np)).numpy()
            probs = hetu.from_numpy(probs_np)
            labels = hetu.from_numpy(labels_np)
            loss = hetu.nll_loss(probs, labels)
            self.assertTrue(allclose(loss, gt))
    
    def test_kldivloss_op(self):
        for shape in TestLossOps._test_binary_label_shapes:
            probs_np = np.random.uniform(1e-10, 1, size=shape).astype(np.float32)
            labels_np = np.random.choice([0, 1], size=shape).astype(np.float32)
            # t1_np = np.maximum(np.log(probs_np), MIN_VALUE)
            # t2_np = np.maximum(np.log(1 - probs_np), MIN_VALUE)
            # gt = -(labels_np * t1_np + (1 - labels_np) * t2_np)
            gt = torch.nn.functional.kl_div(torch.from_numpy(probs_np), torch.from_numpy(labels_np)).numpy()
            probs = hetu.from_numpy(probs_np)
            labels = hetu.from_numpy(labels_np)
            loss = hetu.kl_div(probs, labels)
            self.assertTrue(allclose(loss, gt))
    
    def test_mseloss_op(self):
        MIN_VALUE = -100.0
        for shape in TestLossOps._test_binary_label_shapes:
            probs_np = np.random.uniform(1e-10, 1, size=shape).astype(np.float32)
            labels_np = np.random.choice([0, 1], size=shape).astype(np.float32)
            # t1_np = np.maximum(np.log(probs_np), MIN_VALUE)
            # t2_np = np.maximum(np.log(1 - probs_np), MIN_VALUE)
            # gt = -(labels_np * t1_np + (1 - labels_np) * t2_np)
            gt = torch.nn.functional.mse_loss(torch.from_numpy(probs_np), torch.from_numpy(labels_np)).numpy()
            probs = hetu.from_numpy(probs_np)
            labels = hetu.from_numpy(labels_np)
            loss = hetu.mse_loss(probs, labels)
            self.assertTrue(allclose(loss, gt))

    # def test_softmax_cross_entropy_op(self):
    #     MIN_VALUE = -100.0
    #     for shape in TestLossOps._test_cross_entropy_label_shapes:
    #         probs_np = np.random.uniform(1e-10, 1, size=shape).astype(np.float32)
    #         labels_np = np.random.uniform(0.25, 0.5, size=shape).astype(np.float32)
    #         # probs_np = np.arange(4).astype(np.float32) + 1
    #         # probs_np = probs_np.reshape(2,2)
    #         # labels_np = np.array([[1,0],[0,1]]).astype(np.float32).reshape(2,2)
    #         crs_etp = torch.nn.CrossEntropyLoss()
    #         gt = crs_etp(torch.from_numpy(probs_np), torch.from_numpy(labels_np)).numpy()
    #         # gt = torch.nn.functional.cross_entropy(torch.from_numpy(probs_np), torch.from_numpy(labels_np)).numpy()
    #         probs = hetu.from_numpy(probs_np)
    #         labels = hetu.from_numpy(labels_np)
    #         loss = hetu.softmax_cross_entroy(probs, labels)
    #         self.assertTrue(allclose(loss, gt))

class TestEinsumOps(unittest.TestCase):

    _test_args = [
        ("ij->ji",((64, 32),)),
        ("ij,ij->ij", ((64, 32), (64, 32))),
        ("ii->i",((64, 64),)),
        ("...ij->...ji",((64, 32, 4, 2, 4),)),
        ("ij->",((64, 32),)),
        ("ij->j",((64, 32),)),
        ("ik,k",((64, 32),(32,))),
        ("ik,kj",((64, 32),(32, 16))),
        ("i,i",((2,),(2,))),
        ("ij,ij",((64, 32),(64, 32))),
        ("i,j",((64, ),(32, ))),
        ("ijk,ikl->ijl",((64, 32, 16), (64, 16, 24))),
        ("pqrs,tuqvr->pstuv", ((4, 5, 6, 8), (9, 7, 5, 13, 6))),
        ("ik,jkl,il->ij",((64, 32), (16, 32, 48), (64, 48))),
        ("ijk",((64, 32, 16),)),
        ("b n h w, n d -> b d h w",((64, 32, 8, 4), (32, 16))),
        ("n d, n d -> n",((64, 32), (64, 32))),
        ("i d, j d -> i j",((64, 32), (48, 32))),
        ("b h i d, b h j d -> b h i j",((64, 32, 4, 8), (64, 32, 6, 8))),
        ("b h i j, b h j d -> b h i d",((64, 32, 4, 8), (64, 32, 8, 6))),
        ("b i d, b i j d -> b i j",((64, 32, 4), (64, 32, 8, 4))),
        ("b x i d, b j d -> b x i j",((64, 32, 4, 8), (64, 5, 8))),
        ("b x i j, b j d -> b x i d",((64, 32, 4, 5), (64, 5, 8))),
        ("hij, ijc->ihc",((64, 32, 16), (32, 16, 8))),
        ("rac,rab->rbc",((64, 32, 4), (64, 32, 7))),
        ("ra,rab->rb",((64, 32), (64, 32, 8))),
        ("qhc,khc->qkh",((64, 32, 4), (48, 32, 4))),
        ("nm, mrc->nrc",((64, 32), (32, 8, 6))),
        ("abc,adc->bdc",((64, 32, 15), (64, 13, 15))),
        ("dceb,cef->dbf",((64, 32, 4, 8), (32, 4, 13))),
        ("acb,ade->dceb",((64, 32, 7), (64, 15, 9))),
        ("qkc,ch->hqk",((64, 32, 4), (4, 13))),
        ("bhqk,bkhc->bqhc",((64, 32, 4, 8), (64, 8, 32, 7))),
        ("bqa,ahc->bqhc",((64, 32, 8), (8, 15, 9))),
        ("...lc, ...c -> ...l",((64, 32, 7), (64, 7))),
        ("...lc, ...lc -> ...l",((64, 32, 7), (64, 32, 7))),
        ("...id,...jd->...ij",((64, 32, 4, 8), (64, 32, 5, 8))),
        ("...klm,kmn->...kln",((64, 32, 4, 8), (32, 8, 11))),
        ("...ikl, ...jk -> ...ijl",((64, 32, 4, 8), (64, 15, 4))),
        ("...l,...l->...",((64, 32, 17), (64, 32, 17))),
        ("ijk,ijk...->ij...",((64, 32, 4), (64, 32, 4, 9))),
        ("bxi,oij,byj->boxy",((64, 32, 5), (17, 5, 13), (64, 9, 13))),
        ("ijac,ijkp->ijakcp",((64, 32, 4, 8), (64, 32, 5, 7))),
        ("cdij,cbi->cdbj",((64, 32, 4, 8), (64, 19, 4))),
        ("bsid,bsjd->bijd",((64, 32, 4, 8), (64, 32, 17, 8))),
        ("bsid,bsje->bijde",((64, 32, 4, 8), (64, 32, 17, 9))),
        ("...bac,...dae->...bdce",((64, 32, 4, 8), (64, 19, 4, 5))),
        ("...abc,...adc->...bdc",((64, 32, 4, 8), (64, 32, 7, 8))),
        ("...qhd,...khd->...hqk",((64, 32, 4, 8), (64, 23, 4, 8))),
        ("...vhf,...qhv->...qhf",((64, 32, 4, 8), (64, 19, 4, 32))),
        ("...ij,jk->ik",((64, 32, 4, 8), (8, 13))),
    ]
    
    def test_einsum_op_simple(self):
        for equation, nshapes in TestEinsumOps._test_args:
            inputs_np = []
            inputs_hetu = []
            for shape in nshapes:
                input_np = np.random.randn(*shape) * 10
                input_hetu = hetu.from_numpy(input_np)
                inputs_np.append(torch.from_numpy(input_np))
                inputs_hetu.append(input_hetu)
            gt = torch.einsum(equation, *inputs_np).numpy()
            self.assertTrue(allclose(hetu.einsum(equation, inputs_hetu), gt))

class TestOtherOps(unittest.TestCase):

    _embedding_test_shapes = [
        ((4, 4), (5)),
        ((16, 32), (16))
    ]

    _onehot_test_shapes = [
        (32, 4),
        (64,)
    ]
    
    def test_embedding_lookupop(self):
        for shape_x, shape_id in TestOtherOps._embedding_test_shapes:
            x_np = np.random.randn(*shape_x)
            id_np = np.random.randint(0, shape_x[0], size=shape_id)
            gt = torch.embedding(torch.from_numpy(x_np), torch.from_numpy(id_np)).numpy()
            x = hetu.from_numpy(x_np)
            id = hetu.from_numpy(id_np)
            self.assertTrue(allclose(hetu.embedding_lookup(x, id), gt))
            self.assertTrue(allclose(x.embedding_lookup(id), gt))

    def test_onehotop(self):
        for shape_x in TestOtherOps._onehot_test_shapes:
            x_np = np.random.randint(0, 16, size=shape_x)
            gt = torch.nn.functional.one_hot(torch.from_numpy(x_np), num_classes = 16).numpy()
            x = hetu.from_numpy(x_np)
            self.assertTrue(allclose(hetu.onehot(x, 16), gt))
            self.assertTrue(allclose(x.onehot(16), gt))

    def test_whereop(self):
        for shape_x in TestOtherOps._onehot_test_shapes:
            cond_np = np.random.choice([True, False], size=shape_x)
            x_np = np.random.randn(*shape_x)
            y_np = np.random.randn(*shape_x)
            gt = np.where(cond_np, x_np, y_np)
            cond = hetu.from_numpy(cond_np) 
            x = hetu.from_numpy(x_np)
            y = hetu.from_numpy(y_np)
            self.assertTrue(allclose(hetu.where(cond, x, y), gt))

                

if __name__ == "__main__":
    unittest.main()
