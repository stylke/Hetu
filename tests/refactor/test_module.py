import hetu
import hetu.nn as nn
import numpy as np
import torch
import unittest
from test_utils import allclose

class TestLinearModules(unittest.TestCase):

    _test_shapes = [
        ((64, 256), (256, 128))
    ]
    
    def test_linear_op(self):
        for shape_x, shape_y in TestLinearModules._test_shapes:
            x_np = np.random.randn(*shape_x).astype(np.float32)
            x = hetu.from_numpy(x_np)
            nn_linear = hetu.nn.Linear(256, 128)
            out = nn_linear(x).numpy(force=True)

class TestActivationModules(unittest.TestCase):

    _test_shapes = [
        (64, 256),
        (1024, 16)
    ]

    def test_sigmoid_op(self):
        for shape in TestActivationModules._test_shapes:
            x_np = np.random.randn(*shape)
            gt = 1 / (1 + np.exp(-x_np))
            x = hetu.from_numpy(x_np)
            nn_sigmoid = nn.modules.Sigmoid()
            self.assertTrue(allclose(nn_sigmoid(x), gt))
    
    def test_relu_op(self):
        for shape in TestActivationModules._test_shapes:
            x_np = np.random.randn(*shape)
            gt = x_np * (x_np > 0).astype(x_np.dtype)
            x = hetu.from_numpy(x_np)
            nn_relu = nn.modules.ReLU()
            self.assertTrue(allclose(nn_relu(x), gt))
            
    
    def test_leaky_relu_op(self):
        for shape in TestActivationModules._test_shapes:
            x_np = np.random.randn(*shape)
            alphas = [0.1, 0.2, 0.5]
            for alpha in alphas:
                gt = np.where(x_np > 0, x_np, alpha * x_np)
                x = hetu.from_numpy(x_np)
                nn_leakyrelu = nn.modules.LeakyReLU(alpha)
                self.assertTrue(allclose(nn_leakyrelu(x), gt))

    def test_tanh_op(self):
        for shape in TestActivationModules._test_shapes:
            x_np = np.random.randn(*shape)
            gt = np.tanh(x_np)
            x = hetu.from_numpy(x_np)
            nn_tanh = nn.modules.Tanh()
            self.assertTrue(allclose(nn_tanh(x), gt))


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
            x_np = np.random.randn(*shape).astype(np.float32)
            x = hetu.from_numpy(x_np)
            for f_shape in TestConv2dOps._filter_shapes:
                f_np = np.random.randn(*f_shape).astype(np.float32)
                f = hetu.from_numpy(f_np)
                bias_shape = [f_shape[0]]
                # test conv2d add bias
                nn_conv2d = hetu.nn.Conv2d(shape[1], f_shape[0], (f_shape[2], f_shape[3]))
                ot = nn_conv2d(x)

class TestSequential(unittest.TestCase):

    _data_shapes = [
        (4, 3, 16, 16),        
    ]

    _filter_shapes = [
        (3, 3, 2, 2),
        (4, 3, 4, 4)        
    ]

    def test_conv2d_op(self):
        for shape in TestConv2dOps._data_shapes:
            x_np = np.random.randn(*shape).astype(np.float32)
            x = hetu.from_numpy(x_np)
            for f_shape in TestConv2dOps._filter_shapes:
                f_np = np.random.randn(*f_shape).astype(np.float32)
                f = hetu.from_numpy(f_np)
                bias_shape = [f_shape[0]]
                # test conv2d add bias
                nn_seq = hetu.nn.Sequential(hetu.nn.Conv2d(shape[1], f_shape[0], (f_shape[2], f_shape[3])),
                                            hetu.nn.Sigmoid())
                ot = nn_seq(x)


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
            nn_maxpool2d = hetu.nn.MaxPool2d(2, 1, 0)
            self.assertTrue(allclose(nn_maxpool2d(x), gt))

    def test_avgpool_op(self):
        for shape in TestPoolOps._test_shapes:
            x_np = np.random.randn(*shape)
            x = hetu.from_numpy(x_np)
            avgpool2d = torch.nn.AvgPool2d(2, 1, 0)
            gt = avgpool2d(torch.from_numpy(x_np)).numpy()
            nn_avgpool2d = hetu.nn.AvgPool2d(2, 1, 0)
            self.assertTrue(allclose(nn_avgpool2d(x), gt))

class TestNormModules(unittest.TestCase):

    _test_shapes = [
        (4, 3, 16, 16),      
        (5, 8, 16, 16)  
    ]


    def test_batchnorm_op(self):
        for shape in TestPoolOps._test_shapes:
            x_np = np.random.randn(*shape).astype(np.float32)
            x = hetu.from_numpy(x_np)
            scale_np = np.ones(shape[1]).astype(np.float32)
            bias_np = np.zeros(shape[1]).astype(np.float32)
            # ssf = torch.nn.BatchNorm2d(shape[1])
            gt = torch.batch_norm(torch.from_numpy(x_np), weight = torch.from_numpy(scale_np), bias = torch.from_numpy(bias_np),
                                 running_mean=None, running_var=None, training=True, momentum=0.1, eps=1e-5, cudnn_enabled=True).numpy()
            # gt = ssf(torch.from_numpy(x_np)).detach().numpy()
            nn_batchnorm = hetu.nn.BatchNorm(shape[1], 1e-5, 0.1)
            self.assertTrue(allclose(nn_batchnorm(x), gt))

    def test_layernorm_op(self):
        for shape in TestPoolOps._test_shapes:
            norm_shape = shape[3:]
            x_np = np.random.randn(*shape).astype(np.float32)
            x = hetu.from_numpy(x_np)
            scale_np = np.ones(norm_shape).astype(np.float32)
            bias_np = np.zeros(norm_shape).astype(np.float32)
            layernorm = torch.nn.LayerNorm(norm_shape, 1e-5)
            gt = layernorm(torch.from_numpy(x_np)).detach().numpy()
            nn_layernorm = hetu.nn.LayerNorm(norm_shape, 1e-5)
            self.assertTrue(allclose(nn_layernorm(x), gt))
    
    def test_instancenorm_op(self):
        for shape in TestPoolOps._test_shapes:
            x_np = np.random.randn(*shape).astype(np.float32)
            x = hetu.from_numpy(x_np)
            instancenorm = torch.nn.InstanceNorm2d(num_features=shape[1], eps=0)
            gt = instancenorm(torch.from_numpy(x_np)).detach().numpy()
            nn_instancenorm = hetu.nn.InstanceNorm(shape[1], eps=0)
            self.assertTrue(allclose(nn_instancenorm(x), gt))

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

    def test_bce_op(self):
        MIN_VALUE = -100.0
        for shape in TestLossOps._test_binary_label_shapes:
            probs_np = np.random.uniform(1e-10, 1, size=shape).astype(np.float32)
            labels_np = np.random.choice([0, 1], size=shape).astype(np.float32)
            bce = torch.nn.BCELoss()
            gt = bce(torch.from_numpy(probs_np), torch.from_numpy(labels_np)).numpy()
            nn_bce = hetu.nn.BCELoss()
            probs = hetu.from_numpy(probs_np)
            labels = hetu.from_numpy(labels_np)
            loss = nn_bce(probs, labels)
            self.assertTrue(allclose(loss, gt))
    
    def test_nllloss_op(self):
        for shape, lshape in TestLossOps._test_nllloss_label_shapes:
            probs_np = np.random.uniform(1e-10, 1, size=shape).astype(np.float32)
            labels_np = np.random.choice(range(16), size=lshape).astype(np.int64)
            nll = torch.nn.NLLLoss()
            gt = nll(torch.from_numpy(probs_np), torch.from_numpy(labels_np)).numpy()
            probs = hetu.from_numpy(probs_np)
            labels = hetu.from_numpy(labels_np)
            nn_nll = hetu.nn.NLLLoss()
            loss = nn_nll(probs, labels)
            self.assertTrue(allclose(loss, gt))
    
    def test_kldivloss_op(self):
        MIN_VALUE = -100.0
        for shape in TestLossOps._test_binary_label_shapes:
            probs_np = np.random.uniform(1e-10, 1, size=shape).astype(np.float32)
            labels_np = np.random.choice([0, 1], size=shape).astype(np.float32)
            kldiv = torch.nn.KLDivLoss()
            gt = kldiv(torch.from_numpy(probs_np), torch.from_numpy(labels_np)).numpy()
            nn_kldiv = hetu.nn.KLDivLoss()
            probs = hetu.from_numpy(probs_np)
            labels = hetu.from_numpy(labels_np)
            loss = nn_kldiv(probs, labels)
            self.assertTrue(allclose(loss, gt))
    
    def test_mseloss_op(self):
        MIN_VALUE = -100.0
        for shape in TestLossOps._test_binary_label_shapes:
            probs_np = np.random.uniform(1e-10, 1, size=shape).astype(np.float32)
            labels_np = np.random.choice([0, 1], size=shape).astype(np.float32)
            mse = torch.nn.MSELoss()
            gt = mse(torch.from_numpy(probs_np), torch.from_numpy(labels_np)).numpy()
            nn_mse = hetu.nn.MSELoss()
            probs = hetu.from_numpy(probs_np)
            labels = hetu.from_numpy(labels_np)
            loss = nn_mse(probs, labels)
            self.assertTrue(allclose(loss, gt))
                

if __name__ == "__main__":
    unittest.main()
