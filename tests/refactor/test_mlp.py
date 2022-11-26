import hetu
import numpy as np
import torch
import unittest

# from hetu import device

# class DerivedDevice(torch.device):
#     def __init__(self):
#         pass

# class TorchMLP(torch.nn.Module):
    
#     def __init__(self, dims=[256, 64, 16, 1]):
#         super(TorchMLP, self).__init__()
#         layers = []
#         for i in range(1, len(dims)):
#             layers.append(torch.nn.Linear(dims[i - 1], dims[i]))
#             if i + 1 < len(dims):
#                 layers.append(torch.nn.ReLU())
#         self.mlp = torch.nn.Sequential(*layers)
    
#     def forward(self, x):
#         return self.mlp(x)

# class HetuMLP(object):

#     def __init__(self, torch_mlp):
#         torch_weights = list(torch_mlp.mlp.parameters())

# class XXX(hetu.nn.Module):

#     def __init__(self):
#         super(XXX, self).__init__()
#         w_np = np.random.uniform(0.0, 1.0, size=(64, 16))
#         self.a1 = hetu.Tensor(w_np)
#         self.w2 = hetu.Tensor(w_np)
#         self.weight = hetu.nn.Parameter(hetu.Tensor(w_np))

class TorchMLP(torch.nn.Module):
    
    def __init__(self, weights_np):
        super(TorchMLP, self).__init__()
        for i, w_np in enumerate(weights_np):
            self.register_parameter(
                f"w{i}", 
                torch.nn.Parameter(torch.tensor(w_np)))
        self.num_layers = len(weights_np)
    
    def forward(self, x):
        act = x
        for i in range(self.num_layers):
            w = self.__getattr__(f"w{i}")
            act = torch.matmul(act, w)
            if i + 1 < self.num_layers:
                act = act.relu()
        return act

class HetuMLP(hetu.nn.Module):
    
    def __init__(self, weights_np):
        super(HetuMLP, self).__init__()
        for i, w_np in enumerate(weights_np):
            # TODO: set trainable inside make subclass
            self.register_parameter(
                f"w{i}", 
                hetu.nn.Parameter(hetu.Tensor(w_np, trainable=True)))
        self.num_layers = len(weights_np)
    
    def forward(self, x):
        act = x
        for i in range(self.num_layers):
            w = self.__getattr__(f"w{i}")
            # act = hetu.matmul(act, w)
            with hetu.context(extra_deps=[w]):
                act = act.matmul(w)
            if i + 1 < self.num_layers:
                act = hetu.relu(act)
        return act
    
    # def __init__(self, dims=[256, 64, 16, 1]):
    #     super(HetuMLP, self).__init__()
    #     w_np = np.random.uniform(0.0, 1.0, size=(64, 16))
    #     self.w1 = hetu.Tensor(w_np)
    #     self.w2 = hetu.Tensor(w_np)
    #     self.w3 = hetu.Tensor(w_np)
    #     # self.w1 = 1
    #     self.sub = XXX()
    #     for name, v in self.named_parameters():
    #         print(name, v)
    #     for name, v in self.named_buffers():
    #         print(name, v)

# mlp = HetuMLP()
# if 1 + 1 == 2:
#     import sys
#     sys.exit(0)


class TestMLP(unittest.TestCase):

    def test_simple(self):
        # return
        # w_np = np.random.randn(10, 1).astype(np.float32)
        weights_np = []
        dims = [256, 64, 16, 1]
        # dims = [10, 1]
        for i in range(1, len(dims)):
            w = np.random.uniform(0.0, 1.0, size=(dims[i - 1], dims[i]))
            w = ((w - 0.5) * (2.0 / np.sqrt(dims[i - 1]))).astype(np.float32)
            # w = np.random.randn(dims[i - 1], dims[i]).astype(np.float32)
            weights_np.append(w)

        hetu_model = HetuMLP(weights_np)
        optimizer = hetu.optim.SGD(hetu_model.parameters(), lr=0.1)
        
        torch_model = TorchMLP(weights_np)
        optimizer_torch = torch.optim.SGD(torch_model.parameters(), lr=0.1)
        
        '''
        # w = hetu.Tensor(w_np, trainable=True)
        weights = [hetu.Tensor(w_np, trainable=True) for w_np in weights_np]
        # from hetu.optim.sgd import SGD
        optimizer = hetu.optim.SGD(weights, lr=0.1)

        # import torch
        weights_torch = [torch.tensor(w_np, requires_grad=True) for w_np in weights_np]
        # w_torch = torch.tensor(w_np, requires_grad=True)
        optimizer_torch = torch.optim.SGD(weights_torch, lr=0.1)
        '''

        # diffs, ratios = [], []
        # for w1, w2 in zip(hetu_model.parameters(), torch_model.parameters()):
        #     w1 = w1.numpy(force=True)
        #     w2 = w2.detach().numpy()
        #     # print(w1.shape, w2.shape)
        #     assert w1.shape == w2.shape
        #     diff = np.abs(w1 - w2).sum()
        #     ratio = (np.abs((w1 - w2) / w2)).sum()
        #     diffs.append(diff)
        #     ratios.append(ratio)
        # print(diffs, ratios)

        for i in range(100):
            x_np = np.random.randn(16, dims[0]).astype(np.float32)
            y_np = np.random.choice([0, 1], size=(16, 1)).astype(np.float32)

            x = hetu.from_numpy(x_np)
            y = hetu.from_numpy(y_np)
            
            # act = x
            # for i, w in enumerate(weights):
            #     act = hetu.matmul(act, w)
            #     if i + 1 < len(weights):
            #         act = hetu.relu(act)
            # preds = act
            preds = hetu_model(x)
            probs = hetu.sigmoid(preds)
            # preds = hetu.matmul(x, w)
            # probs = hetu.sigmoid(preds)
            loss = hetu.binary_cross_entroy(probs, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # MIN_VALUE = -100.0
            # preds_np = np.matmul(x_np, w_np)
            # # print(x_np.shape)
            # # print(w_np.shape)
            # # print(preds_np.shape)
            # probs_np = 1 / (1 + np.exp(-preds_np))
            # t1_np = np.maximum(np.log(probs_np), MIN_VALUE)
            # t2_np = np.maximum(np.log(1 - probs_np), MIN_VALUE)
            # loss_np = -(y_np * t1_np + (1 - y_np) * t2_np)

            # self.assertTrue(np.allclose(loss.numpy(force=True), loss_np))
            # self.assertTrue(np.allclose(preds.numpy(force=True), preds_np))
            # self.assertTrue(np.allclose(probs.numpy(force=True), probs_np))

            x_torch = torch.from_numpy(x_np)
            y_torch = torch.from_numpy(y_np)

            # act_torch = x_torch
            # for i, w in enumerate(weights_torch):
            #     act_torch = torch.matmul(act_torch, w)
            #     if i + 1 < len(weights_torch):
            #         act_torch = act_torch.relu()
            # preds_torch = act_torch
            preds_torch = torch_model(x_torch)
            # preds_torch = torch.matmul(x_torch, w_torch)
            probs_torch = preds_torch.sigmoid()
            loss_torch = torch.nn.BCELoss(reduction='none')(probs_torch, y_torch)
            optimizer_torch.zero_grad()
            loss_torch.sum().backward()
            optimizer_torch.step()
            
            # print(loss_torch.detach().numpy().reshape(-1))
            # print(loss.numpy(force=True).reshape(-1))

            # print(w_torch.detach().numpy().reshape(-1))
            # print(w.data.numpy(force=True).reshape(-1))
            
            # print(w_torch.grad.detach().numpy().reshape(-1))
            # print(w.grad.numpy(force=True).reshape(-1))

            diffs, ratios = [], []
            # print('loss1', loss.numpy(True).reshape(-1))
            # print('loss2', loss_torch.detach().numpy().reshape(-1))
            for w1, w2 in zip(hetu_model.parameters(), torch_model.parameters()):
                # print('g1', w1.grad.numpy(force=True).reshape(-1))
                # print('g2', w2.detach().numpy().reshape(-1))
                w1 = w1.numpy(force=True)
                w2 = w2.detach().numpy()
                assert w1.shape == w2.shape
                diff = np.abs(w1 - w2).sum()
                ratio = (np.abs((w1 - w2) / w2)).sum()
                diffs.append(diff)
                ratios.append(ratio)
            # for i in range(len(weights_np)):
            #     w1 = weights[i].numpy(force=True)
            #     w2 = weights_torch[i].detach().numpy()
            #     diff = np.abs(w1 - w2).sum()
            #     ratio = (np.abs((w1 - w2) / w2)).sum()
            #     diffs.append(diff)
            #     ratios.append(ratio)
            
            print(diffs, ratios)

            # print((w_torch.detach().numpy() - w.data.numpy(force=True)).sum())
            # print((w_torch.grad.detach().numpy() - w.grad.numpy(force=True)).sum())

            # self.assertTrue(np.allclose(loss.numpy(force=True), loss_torch.detach().numpy()))
            # self.assertTrue(np.allclose(w.numpy(force=True), w_torch.detach().numpy()))
            # self.assertTrue(np.allclose(w.grad.numpy(force=True), w_torch.grad.detach().numpy()))
        
        
        # xxx = loss.numpy(force=True)
        # print(xxx.reshape(-1), xxx.shape)
        # print(loss_np.reshape(-1), loss_np.shape)
        # self.assertTrue(np.allclose(loss.numpy(force=True), loss_np))

    def test_init(self):
        return
        # weights_np = []
        # dims = [256, 64, 16, 1]
        # # dims = [10, 1]
        # for i in range(1, len(dims)):
        #     w = np.random.uniform(0.0, 1.0, size=(dims[i - 1], dims[i]))
        #     w = ((w - 0.5) * (2.0 / np.sqrt(dims[i - 1]))).astype(np.float32)
        #     # w = np.random.randn(dims[i - 1], dims[i]).astype(np.float32)
        #     weights_np.append(w)
        
        # hetu_model = HetuMLP(weights_np)

        def moments(t):
            if isinstance(t, hetu.Tensor):
                t_np = t.numpy(True).reshape(-1)
            elif isinstance(t, torch.Tensor):
                t_np = t.detach().numpy().reshape(-1)
            else:
                raise ValueError(f"{type(t).__name__}")
            from scipy.stats import moment
            # m = [moment(t_np, moment=i) for i in range(1, 5)]
            # print(m)
            print(t_np.mean(), t_np.std())

        x = hetu.nn.Parameter(hetu.empty([1024 * 4, 1024]))
        x_torch = torch.nn.Parameter(torch.empty([1024 * 4, 1024]))
        # x.uniform_()
        # x_torch.uniform_()
        moments(hetu.nn.init.uniform_(x))
        moments(torch.nn.init.uniform_(x_torch))

        moments(hetu.nn.init.normal_(x))
        moments(torch.nn.init.normal_(x_torch))

        moments(hetu.nn.init.zeros_(x))
        moments(torch.nn.init.zeros_(x_torch))

        moments(hetu.nn.init.ones_(x))
        moments(torch.nn.init.ones_(x_torch))

        moments(hetu.nn.init.trunc_normal_(x))
        moments(torch.nn.init.trunc_normal_(x_torch))

        moments(hetu.nn.init.he_uniform_(x))
        moments(torch.nn.init.kaiming_uniform_(x_torch))

        moments(hetu.nn.init.he_normal_(x))
        moments(torch.nn.init.kaiming_normal_(x_torch))

        moments(hetu.nn.init.xavier_normal_(x))
        moments(torch.nn.init.xavier_normal_(x_torch))

        moments(hetu.nn.init.xavier_uniform_(x))
        moments(torch.nn.init.xavier_uniform_(x_torch))
        
        
        # print(x.data)
        # x.normal_()
        # print(x.data)
        # x.zero_()
        # print(x.data)
        # x.trunc_normal_()
        # print(x.data)
        # hetu.nn.init.he_uniform_(x)
        # print(x.data)
    
    def test_linear(self):
        model = hetu.nn.Linear(8, 1)
        x = hetu.randn([10, 8])
        y = hetu.zeros([10, 1])
        output = model(x)
        print(output.data)
        print(model.weight.data)
        print(model.bias.data)


if __name__ == "__main__":
    unittest.main()
