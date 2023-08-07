import hetu
import hetu.nn as nn
import numpy as np
import torch
import unittest
from test_utils import allclose
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from struct import unpack
import gzip
import torchvision

BATCHSIZE = 32


def naive_mnist():
    train = torchvision.datasets.MNIST(root="/root/hetu-pytest/datasets/", train=True, download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(), 
                                           torchvision.transforms.Normalize((0.1307,), (0.3081,)) 
                                       ]))
    train_loader = DataLoader(train, batch_size=BATCHSIZE)
    test = torchvision.datasets.MNIST(root="/root/hetu-pytest/datasets/", train=False, download=True,
                                      transform=torchvision.transforms.Compose([
                                          torchvision.transforms.ToTensor(), 
                                          torchvision.transforms.Normalize((0.1307,), (0.3081,)) 
                                      ]))
    test_loader = DataLoader(test, batch_size=BATCHSIZE) 

    return train_loader, test_loader

class Torch_RNN(torch.nn.Module):
    
    def __init__(self):
        super(Torch_RNN, self).__init__()
        self.indim = 28
        self.hiddendim = 128
        self.outdim = 10
        self.linear1 = torch.nn.Linear(self.indim, self.hiddendim)
        self.linear2 = torch.nn.Linear(self.hiddendim * 2, self.hiddendim)
        self.linear3 = torch.nn.Linear(self.hiddendim, self.outdim)
        self.last_layer = None
 
    def forward(self, x):#1*28*28
        pic = torch.reshape(x, (-1, 784)) #784      
        for i in range(28):
            tmp = pic[:,i*28:(i+1)*28]
            if (i == 0):
                self.last_layer = torch.zeros(pic.shape[0], self.hiddendim).cuda()
            tmp = self.linear1(tmp)
            tmp = torch.cat((tmp, self.last_layer), dim=1)
            tmp = self.linear2(tmp)
            tmp = torch.relu(tmp)
            self.last_layer = tmp 
        out = self.last_layer
        out = self.linear3(out)
        return out

class Hetu_RNN(hetu.nn.Module):
    
    def __init__(self):
        super(Hetu_RNN, self).__init__()
        self.indim = 28
        self.hiddendim = 128
        self.outdim = 10
        self.linear1 = hetu.nn.Linear(self.indim, self.hiddendim)
        self.linear2 = hetu.nn.Linear(self.hiddendim * 2, self.hiddendim)
        self.linear3 = hetu.nn.Linear(self.hiddendim, self.outdim)
        # self.last_layer = hetu.nn.Parameter(hetu.empty([self.hiddendim], trainable=True))
    
    def forward(self, x):#3*32*32
        pic = hetu.reshape(x, (x.shape[0], 784)) #784      
        for i in range(28):
            tmp = hetu.slice(pic, [0, i*28], [-1, 28])
            tmp = self.linear1(tmp)
            if (i == 0):
                self.last_layer = hetu.broadcast(hetu.zeros([self.hiddendim]), tmp)
            tmp = hetu.concat(tmp, self.last_layer, axis=1)
            tmp = self.linear2(tmp)
            tmp = hetu.relu(tmp)
            self.last_layer = tmp 
        out = self.last_layer
        out = self.linear3(out)
        return out



def train(model, epoch, train_loader, optimizer):
    model.train()

    for step, (x, y) in enumerate(train_loader):

        model = model.cuda()

        x, y = x.cuda(), y.cuda()

        optimizer.zero_grad()
 
        output = model(x)

        loss = F.cross_entropy(output, y)
 
        loss.backward()
 
        optimizer.step()
 
        if step % 500 == 0:
            print('Epoch: {}, Step {}, Loss: {}'.format(epoch, step, loss))

def Nan_detect(torch_model, hetu_model):
    pass

def test(model, test_loader):
    """测试"""
 
    # 测试模式
    model.eval()
 
    # 存放正确个数
    correct = 0
 
    with torch.no_grad():
        for x, y in test_loader:
 
            model = model.cuda()
            x, y = x.cuda(), y.cuda()
 
            output = model(x)
 
            pred = output.argmax(dim=1, keepdim=True)
 
            correct += pred.eq(y.view_as(pred)).sum().item()
 
    accuracy = correct / len(test_loader.dataset) * 100
 
    print("Test Accuracy: {}%".format(accuracy))


def hetu_train_ex(hetu_model, epoch, train_loader):

    x = hetu.placeholder(hetu.float32, shape=[-1, 1, 28, 28])
    y = hetu.placeholder(hetu.float32, shape=[-1])
    pred = hetu_model(x)
    # test_out = hetu_model.test(x)
    # test_out2 = hetu_model.test2(x)
    # test_out3 = hetu_model.test3(x)
    loss = hetu.softmax_cross_entropy(pred, hetu.onehot(y,10))
    # probs = hetu.sigmoid(pred)
    # loss = hetu.binary_cross_entropy(probs, y, "mean")

    optimizer = hetu.optim.SGD(lr=0.01)
    train_op = optimizer.minimize(loss)

    executor = hetu.DARExecutor("cuda:0")

    import time 
    st = time.time()

    for step, (x0, y0) in enumerate(train_loader):
        x1 = x0.numpy().astype(np.float32)
        y1 = y0.numpy().astype(np.float32) #F.one_hot(y0, num_classes = 10).numpy().astype(np.float32)
        loss_val, _ = executor.run([loss, train_op], feed_dict={
            x: x1, 
            y: y1, 
        })

        if step % 200 == 0 :
            # print("TESTOUT:", testtime[0].numpy(force=True).flat[:20], " ", testtime[0].numpy(force=True).shape)
            # print("TESTOUT2:", testtime2[0].numpy(force=True).flat[:20])
            # print("TESTOUT3:", testtime3[0].numpy(force=True).flat[:20])
            print('Epoch: {}, Step {}, Loss: {}, Loss_val: {}'.format(epoch, step, loss_val, loss_val.numpy(force=True)[0]))
            # assert(1==0)

def hetu_test_ex(model, test_loader):
    x = hetu.placeholder(hetu.float32, shape=[-1, 3, 32, 32])
    y = hetu.placeholder(hetu.float32, shape=[-1])
    output = model(x)
    executor = hetu.DARExecutor("cuda:0")

    correct = 0
 
    for x0, y0 in test_loader:

        x1, y1 = x0.numpy().astype(np.float32), y0.numpy().astype(np.float32)#F.one_hot(y0, num_classes = 10).numpy().astype(np.float32)

        out = executor.run([output], feed_dict={
            x: x1, 
            y: y1, 
        })
        
        out = out[0].numpy(force=True)

        pred = torch.from_numpy(out).argmax(dim=1, keepdim=True)

        correct += pred.eq(y0.view_as(pred)).sum().item()

    accuracy = correct / len(test_loader.dataset) * 100
    print("Test Accuracy: {}%".format(accuracy))

MNIST_PATH = "/root/hetu-pytest/datasets/mnist_dataset/"
def torch_mnist():
    learning_rate = 0.01
    epoches = 200
    trainset, testset = naive_mnist()
    # network = Torch_CNN() 
    network = Torch_RNN() 
    print(torch.cuda.is_available())
    optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate)  # 优化器
    for epoch in range(epoches):
        print("\n================ epoch: {} ================".format(epoch))
        train(network, epoch, trainset, optimizer)
        test(network, testset)

def hetu_mnist_ex():
    epoches = 200
    trainset, testset = naive_mnist()
    network = Hetu_RNN()  
    print(torch.cuda.is_available())
      # 优化器
    for epoch in range(epoches):
        print("\n================ epoch: {} ================".format(epoch))
        hetu_train_ex(network, epoch, trainset)
        hetu_test_ex(network, testset)


def pa_train(hetu_model, torch_model, epoch, train_loader):
    
    x = hetu.placeholder(hetu.float32, shape=[-1, 1, 28, 28])
    y = hetu.placeholder(hetu.float32, shape=[-1, 10])
    pred = hetu_model(x)
    loss = hetu.softmax_cross_entropy(pred, y)
    # probs = hetu.sigmoid(pred)
    # loss = hetu.binary_cross_entropy(probs, y, "mean")

    optimizer = hetu.optim.SGD(lr=0.1)
    train_op = optimizer.minimize(loss)

    criterion_torch = torch.nn.CrossEntropyLoss()
    optimizer_torch = torch.optim.SGD(torch_model.parameters(), lr=0.1)

    # criterion_hetu = hetu.nn.CrossEntropyLoss()
    optimizer_hetu = hetu.optim.SGD(hetu_model.parameters(), lr=0.001)
    # diffs, ratios = [], []
    # w1s, w2s = [], []
    # for w1, w2 in zip(hetu_model.parameters(), torch_model.parameters()):
    #     w1 = w1.numpy(force=True)
    #     w2 = w2.cpu().detach().numpy()
    #     # print(w1.shape, " ", w2.shape)
    #     assert w1.shape == w2.shape
    #     diff = np.abs(w1 - w2).sum()
    #     ratio = (np.abs((w1 - w2) / w2)).sum()
    #     diffs.append(diff)
    #     ratios.append(ratio)
    #     w1s.append(w1.sum())
    #     w2s.append(w2.sum())
    #     print(w1.shape, " ", diff, " ", ratio, " ", w1.flat[:10], " ", w2.flat[:10])


    executor = hetu.DARExecutor("cuda:0")

    torch_model = torch_model.cuda()

    for step, (x0, y0) in enumerate(train_loader):

        x_val = x0.numpy().astype(np.float32)
        y_val = y0.numpy().astype(np.int64)

        y_onehot = torch.nn.functional.one_hot(torch.from_numpy(y_val), num_classes = 10).numpy().astype(np.float32)

        loss_val, _ = executor.run([loss, train_op], feed_dict={
            x: x_val, 
            y: y_onehot, 
        })

        x_torch = torch.from_numpy(x_val)
        y_torch = torch.from_numpy(y_val)
    
        optimizer_torch.zero_grad()
        preds_torch = torch_model(x_torch.cuda())
        loss_torch = criterion_torch(preds_torch, y_torch.cuda())
        loss_torch.backward()
        optimizer_torch.step()

        # x_hetu = hetu.from_numpy(x_val)
        # y_hetu = hetu.from_numpy(y_onehot)
    
        # optimizer_hetu.zero_grad()
        # preds_hetu = hetu_model(x_hetu)
        # loss_hetu = hetu.softmax_cross_entropy(preds_hetu, y_hetu)
        # loss_hetu.backward()
        # optimizer_hetu.step()

        # loss_val = loss_hetu.numpy(force=True)

        if step % 100 == 0:
            
            print('Epoch: {}, Step {}, Loss_t: {}, Loss_h:{} '.format(epoch, 
                step, loss_torch, loss_val))


            diffs, ratios, shapes = [], [], []
            w1s, w2s = [], []
            for w1, w2 in zip(hetu_model.parameters(), torch_model.parameters()):
                w1 = w1.numpy(force=True)
                w2 = w2.cpu().detach().numpy()
                # print(w1.flat[:10], " ", w2.flat[:10], " ", w1.shape)
                # print(w1.shape, " ", w2.shape)
                assert w1.shape == w2.shape
                diff = np.abs(w1 - w2).sum()
                ratio = (np.abs((w1 - w2) / w2)).sum()
                diffs.append(diff)
                ratios.append(ratio)
                w1s.append(w1.sum())
                w2s.append(w2.sum())
                shapes.append(w1.shape)
            # print(w1s, " ", w2s)
            # print(diffs, ratios, shapes)
            # print(torch_model.tmp.grad.cpu().detach().numpy().flat[:10])
            # print(torch_model.tmp2.grad.cpu().detach().numpy().flat[:10])
            # assert(1==0)
            # print(torch_model.norm_f.grad.cpu().detach().numpy().flat[:10])

    torch_model = torch_model.cpu()

def pa_test(hetu_model, torch_model, test_loader):
   
    x = hetu.placeholder(hetu.float32, shape=[-1, 3, 32, 32])
    y = hetu.placeholder(hetu.float32, shape=[-1, 10])
    pred = hetu_model(x)
    # probs = hetu.sigmoid(pred)
    # loss = hetu.binary_cross_entropy(probs, y, "mean")

    executor = hetu.DARExecutor("cuda:0")

    correct_torch = 0
    correct_hetu = 0

    torch_model = torch_model.cuda()

    for step, (x0, y0) in enumerate(test_loader):

        x_val = x0.numpy().astype(np.float32)
        y_val = y0.numpy().astype(np.int64)

        y_onehot = torch.nn.functional.one_hot(torch.from_numpy(y_val), num_classes = 10).numpy().astype(np.float32)

        pred_val = executor.run([pred], feed_dict={
            x: x_val, 
            y: y_onehot, 
        })

        x_torch = torch.from_numpy(x_val)
        y_torch = torch.from_numpy(y_val)

        preds_torch = torch_model(x_torch.cuda())

        preds_torch = preds_torch.argmax(dim=1, keepdim=True).cpu()
 
        correct_torch += preds_torch.eq(y_torch.view_as(preds_torch)).sum().item()

        output = pred_val[0].numpy(force=True)

        preds_hetu = torch.from_numpy(output).argmax(dim=1, keepdim=True)

        correct_hetu += preds_hetu.eq(y_torch.view_as(preds_hetu)).sum().item()

    accuracy_torch = correct_torch / len(test_loader.dataset) * 100

    accuracy_hetu = correct_hetu / len(test_loader.dataset) * 100

    print("ACC:", accuracy_torch, " ", accuracy_hetu)

    torch_model = torch_model.cpu()




 


def parellel_mnist():
    epoches = 50
    trainset, testset = naive_mnist()
    print(torch.cuda.is_available())

    torch_model = Torch_RNN()

    hetu_model = Hetu_RNN()

    # diffs, ratios = [], []
    # w1s, w2s = [], []
    # for w1, w2 in zip(hetu_model.parameters(), torch_model.parameters()):
    #     w1 = w1.numpy(force=True)
    #     w2 = w2.cpu().detach().numpy()
    #     # print(w1.shape, " ", w2.shape)
    #     assert w1.shape == w2.shape
    #     diff = np.abs(w1 - w2).sum()
    #     ratio = (np.abs((w1 - w2) / w2)).sum()
    #     diffs.append(diff)
    #     ratios.append(ratio)
    #     w1s.append(w1.sum())
    #     w2s.append(w2.sum())
    #     print(w1.shape, " ", diff, " ", ratio, " ", w1.flat[:10], " ", w2.flat[:10])
    # hetu_model = Hetu_CNN()


    for epoch in range(epoches):
        print("\n================ epoch: {} ================".format(epoch))
        # pa_test(hetu_model, torch_model, testset)
        pa_train(hetu_model, torch_model, epoch, trainset)

if __name__ == "__main__":
    # torch_mnist()
    #hetu_mnist()
    hetu_mnist_ex()
    # parellel_mnist()
    # unittest.main()