import hetu
import hetu.nn as nn
import numpy as np
import torch
import unittest
import random
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


def naive_cifar():
    train = torchvision.datasets.CIFAR10(root="./data", train=True, download=True,
                                        transform=torchvision.transforms.Compose([
                                        torchvision.transforms.RandomCrop(32, padding=4),
                                        torchvision.transforms.RandomHorizontalFlip(),
                                        torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                        ]))
    train_loader = DataLoader(train, batch_size=BATCHSIZE)
    test = torchvision.datasets.CIFAR10(root="./data", train=False, download=True,
                                      transform=torchvision.transforms.Compose([
                                                torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
    test_loader = DataLoader(test, batch_size=BATCHSIZE) 

    train_dataset = []
    test_dataset = []
    for step, (x0, y0) in enumerate(train_loader):
        x1 = x0.numpy().astype(np.float32)
        y1 = F.one_hot(y0, num_classes = 10).numpy().astype(np.float32)
        train_dataset.append((x1, y1))

    for step, (x0, y0) in enumerate(test_loader):
        x1 = x0.numpy().astype(np.float32)
        y1 = y0.numpy().astype(np.float32)
        test_dataset.append((x1, y1))

    return train_dataset, test_dataset

class Torch_CNN(nn.Module):
    def __init__(self):
        super(Torch_CNN, self).__init__()
 
        # 卷积层
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=(5, 5), stride=(1, 1), padding=2)
        self.conv2 = torch.nn.Conv2d(32, 32, kernel_size=(5, 5), stride=(1, 1), padding=2)
        self.conv3 = torch.nn.Conv2d(32, 64, kernel_size=(5, 5), stride=(1, 1), padding=2)
 
        # 全连接层
        self.fc1 = torch.nn.Linear(1024, 64)
        self.fc2 = torch.nn.Linear(64, 10)

        self.batchsize = BATCHSIZE
 
    def forward(self, x):
         
        # [b, 3, 32, 32] => [b, 32, 16, 16]
        out = self.conv1(x)
        out = F.max_pool2d(out, 2)
         
        # [b, 32, 16, 16] => [b, 32, 8, 8]
        out = self.conv2(out)
        out = F.max_pool2d(out, 2)
 
        # [b, 32, 8, 8] => [b, 64, 4, 4]
        out = self.conv3(out)
        out = F.max_pool2d(out, 2)
         
        # [b, 64, 4, 4] => [b, 64 * 4 * 4] => [b, 1024]
        out = torch.reshape(out, (-1, 1024))
         
        # [b, 1024] => [b, 64]
        out = self.fc1(out)
 
        # [b, 64] => [b, 10]
        out = self.fc2(out)
 
        return out

class Hetu_CNN(hetu.nn.Module):
    def __init__(self):
        super(Hetu_CNN, self).__init__()
 
        # 卷积层
        self.conv1 = hetu.nn.Conv2d(3, 32, kernel_size=(5, 5), stride=(1, 1), padding=2)
        self.conv2 = hetu.nn.Conv2d(32, 32, kernel_size=(5, 5), stride=(1, 1), padding=2)
        self.conv3 = hetu.nn.Conv2d(32, 64, kernel_size=(5, 5), stride=(1, 1), padding=2)
 
        # 全连接层
        self.fc1 = hetu.nn.Linear(1024, 64)
        self.fc2 = hetu.nn.Linear(64, 10)

        self.batchsize = BATCHSIZE
 
    def forward(self, x):
        
        out = self.conv1(x)
        out = hetu.maxpool(out, 2, 2, 0, 2)

        out = self.conv2(out)
        out = hetu.maxpool(out, 2, 2, 0, 2)
 
        out = self.conv3(out)
        out = hetu.maxpool(out, 2, 2, 0, 2)
        # out = self.dropout1(out)
         
        out = hetu.reshape(out, [out.shape[0], 1024])
         
        out = self.fc1(out)
 
        out = self.fc2(out)
 
        # out = hetu.sigmoid(out)
 
        return out

class HetuCNN(hetu.nn.Module):
    
    def __init__(self, weights):
        super(HetuCNN, self).__init__()

        self.dropout1 = hetu.nn.Dropout(0.25)
        self.dropout2 = hetu.nn.Dropout(0.5)
 
        self.register_parameter(
                f"filter1", 
                hetu.nn.Parameter(hetu.Tensor(weights[0], trainable=True)))
        self.register_parameter(
                f"bias1", 
                hetu.nn.Parameter(hetu.Tensor(weights[1], trainable=True)))
        self.register_parameter(
                f"filter2", 
                hetu.nn.Parameter(hetu.Tensor(weights[2], trainable=True)))
        self.register_parameter(
                f"bias2", 
                hetu.nn.Parameter(hetu.Tensor(weights[3], trainable=True)))
        self.register_parameter(
                f"matmul1", 
                hetu.nn.Parameter(hetu.Tensor(weights[4], trainable=True)))
        self.register_parameter(
                f"bias3", 
                hetu.nn.Parameter(hetu.Tensor(weights[5], trainable=True)))
        self.register_parameter(
                f"matmul2", 
                hetu.nn.Parameter(hetu.Tensor(weights[6], trainable=True)))
        self.register_parameter(
                f"bias4", 
                hetu.nn.Parameter(hetu.Tensor(weights[7], trainable=True)))
    
    def forward(self, x):

        f = self.__getattr__(f"filter1")
        b = self.__getattr__(f"bias1")
        out = hetu.conv2d(x, f, b, 0, 1)
        out = hetu.relu(out)
         
        f = self.__getattr__(f"filter2")
        b = self.__getattr__(f"bias2")
        out = hetu.conv2d(out, f, b, 0, 1)
        out = hetu.relu(out)
 
        out = hetu.maxpool(out, 2, 2, 0, 2)
        # # out = self.dropout1(out)
         
        out = hetu.reshape(out, [out.shape[0], 9216])
         
        f = self.__getattr__(f"matmul1")
        b = self.__getattr__(f"bias3")
        out = hetu.linear(out, f, b)
        out = hetu.relu(out)
 
        # # [b, 128] => [b, 10]
        # # out = self.dropout2(out)
        # # out = self.fc2(out)
        f = self.__getattr__(f"matmul2")
        b = self.__getattr__(f"bias4")
        out = hetu.linear(out, f, b)
 
        # out = hetu.softmax(out)
        # out = hetu.sigmoid(out)

        return out


class ResidualBlock_low(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock_low, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel, eps=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel, eps=0)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel, eps=0)
            )
 
    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Torch_ResNet(torch.nn.Module):
    
    def __init__(self, ResidualBlock, num_classes=10):
        super(Torch_ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64, eps=0),
            nn.ReLU(),
        )
        self.fc = nn.Linear(512, num_classes)
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)
 
    def forward(self, x):#3*32*32
        out = self.conv1(x)#64*32*32
        out = self.layer1(out)#64*32*32
        out = self.layer2(out)#128*16*16
        out = self.layer3(out)#256*8*8
        out = self.layer4(out)#512*4*4
        out = F.avg_pool2d(out, 4)#512*1*1
        out = out.view(out.size(0), -1)#512
        out = self.fc(out)
        return out

class Hetu_ResidualBlock(hetu.nn.Module):
    def __init__(self, inchannel, outchannel, stride=1, torchblock=None):
        super(Hetu_ResidualBlock, self).__init__()
        self.left = hetu.nn.Sequential(
            hetu.nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            hetu.nn.BatchNorm(outchannel, eps=0),
            hetu.nn.ReLU(),
            hetu.nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            hetu.nn.BatchNorm(outchannel, eps=0)
        )
        self.shortcut = hetu.nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = hetu.nn.Sequential(
                hetu.nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                hetu.nn.BatchNorm(outchannel, eps=0)
                )
        if (torchblock != None):
            self.set_by_torch(torchblock)
    
    def set_by_torch(self, torchblock):
        self.left.__getitem__(0).weight = hetu.nn.Parameter(hetu.Tensor(
        torchblock.left.__getitem__(0).weight.detach().numpy(), trainable=True))
        self.left.__getitem__(3).weight = hetu.nn.Parameter(hetu.Tensor(
        torchblock.left.__getitem__(3).weight.detach().numpy(), trainable=True))
        if len(torchblock.shortcut) > 0:
            self.shortcut.__getitem__(0).weight = hetu.nn.Parameter(hetu.Tensor(
                torchblock.shortcut.__getitem__(0).weight.detach().numpy(), trainable=True))
        
 
    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = hetu.relu(out)
        return out

class Hetu_ResNet(hetu.nn.Module):
    
    def __init__(self, ResidualBlock, num_classes=10, torchresnet=None):
        super(Hetu_ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = hetu.nn.Sequential(
            hetu.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            hetu.nn.BatchNorm(64, eps=0),
            hetu.nn.ReLU(),
        )
        # self.conv1 = hetu.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.ins1 = hetu.nn.BatchNorm(64)
        # self.relu1 = hetu.nn.ReLU()
        # )
        self.fc = hetu.nn.Linear(512, num_classes)
        if (torchresnet != None):
            self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1, torchblock=torchresnet.layer1)
            self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2, torchblock=torchresnet.layer2)
            self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2, torchblock=torchresnet.layer3)
            self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2, torchblock=torchresnet.layer4)
            self.set_by_torch(torchresnet)
        else:
            self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
            self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
            self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
            self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
    
    def set_by_torch(self, torchresnet):
        self.conv1.__getitem__(0).weight = hetu.nn.Parameter(hetu.Tensor(
                                           torchresnet.conv1.__getitem__(0).weight.detach().numpy(), trainable=True))
        self.fc.weight = hetu.nn.Parameter(hetu.Tensor(
                         torchresnet.fc.weight.detach().numpy(), trainable=True))
        self.fc.bias = hetu.nn.Parameter(hetu.Tensor(
                       torchresnet.fc.bias.detach().numpy(), trainable=True))
        

    def make_layer(self, block, channels, num_blocks, stride, torchblock=None):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        intr = 0
        for stride in strides:
            if torchblock!=None:
                layers.append(block(self.inchannel, channels, stride, torchblock.__getitem__(intr)))
                intr += 1
            else:
                layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return hetu.nn.Sequential(*layers)
    

 
    def forward(self, x):#3*32*32
        out = self.conv1(x)#64*32*32
        # out = self.ins1(out)
        # out = self.relu1(out)
        out = self.layer1(out)#64*32*32
        out = self.layer2(out)#128*16*16
        out = self.layer3(out)#256*8*8
        out = self.layer4(out)#512*4*4
        out = hetu.avgpool(out, 4, 4, 0, 4)#512*1*1
        out = hetu.reshape(out, [out.shape[0], 512])#512
        out = self.fc(out)
        return out
    

class Torch_SP(torch.nn.Module):
    
    def __init__(self, num_classes=10):
        super(Torch_SP, self).__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.fc = nn.Linear(3072, num_classes)
        self.norm = nn.LayerNorm(32, eps=0)

 
    def forward(self, x):#3*32*32
        out = self.conv(x)
        self.tmp2 = out
        self.tmp2.retain_grad()
        out = self.norm(self.tmp2)
        self.tmp = out
        self.tmp.retain_grad()
        out = self.tmp
        out = out.view(out.size(0), -1)#512
        out = self.fc(out)
        return out


class Hetu_SP(hetu.nn.Module):
    
    def __init__(self, num_classes = 10,torchresnet=None):
        super(Hetu_SP, self).__init__()
        self.conv = hetu.nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.fc = hetu.nn.Linear(3072, num_classes)
        self.norm = hetu.nn.LayerNorm(32, eps=0)
        if (torchresnet!=None):
            self.set_by_torch(torchresnet)
    
    def set_by_torch(self, torchresnet):
        self.conv.weight = hetu.nn.Parameter(hetu.Tensor(
                         torchresnet.conv.weight.detach().numpy(), trainable=True))
        self.fc.weight = hetu.nn.Parameter(hetu.Tensor(
                         torchresnet.fc.weight.detach().numpy(), trainable=True))
        self.fc.bias = hetu.nn.Parameter(hetu.Tensor(
                       torchresnet.fc.bias.detach().numpy(), trainable=True))
    

 
    def forward(self, x):#3*32*32
        out = self.conv(x)
        out = self.norm(out)#64*32*32
        out = hetu.reshape(out, [out.shape[0], 3072])#512
        out = self.fc(out)
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

    x = hetu.placeholder(hetu.float32, shape=[-1, 3, 32, 32])
    y = hetu.placeholder(hetu.float32, shape=[-1, 10])
    pred = hetu_model(x)
    # test_out = hetu_model.test(x)
    # test_out2 = hetu_model.test2(x)
    # test_out3 = hetu_model.test3(x)
    loss = hetu.softmax_cross_entropy(pred, y)
    # probs = hetu.sigmoid(pred)
    # loss = hetu.binary_cross_entropy(probs, y, "mean")

    optimizer = hetu.optim.SGD(lr=0.01)
    train_op = optimizer.minimize(loss)

    executor = hetu.DARExecutor("cuda:0")

    random.shuffle(train_loader)

    import time
    st = time.time()

    for step, (x1, y1) in enumerate(train_loader):

        loss_val, _ = executor.run([loss, train_op], feed_dict={
            x: x1, 
            y: y1, 
        })

        if step % 200 == 0 :
            # print("TESTOUT:", testtime[0].numpy(force=True).flat[:20], " ", testtime[0].numpy(force=True).shape)
            # print("TESTOUT2:", testtime2[0].numpy(force=True).flat[:20])
            # print("TESTOUT3:", testtime3[0].numpy(force=True).flat[:20])
            print('Epoch: {}, Step {}, Loss: {}, Loss_val: {}'.format(epoch, step, loss_val, loss_val.numpy(force=True)[0]))

    print("TIME:", time.time() - st)
    print(len(train_loader))

def hetu_test_ex(model, test_loader):
    x = hetu.placeholder(hetu.float32, shape=[-1, 3, 32, 32])
    y = hetu.placeholder(hetu.float32, shape=[-1, 10])
    output = model(x)
    executor = hetu.DARExecutor("cuda:0")

    correct = 0
    random.shuffle(test_loader)
 
    for x1, y1 in test_loader:

        out = executor.run([output], feed_dict={
            x: x1, 
        })
        
        out = out[0].numpy(force=True)

        pred = torch.from_numpy(out).argmax(dim=1, keepdim=True)

        correct += pred.eq(torch.from_numpy(y1).view_as(pred)).sum().item()

    accuracy = correct / 10000 * 100
    print("Test Accuracy: {}%".format(accuracy))

MNIST_PATH = "/root/hetu-pytest/datasets/mnist_dataset/"
def torch_mnist():
    learning_rate = 0.01
    epoches = 200
    trainset, testset = naive_cifar()
    # network = Torch_CNN() 
    network = Torch_ResNet(ResidualBlock_low) 
    print(torch.cuda.is_available())
    optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate)  # 优化器
    for epoch in range(epoches):
        print("\n================ epoch: {} ================".format(epoch))
        train(network, epoch, trainset, optimizer)
        test(network, testset)

def hetu_mnist_ex():
    epoches = 200
    trainset, testset = naive_cifar()
    network = Hetu_ResNet(Hetu_ResidualBlock)  
    print(torch.cuda.is_available())
      # 优化器
    for epoch in range(epoches):
        print("\n================ epoch: {} ================".format(epoch))
        hetu_train_ex(network, epoch, trainset)
        hetu_test_ex(network, testset)


def pa_train(hetu_model, torch_model, epoch, train_loader):
    
    x = hetu.placeholder(hetu.float32, shape=[-1, 3, 32, 32])
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
                diffs.appewxynd(diff)
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
    trainset, testset = naive_cifar()
    print(torch.cuda.is_available())

    torch_model = Torch_ResNet(ResidualBlock_low)

    hetu_model = Hetu_ResNet(Hetu_ResidualBlock, 10, torch_model)

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
        pa_test(hetu_model, torch_model, testset)
        pa_train(hetu_model, torch_model, epoch, trainset)

if __name__ == "__main__":
    # torch_mnist()
    #hetu_mnist()
    hetu_mnist_ex()
    # parellel_mnist()
    # unittest.main()