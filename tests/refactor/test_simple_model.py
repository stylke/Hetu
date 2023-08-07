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

def read_image(path):
    with gzip.open(path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        img = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    return img

def read_label(path):
    with gzip.open(path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        lab = np.frombuffer(f.read(), dtype=np.uint8)
        # print(lab[1])
    return lab
           
def load_mnist(path):
    x_train_path = path + 'train-images-idx3-ubyte.gz'
    y_train_path = path + 'train-labels-idx1-ubyte.gz'
    x_test_path = path + 't10k-images-idx3-ubyte.gz'
    y_test_path = path + 't10k-labels-idx1-ubyte.gz'

    x_train = read_image(x_train_path)
    print(x_train.shape)

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

def naive_cifar():
    train = torchvision.datasets.CIFAR10(root="./data", train=True, download=True,
                                       transform=torchvision.transforms.Compose([
                                                torchvision.transforms.Resize(32),
                                                torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    train_loader = DataLoader(train, batch_size=BATCHSIZE)
    test = torchvision.datasets.CIFAR10(root="./data", train=False, download=True,
                                      transform=torchvision.transforms.Compose([
                                                torchvision.transforms.Resize(32),
                                                torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    test_loader = DataLoader(test, batch_size=BATCHSIZE) 

    return train_loader, test_loader

class Torch_CNN(nn.Module):
    def __init__(self):
        super(Torch_CNN, self).__init__()
 
        # 卷积层
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1))
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
 
        # Dropout层
        self.dropout1 = torch.nn.Dropout(0.25)
        self.dropout2 = torch.nn.Dropout(0.5)
 
        # 全连接层
        self.fc1 = torch.nn.Linear(9216, 128)
        self.fc2 = torch.nn.Linear(128, 10)

        self.batchsize = BATCHSIZE

        self.norm = torch.nn.InstanceNorm2d(1, eps=1e-5)

        self.norm2 = torch.nn.InstanceNorm2d(32, eps=1e-5)
 
    def forward(self, x):
         
        # out = self.norm(x)
        out = x
        # [b, 1, 28, 28] => [b, 32, 26, 26]
        out = self.conv1(out)
        out = F.relu(out)
        # out = self.norm2(out)
         
        # [b, 32, 26, 26] => [b, 64, 24, 24]
        out = self.conv2(out)
        out = F.relu(out)
 
        # [b, 64, 24, 24] => [b, 64, 12, 12]
        out = F.max_pool2d(out, 2)
        # out = self.dropout1(out)
         
        # [b, 64, 12, 12] => [b, 64 * 12 * 12] => [b, 9216]
        # out = torch.flatten(out, 1)
        out = torch.reshape(out, (-1, 9216))
         
        # [b, 9216] => [b, 128]
        out = self.fc1(out)
        out = F.relu(out)
 
        # [b, 128] => [b, 10]
        # out = self.dropout2(out)
        out = self.fc2(out)
        # output = F.log_softmax(out, dim=1)
        # output = F.softmax(out, dim=1)
        # out = torch.sigmoid(out)
 
        return out

class Hetu_CNN(hetu.nn.Module):
    def __init__(self):
        super(Hetu_CNN, self).__init__()
 
        # 卷积层
        self.conv1 = hetu.nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1))
        self.conv2 = hetu.nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
 
        # 全连接层
        self.fc1 = hetu.nn.Linear(9216, 128)
        self.fc2 = hetu.nn.Linear(128, 10)

        self.batchsize = BATCHSIZE
 
    def forward(self, x):
        
        # [b, 1, 28, 28] => [b, 32, 26, 26]
        out = self.conv1(x)

         
        # [b, 32, 26, 26] => [b, 64, 24, 24]
        out = self.conv2(out)
        out = hetu.relu(out)
 
        # [b, 64, 24, 24] => [b, 64, 12, 12]
        out = hetu.maxpool(out, 2, 2, 0, 2)
        # out = self.dropout1(out)
         
        # [b, 64, 12, 12] => [b, 64 * 12 * 12] => [b, 9216]
        out = hetu.reshape(out, [out.shape[0], 9216])
         
        # [b, 9216] => [b, 128]
        out = self.fc1(out)
        out = hetu.relu(out)
 
        # [b, 128] => [b, 10]
        # out = self.dropout2(out)
        out = self.fc2(out)
 
        # out = hetu.sigmoid(out)
 
        return out

class HetuCNN(hetu.nn.Module):
    
    def __init__(self, weights):
        super(HetuCNN, self).__init__()

        with hetu.graph("define_and_run"):

            self.dropout1 = hetu.nn.Dropout(0.25)
            self.dropout2 = hetu.nn.Dropout(0.5)
    
            self.register_parameter(
                    f"filter1", 
                    hetu.Tensor(weights[0], requires_grad=True))
            self.register_parameter(
                    f"bias1", 
                    hetu.Tensor(weights[1], requires_grad=True))
            self.register_parameter(
                    f"filter2", 
                    hetu.Tensor(weights[2], requires_grad=True))
            self.register_parameter(
                    f"bias2", 
                    hetu.Tensor(weights[3], requires_grad=True))
            self.register_parameter(
                    f"matmul1", 
                    hetu.Tensor(weights[4], requires_grad=True))
            self.register_parameter(
                    f"bias3", 
                    hetu.Tensor(weights[5], requires_grad=True))
            self.register_parameter(
                    f"matmul2", 
                    hetu.Tensor(weights[6], requires_grad=True))
            self.register_parameter(
                    f"bias4", 
                    hetu.Tensor(weights[7], requires_grad=True))
            
            self.norm = hetu.nn.InstanceNorm(1, eps=1e-5)

            self.norm2 = hetu.nn.InstanceNorm(32, eps=1e-5)
    
    def forward(self, x):
        with hetu.graph("define_and_run"):
        
            # out = self.norm(x)
            out = x
            f = self.__getattr__(f"filter1")
            b = self.__getattr__(f"bias1")
            out = hetu.conv2d(out, f, b, 0, 1)
            out = hetu.relu(out)
            # out = self.norm2(out)
            
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

class TorchCNN(torch.nn.Module):
    
    def __init__(self, weights):
        super(TorchCNN, self).__init__()

        # self.dropout1 = torch.nn.Dropout(0.25)
        # self.dropout2 = torch.nn.Dropout(0.5)
 
        self.register_parameter(
                f"filter1", 
                torch.nn.Parameter(torch.tensor(weights[0])))
        self.register_parameter(
                f"bias1", 
                torch.nn.Parameter(torch.tensor(weights[1])))
        self.register_parameter(
                f"filter2", 
                torch.nn.Parameter(torch.tensor(weights[2])))
        self.register_parameter(
                f"bias2", 
                torch.nn.Parameter(torch.tensor(weights[3])))
        self.register_parameter(
                f"matmul1", 
                torch.nn.Parameter(torch.tensor(weights[4])))
        self.register_parameter(
                f"bias3", 
                torch.nn.Parameter(torch.tensor(weights[5])))
        self.register_parameter(
                f"matmul2", 
                torch.nn.Parameter(torch.tensor(weights[6])))
        self.register_parameter(
                f"bias4", 
                torch.nn.Parameter(torch.tensor(weights[7])))
        
        self.norm = torch.nn.InstanceNorm2d(1, eps=1e-5)

        self.norm2 = torch.nn.InstanceNorm2d(32, eps=1e-5)
    
    def forward(self, x):
        # out = self.norm(x)
        out = x
        f = self.__getattr__(f"filter1")
        b = self.__getattr__(f"bias1")
        out = torch.conv2d(out, f, b, stride = 1, padding = 0)
        out = torch.relu(out)
        # out = self.norm2(out)
         
        f = self.__getattr__(f"filter2")
        b = self.__getattr__(f"bias2")
        out = torch.conv2d(out, f, b, stride = 1, padding = 0)
        out = torch.relu(out)

        maxpool2d = torch.nn.MaxPool2d(2, 2, 0)
        out = maxpool2d(out)
        # # out = self.dropout1(out)
         
        out = torch.reshape(out, [out.shape[0], 9216])
         
        f = self.__getattr__(f"matmul1")
        b = self.__getattr__(f"bias3")
        out = torch.matmul(out, f) + b
        out = torch.relu(out)
 
        # # [b, 128] => [b, 10]
        # # out = self.dropout2(out)
        # # out = self.fc2(out)
        f = self.__getattr__(f"matmul2")
        b = self.__getattr__(f"bias4")
        out = torch.matmul(out, f) + b
        
 
        # out = torch.softmax(out, dim = 1)
        # out = torch.sigmoid(out)
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
 
        if step % 1000 == 0:
            print('Epoch: {}, Step {}, Loss: {}'.format(epoch, step, loss))

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

def hetu_train(model, epoch, train_loader, optimizer):

    for step, (x, y) in enumerate(train_loader):
        if (step < 1):
            model = model
            x1 = x.numpy()
            y1 = y.numpy()
            x0 = hetu.from_numpy(x1)
            y0 = hetu.from_numpy(y1)
            optimizer.zero_grad()
            
            output = model(x0)

            loss = hetu.nll_loss(output, y0)

            loss.backward()
          
            optimizer.step()

            if step % 1000 == 0:
                print('Epoch: {}, Step {}, Loss: {}'.format(epoch, step, loss))

def hetu_test(model, test_loader):
 
    correct = 0
 
    for x, y in test_loader:

        model = model
        x1, y1 = x.numpy(), y.numpy()
        x0 = hetu.from_numpy(x1)
        y0 = torch.from_numpy(y1)

        output = model(x0).numpy(force=True)

        pred = torch.from_numpy(output).argmax(dim=1, keepdim=True)

        correct += pred.eq(y0.view_as(pred)).sum().item()

    accuracy = correct / len(test_loader.dataset) * 100

    print("Test Accuracy: {}%".format(accuracy))

def hetu_train_ex(hetu_model, epoch, train_loader):

    x = hetu.placeholder(hetu.float32, shape=[-1, 1, 28, 28])
    y = hetu.placeholder(hetu.float32, shape=[-1, 10])
    pred = hetu_model(x)
    loss = hetu.softmax_cross_entropy(pred, y)
    # probs = hetu.sigmoid(pred)
    # loss = hetu.binary_cross_entropy(probs, y, "mean")

    optimizer = hetu.optim.SGD(lr=0.001)
    train_op = optimizer.minimize(loss)

    executor = hetu.DARExecutor("cuda:0")

    for step, (x0, y0) in enumerate(train_loader):
        x1 = x0.numpy()
        y1 = F.one_hot(y0, num_classes = 10).numpy().astype(np.float32)
        loss_val, _ = executor.run([loss, train_op], feed_dict={
            x: x1, 
            y: y1, 
        })

        if step % 100 == 0:
            print('Epoch: {}, Step {}, Loss: {}'.format(epoch, step, loss_val))

def hetu_test_ex(model, test_loader):
    x = hetu.placeholder(hetu.float32, shape=[-1, 1, 28, 28])
    y = hetu.placeholder(hetu.float32, shape=[-1, 10])
    output = model(x)
    executor = hetu.DARExecutor("cuda:0")

    correct = 0
 
    for x0, y0 in test_loader:

        x1, y1 = x0.numpy(), F.one_hot(y0, num_classes = 10).numpy().astype(np.float32)

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
    learning_rate = 0.0001
    epoches = 5
    trainset, testset = naive_mnist()
    network = Torch_CNN()  
    print(torch.cuda.is_available())
    optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate)  # 优化器
    for epoch in range(epoches):
        print("\n================ epoch: {} ================".format(epoch))
        train(network, epoch, trainset, optimizer)
        test(network, testset)

def hetu_mnist():
    learning_rate = 0.0001
    epoches = 5
    trainset, testset = naive_mnist()
    network = Hetu_CNN()  
    print(torch.cuda.is_available())
    optimizer = hetu.optim.SGD(network.parameters(), lr=learning_rate)  # 优化器
    for epoch in range(epoches):
        print("\n================ epoch: {} ================".format(epoch))
        hetu_train(network, epoch, trainset, optimizer)
        hetu_test(network, testset)

def hetu_mnist_ex():
    epoches = 5
    trainset, testset = naive_mnist()
    network = Hetu_CNN()  
    print(torch.cuda.is_available())
      # 优化器
    for epoch in range(epoches):
        print("\n================ epoch: {} ================".format(epoch))
        hetu_train_ex(network, epoch, trainset)
        hetu_test_ex(network, testset)


def pa_train(hetu_model, torch_model, example_model, epoch, train_loader):
    
    x = hetu.placeholder(hetu.float32, shape=[-1, 1, 28, 28])
    y = hetu.placeholder(hetu.float32, shape=[-1, 10])
    pred = hetu_model(x)
    with hetu.graph("define_and_run"):
        loss = hetu.softmax_cross_entropy(pred, y)
    # probs = hetu.sigmoid(pred)
    # loss = hetu.binary_cross_entropy(probs, y, "mean")

        optimizer = hetu.SGDOptimizer(lr=0.001, momentum = 0.0)
        train_op = optimizer.minimize(loss)

    criterion_torch = torch.nn.CrossEntropyLoss()
    optimizer_torch = torch.optim.SGD(torch_model.parameters(), lr=0.001)

    ex_criterion_torch = torch.nn.CrossEntropyLoss()
    ex_optimizer_torch = torch.optim.SGD(example_model.parameters(), lr=0.001)

    w1s = []
    for w1 in hetu_model.parameters():
        w1 = w1.numpy(force=True)
        w1s.append(w1.sum())
    print(w1s)

    w1s = []
    for w1 in hetu_model.parameters():
        w1 = w1.numpy(force=True)
        w1s.append(w1.sum())
    print(w1s)

    for step, (x0, y0) in enumerate(train_loader):

        x_val = x0.numpy().astype(np.float32)
        y_val = y0.numpy().astype(np.int64)

        y_onehot = torch.nn.functional.one_hot(torch.from_numpy(y_val), num_classes = 10).numpy().astype(np.float32)

        with hetu.graph("define_and_run"):
            loss_val, _ = loss.graph.run([loss, train_op], feed_dict={
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

        ex_optimizer_torch.zero_grad()
        ex_preds_torch = example_model(x_torch.cuda())
        ex_loss_torch = ex_criterion_torch(ex_preds_torch, y_torch.cuda())
        ex_loss_torch.backward()
        ex_optimizer_torch.step()

        if step % 100 == 0:
            print('Epoch: {}, Step {}, Loss_t: {}, Loss_h:{}, Loss_ex:{}'.format(epoch, step, loss_torch, loss_val, ex_loss_torch))


        diffs, ratios = [], []
        w1s, w2s = [], []
        for w1, w2 in zip(hetu_model.parameters(), torch_model.parameters()):
            w1 = w1.numpy(force=True)
            w2 = w2.cpu().detach().numpy()
            assert w1.shape == w2.shape
            diff = np.abs(w1 - w2).sum()
            ratio = (np.abs((w1 - w2) / w2)).sum()
            diffs.append(diff)
            ratios.append(ratio)
            w1s.append(w1.sum())
            w2s.append(w2.sum())
        # print(w1s, " ", w2s)
        # print(diffs, ratios)

def pa_test(hetu_model, torch_model, example_model, test_loader):
   
    x = hetu.placeholder(hetu.float32, shape=[-1, 1, 28, 28])
    y = hetu.placeholder(hetu.float32, shape=[-1, 10])
    pred = hetu_model(x)
    # probs = hetu.sigmoid(pred)
    # loss = hetu.binary_cross_entropy(probs, y, "mean")


    correct_torch = 0
    correct_hetu = 0
    correct_example = 0

    for step, (x0, y0) in enumerate(test_loader):

        x_val = x0.numpy().astype(np.float32)
        y_val = y0.numpy().astype(np.int64)

        y_onehot = torch.nn.functional.one_hot(torch.from_numpy(y_val), num_classes = 10).numpy().astype(np.float32)

        with hetu.graph("define_and_run"):
            pred_val = pred.graph.run([pred], feed_dict={
                x: x_val, 
                y: y_onehot, 
            })

        x_torch = torch.from_numpy(x_val)
        y_torch = torch.from_numpy(y_val)

        ex_preds_torch = example_model(x_torch.cuda())

        ex_preds_torch = ex_preds_torch.argmax(dim=1, keepdim=True).cpu()
 
        correct_example += ex_preds_torch.eq(y_torch.view_as(ex_preds_torch)).sum().item()

        preds_torch = torch_model(x_torch.cuda())

        preds_torch = preds_torch.argmax(dim=1, keepdim=True).cpu()
 
        correct_torch += preds_torch.eq(y_torch.view_as(preds_torch)).sum().item()

        output = pred_val[0].numpy(force=True)

        preds_hetu = torch.from_numpy(output).argmax(dim=1, keepdim=True)

        correct_hetu += preds_hetu.eq(y_torch.view_as(preds_hetu)).sum().item()

    accuracy_torch = correct_torch / len(test_loader.dataset) * 100

    accuracy_hetu = correct_hetu / len(test_loader.dataset) * 100

    accuracy_ex = correct_example / len(test_loader.dataset) * 100

    print("ACC:", accuracy_torch, " ", accuracy_hetu, " ", accuracy_ex)




 


def parellel_mnist():
    epoches = 20
    trainset, testset = naive_mnist()
    print(torch.cuda.is_available())
    example = Torch_CNN()
    weights_np = [
                example.conv1.weight.detach().numpy(),
                example.conv1.bias.detach().numpy(),
                example.conv2.weight.detach().numpy(),
                example.conv2.bias.detach().numpy(),
                example.fc1.weight.detach().transpose(0, 1).contiguous().numpy(),
                example.fc1.bias.detach().numpy(),
                example.fc2.weight.detach().transpose(0, 1).contiguous().numpy(),
                example.fc2.bias.detach().numpy(),]

    hetu_model = HetuCNN(weights_np)
    # hetu_model = Hetu_CNN()

    torch_model = TorchCNN(weights_np).cuda()

    example_model = example.cuda()

    for epoch in range(epoches):
        print("\n================ epoch: {} ================".format(epoch))
        pa_test(hetu_model, torch_model, example_model, testset)
        pa_train(hetu_model, torch_model, example_model, epoch, trainset)

if __name__ == "__main__":
    # torch_mnist()
    #hetu_mnist()
    # hetu_mnist_ex()
    parellel_mnist()
    # unittest.main()