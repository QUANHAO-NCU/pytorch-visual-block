import torch
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channel)

        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.active = nn.LeakyReLU()

        self.extra = nn.Sequential()
        if out_channel != in_channel:
            self.extra = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.active(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.active(out)
        identity = self.extra(identity)
        out = out + identity
        out = self.active(out)
        return out


"""
编码器
或者叫做特征提取器
"""


class Encoder(nn.Module):
    def __init__(self, dataset='MNIST'):
        super(Encoder, self).__init__()
        self.mode = dataset
        # 针对CIFAR-10数据集，用最大池化将32*32 --> 28*28
        self.preprocessCIFAR = nn.MaxPool2d(kernel_size=5, stride=1, padding=0)
        # 针对MNIST数据集，将通道数从1通过复制扩展为3
        self.preprocessMNIST = lambda x: x.repeat(1, 3, 1, 1)
        self.blk1 = ResBlock(3, 16, 1)

    def forward(self, x):
        if self.mode == 'CIFAR-10':
            x = self.preprocessCIFAR(x)
        if self.mode == 'MNIST':
            x = self.preprocessMNIST(x)
        assert x.shape == torch.Size([32, 3, 28, 28])

        pass


"""
解码器
或者叫做生成器/重建器
"""


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

    def forward(self, x):
        pass


"""
分类器
"""


class Classifier(nn.Module):
    def __init__(self, in_dimension: int, out_dimension: int):
        super(Classifier, self).__init__()
        self.in_dim = in_dimension
        self.out_dim = out_dimension
        assert self.in_dim > 512
        self.process = nn.Sequential(
            nn.InstanceNorm1d(self.in_dim),
            nn.Linear(self.in_dim, 512),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 256),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(256, self.out_dim)
        )

    def forward(self, x):
        return self.process(x)


if __name__ == '__main__':
    mnistImage = torch.randn((32, 1, 28, 28))
    cifarImage = torch.randn((32, 3, 32, 32))
    p1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=0)
    p2 = lambda x: x.repeat(1, 3, 1, 1)
    p1 = p1(cifarImage)
    p2 = p2(mnistImage)
    assert p1.shape == torch.Size([32, 3, 28, 28])
    print(p1.shape, p2.shape)
