import numpy as np
import torch
from torch import nn
from Dataset import myDataset
from matplotlib import pyplot as plt
from Tools import *
import torch.nn.functional as func


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channel)

        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.active = nn.LeakyReLU(0.2, inplace=True)

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


class MNISTEncoder(nn.Module):
    """
    编码器
    或者叫做特征提取器
    MNIST HW：[28,28] --> [28,28] --> [14,14] --> [14,14] -->[7,7]
    结束时宽高减少到原来的四分之一
    Channel 3 --> 16 --> 32 --> 64 --> 64 --> 128
    """

    def __init__(self, feature_dim):
        super(MNISTEncoder, self).__init__()
        self.fea_dim = feature_dim
        # 以MNIST数据集为例
        # [b_s,3,28,28] --> [b_s,16,28,28]
        self.blk1 = ResBlock(3, 16, 1)
        # [b_s,16,28,28] --> [b_s,32,14,14]
        self.blk2 = ResBlock(16, 32, 2)
        # [b_s,32,14,14] --> [b_s,64,14,14]
        self.blk3 = ResBlock(32, 64, 1)
        # [b_s,64,14,7] --> [b_s,128,14,7]
        self.blk4 = ResBlock(64, 128, 2)
        self.linear = nn.Linear(128 * 7 * 7, feature_dim)
        self.active = nn.Sigmoid()

    def forward(self, x_in):
        x = self.blk1(x_in)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)
        x = x.view(x.size(0), int(len(x.view(-1)) / len(x)))
        x = self.linear(x)
        x = self.active(x)
        return x


class MNISTDecoder(nn.Module):
    """
    解码器
    或者叫做生成器/重建器
    MNIST HW：[7,7] --> [14,14] --> [14,14] --> [28,28] --> [28,28]
    结束时宽高扩张到原来的四倍
    Channel 128 --> 64 --> 64 --> 32 --> 64 --> 3
    """

    def __init__(self, feature_dim):
        super(MNISTDecoder, self).__init__()
        self.fea_dim = feature_dim
        self.linear1 = nn.Linear(self.fea_dim, 128 * 7 * 7)
        # 以MNIST数据集为例
        # [b_s,128,7,7] --> [b_s,64,14,14]
        self.convTrans1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0)
        self.bn1 = nn.BatchNorm2d(64)
        self.active1 = nn.LeakyReLU(0.1, inplace=True)
        # [b_s,64,14,14] --> [b_s,64,14,14]
        self.convTrans2 = nn.ConvTranspose2d(64, 64, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(64)
        self.active2 = nn.LeakyReLU(0.1, inplace=True)
        # [b_s,64,14,14] --> [b_s,32,14,14]
        self.convTrans3 = nn.ConvTranspose2d(64, 32, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(32)
        self.active3 = nn.LeakyReLU(0.1, inplace=True)
        # [b_s,32,14,14] --> [b_s,16,28,28]
        self.convTrans4 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2, padding=0)
        self.bn4 = nn.BatchNorm2d(16)
        self.active4 = nn.LeakyReLU(0.1, inplace=True)
        # [b_s,16,28,28] --> [b_s,3,28,28]
        self.convTrans5 = nn.ConvTranspose2d(16, 3, kernel_size=1, stride=1, padding=0)
        self.bn5 = nn.BatchNorm2d(3)
        self.active5 = nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = x.view(x.size(0), 128, 7, 7)
        x = self.convTrans1(x)
        x = self.bn1(x)
        x = self.active1(x)

        x = self.convTrans2(x)
        x = self.bn2(x)
        x = self.active2(x)

        x = self.convTrans3(x)
        x = self.bn3(x)
        x = self.active3(x)

        x = self.convTrans4(x)
        x = self.bn4(x)
        x = self.active4(x)

        x = self.convTrans5(x)
        x = self.bn5(x)
        x = self.active5(x)
        return x


class Discriminator(nn.Module):
    """
    判别器
    MNIST HW：[7,7] --> [14,14] --> [14,14] --> [28,28] --> [28,28]
    """

    def __init__(self):
        super(Discriminator, self).__init__()
        # 以MNIST数据集为例
        # [b_s,3,28,28] --> [b_s,16,14,14]
        self.conv1 = nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.active1 = nn.LeakyReLU(0.2, inplace=True)
        # [b_s,16,14,14] --> [b_s,32,7,7]
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.active2 = nn.LeakyReLU(0.2, inplace=True)
        # [b_s,32,7,7] --> [b_s,64,7,7]
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=3)
        self.bn3 = nn.BatchNorm2d(64)
        self.active3 = nn.LeakyReLU(0.2, inplace=True)
        # [b_s,64,7,7] --> [b_s,128,5,5]
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        self.bn4 = nn.BatchNorm2d(128)
        self.active4 = nn.LeakyReLU(0.2, inplace=True)
        # [b_s,16,28,28] --> [b_s,3,28,28]
        self.conv5 = nn.ConvTranspose2d(128, 2, kernel_size=1, stride=1, padding=0)
        self.bn5 = nn.BatchNorm2d(2)
        self.out = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.active1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.active2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.active3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.active4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.out(x)
        return x


class CIFAR10Encoder(nn.Module):
    """
    编码器
    或者叫做特征提取器
    CIFAR10 HW：[32,32] --> [32,32] --> [16,16] --> [16,16] -->[8,8]
    结束时宽高减少到原来的四分之一
    Channel 3 --> 16 --> 32 --> 64 --> 64 --> 128
    """

    def __init__(self, feature_dim):
        super(CIFAR10Encoder, self).__init__()
        self.fea_dim = feature_dim
        # 以MNIST数据集为例
        # [b_s,3,28,28] --> [b_s,16,28,28]
        self.blk1 = ResBlock(3, 16, 1)
        # [b_s,16,28,28] --> [b_s,32,14,14]
        self.blk2 = ResBlock(16, 32, 2)
        # [b_s,32,14,14] --> [b_s,64,14,14]
        self.blk3 = ResBlock(32, 64, 1)
        # [b_s,64,14,7] --> [b_s,128,14,7]
        self.blk4 = ResBlock(64, 128, 2)
        self.linear = nn.Linear(128 * 8 * 8, feature_dim)
        self.active = nn.Sigmoid()

    def forward(self, x_in):
        x = self.blk1(x_in)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)
        x = x.view(x.size(0), int(len(x.view(-1)) / len(x)))
        x = self.linear(x)
        x = self.active(x)
        return x


class CIFAR10Decoder(nn.Module):
    """
    解码器
    或者叫做生成器/重建器
    MNIST HW：[8,8] --> [16,16] --> [16,16] --> [32,32] --> [32,32]
    结束时宽高扩张到原来的四倍
    Channel 128 --> 64 --> 64 --> 32 --> 64 --> 3
    """

    def __init__(self, feature_dim):
        super(CIFAR10Decoder, self).__init__()
        self.fea_dim = feature_dim
        self.linear1 = nn.Linear(self.fea_dim, 128 * 8 * 8)
        # 以MNIST数据集为例
        # [b_s,128,7,7] --> [b_s,64,14,14]
        self.convTrans1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0)
        self.bn1 = nn.BatchNorm2d(64)
        self.active1 = nn.LeakyReLU(0.1, inplace=True)
        # [b_s,64,14,14] --> [b_s,64,14,14]
        self.convTrans2 = nn.ConvTranspose2d(64, 64, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(64)
        self.active2 = nn.LeakyReLU(0.1, inplace=True)
        # [b_s,64,14,14] --> [b_s,32,14,14]
        self.convTrans3 = nn.ConvTranspose2d(64, 32, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(32)
        self.active3 = nn.LeakyReLU(0.1, inplace=True)
        # [b_s,32,14,14] --> [b_s,16,28,28]
        self.convTrans4 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2, padding=0)
        self.bn4 = nn.BatchNorm2d(16)
        self.active4 = nn.LeakyReLU(0.1, inplace=True)
        # [b_s,16,28,28] --> [b_s,3,28,28]
        self.convTrans5 = nn.ConvTranspose2d(16, 3, kernel_size=1, stride=1, padding=0)
        self.bn5 = nn.BatchNorm2d(3)
        self.active5 = nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = x.view(x.size(0), 128, 8, 8)
        x = self.convTrans1(x)
        x = self.bn1(x)
        x = self.active1(x)

        x = self.convTrans2(x)
        x = self.bn2(x)
        x = self.active2(x)

        x = self.convTrans3(x)
        x = self.bn3(x)
        x = self.active3(x)

        x = self.convTrans4(x)
        x = self.bn4(x)
        x = self.active4(x)

        x = self.convTrans5(x)
        x = self.bn5(x)
        x = self.active5(x)
        return x



class Scoring(nn.Module):
    def __init__(self, feature_dim):
        super(Scoring, self).__init__()
        self.fea_dim = feature_dim
        self.linear1 = nn.Linear(self.fea_dim, 256)
        self.active1 = nn.LeakyReLU(0.2, inplace=True)
        self.linear2 = nn.Linear(256, 128)
        self.active2 = nn.LeakyReLU(0.2, inplace=True)
        self.linear3 = nn.Linear(128, 1)
        self.active3 = nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.active1(x)
        x = self.linear2(x)
        x = self.active2(x)
        x = self.linear3(x)
        x = self.active3(x)
        return x


class Encoder(nn.Module):
    """
    编码器
    或者叫做特征提取器
    MNIST HW：[28,28] --> [28,28] --> [14,14] --> [14,14] -->[7,7]
    CIFAR10 HW：[32,32] --> [32,32] --> [16,16] --> [16,16] -->[8,8]
    结束时宽高减少到原来的四分之一
    Channel 3 --> 16 --> 32 --> 64 --> 64 --> 128
    """

    def __init__(self):
        super(Encoder, self).__init__()
        # self.active1 = nn.ReLU()
        # self.active2 = nn.LeakyReLU()
        # 以MNIST数据集为例
        # [b_s,3,28,28] --> [b_s,16,28,28]
        self.blk1 = ResBlock(3, 16, 1)
        # [b_s,16,28,28] --> [b_s,32,14,14]
        self.blk2 = ResBlock(16, 32, 2)
        # [b_s,32,14,14] --> [b_s,64,14,14]
        self.blk3 = ResBlock(32, 64, 1)
        # [b_s,64,14,14] --> [b_s,64,14,14]
        self.blk4 = ResBlock(64, 64, 1)
        # [b_s,64,14,7] --> [b_s,128,14,7]
        self.blk5 = ResBlock(64, 128, 2)

    def forward(self, x_in):
        x = self.blk1(x_in)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)
        x = self.blk5(x)
        feature_vectors = x.view(int(len(x)), int(len(x.view(-1)) / len(x)))
        # x = self.active1(x)
        # feature_vectors = self.active2(feature_vectors)
        return x, feature_vectors


class Decoder(nn.Module):
    """
    解码器
    或者叫做生成器/重建器
    MNIST HW：[7,7] --> [14,14] --> [14,14] --> [28,28] --> [28,28]
    CIFAR10 HW：[8,8] --> [16,16] --> [16,16] -->  [32,32] --> [32,32]
    结束时宽高扩张到原来的四倍
    Channel 128 --> 64 --> 64 --> 32 --> 64 --> 3
    """

    def __init__(self):
        super(Decoder, self).__init__()
        # 以MNIST数据集为例
        # [b_s,128,7,7] --> [b_s,64,14,14]
        self.convTrans1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0)
        self.bn1 = nn.BatchNorm2d(64)
        self.active1 = nn.LeakyReLU(0.2)
        # [b_s,64,14,14] --> [b_s,64,14,14]
        self.convTrans2 = nn.ConvTranspose2d(64, 64, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(64)
        self.active2 = nn.LeakyReLU(0.1)
        # [b_s,64,14,14] --> [b_s,32,14,14]
        self.convTrans3 = nn.ConvTranspose2d(64, 32, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(32)
        self.active3 = nn.LeakyReLU(0.1)
        # [b_s,32,14,14] --> [b_s,16,28,28]
        self.convTrans4 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2, padding=0)
        self.bn4 = nn.BatchNorm2d(16)
        self.active4 = nn.ReLU()
        # [b_s,16,28,28] --> [b_s,3,28,28]
        self.convTrans5 = nn.ConvTranspose2d(16, 3, kernel_size=1, stride=1, padding=0)
        self.active5 = nn.Sigmoid()

    def forward(self, x):
        x = self.convTrans1(x)
        x = self.bn1(x)
        x = self.active1(x)

        x = self.convTrans2(x)
        x = self.bn2(x)
        x = self.active2(x)

        x = self.convTrans3(x)
        x = self.bn3(x)
        x = self.active3(x)

        x = self.convTrans4(x)
        x = self.bn4(x)
        x = self.active4(x)

        x = self.convTrans5(x)
        x = self.active5(x)
        return x


class Classifier(nn.Module):
    """
    分类器
    """

    def __init__(self, in_features):
        super(Classifier, self).__init__()
        #
        self.ihm = nn.InstanceNorm1d(in_features)
        self.linear1 = nn.Linear(in_features, 2048)
        self.dropout = nn.Dropout(0.5)
        self.leakyRelu = nn.LeakyReLU(0.1)
        self.linear2 = nn.Linear(2048, 256)
        self.active = nn.ReLU()
        self.linear3 = nn.Linear(256, 2)
        self.active = nn.Sigmoid()

    def forward(self, x):
        # x[batch_size,6272]-->x[batch_size,6272,1]
        x = torch.unsqueeze(x, dim=2)
        x = self.ihm(x)
        x = torch.squeeze(x, dim=2)
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.leakyRelu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        x = self.active(x)
        x = self.linear3(x)
        x = self.active(x)
        return x


class myMSELoss(nn.Module):
    """
    效果不好
    图像使用uint8格式应该会更好
    """

    def __init__(self):
        super(myMSELoss, self).__init__()

    def forward(self, images1, images2):
        batch_size = images1.size(0)
        pixel_num = len(images1.view(-1)) / batch_size
        images1 = torch.mul(images1, 255)
        images2 = torch.pow(images2, 255)
        loss = 0
        for image_id in range(batch_size):
            image1 = images1[image_id]
            image2 = images2[image_id]
            image1 = image1.view(-1)
            image2 = image2.view(-1)
            image_loss = 0
            for pixel_id in range(len(image1)):
                dif = torch.sub(image1[pixel_id], image2[pixel_id])
                sqr = torch.pow(dif, 2)
                image_loss = torch.add(sqr, image_loss)
            image_loss = torch.div(image_loss, pixel_num)
            loss = torch.add(image_loss, loss)
        loss = torch.div(loss, batch_size)
        return loss


if __name__ == '__main__':
    feature_dim = 512
    x = torch.randn((32, 3, 32, 32))
    x = torch.sigmoid(x)
    e = MNISTEncoder(feature_dim)
    d = MNISTDecoder(feature_dim)
    s = Scoring(feature_dim)
    di = Discriminator()
    out = di(x)
    print(out.shape)
    print(out)
