import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        #  [bz,w,h,ch]
        #  原网络应用于图像大小为28*28的，现在调整为224*224 ，对应ImageNet挑战赛的规格
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.pool2 = nn.MaxPool2d(2, 2)
        # self.fc1 = nn.Linear(32 * 5 * 5, 120)
        self.fc1 = nn.Linear(32 * 53 * 53, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # input(3, 32, 32) output(16, 28, 28)
        x = self.pool1(x)  # output(16, 14, 14)
        x = F.relu(self.conv2(x))  # output(32, 10, 10)
        x = self.pool2(x)  # output(32, 5, 5)
        # x = x.view(-1, 32 * 5 * 5)  # output(32*5*5)
        x = x.view(-1, 32 * 53 * 53)  # output(32*5*5)
        x = F.relu(self.fc1(x))  # output(120)
        x = F.relu(self.fc2(x))  # output(84)
        x = self.fc3(x)  # output(10)
        return x


if __name__ == '__main__':
    device = torch.device('cuda:0')
    model = LeNet(2)
    a = torch.randn((1, 3, 224, 224))
    b = model(a)
    print(b.shape)
