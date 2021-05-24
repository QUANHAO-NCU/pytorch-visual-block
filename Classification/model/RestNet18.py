import torch
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, inplanes, outplanes, stride=1):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(outplanes)

        self.conv2 = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(outplanes)

        self.relu = nn.ReLU()

        self.extra = nn.Sequential()
        if outplanes != inplanes:
            self.extra = nn.Sequential(
                nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(outplanes)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        identity = self.extra(identity)
        out = out + identity
        # out = self.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, num_class=1000):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm2d(16)
        )
        self.relu = nn.ReLU()
        self.blk1 = ResBlock(16, 32, stride=3)
        self.blk2 = ResBlock(32, 64, stride=3)
        self.blk3 = ResBlock(64, 128, stride=2)
        self.blk4 = ResBlock(128, 256, stride=2)
        self.out = nn.Linear(256 * 3 * 3, num_class)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        return x


if __name__ == '__main__':
    device = torch.device('cuda:0')
    model = ResNet18(5)
    model.training = False
    a = torch.randn((1, 3, 224, 224))
    b = model(a)
    print(b.shape)
