import torch
import torch.nn as nn


class Q(nn.Module):
    def __init__(self, num_classes=5):
        super(Q, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,bias=False)
        self.relu = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.conv5 = nn.Conv2d(256, 256, kernel_size=3)
        self.bn5 = nn.BatchNorm2d(256)

        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(135424, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.bn2(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.bn3(x)

        x = self.conv4(x)
        x = self.relu(x)
        x = self.bn4(x)

        x = self.conv5(x)
        x = self.relu(x)
        x = self.bn5(x)

        x = x.view(-1, 135424)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    device = torch.device('cuda:0')
    model = Q(2)
    a = torch.randn((1, 3, 224, 224))
    b = model(a)
    print(b.shape)
