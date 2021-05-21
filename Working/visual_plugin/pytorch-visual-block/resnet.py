import torch
from torch import nn
from torch.nn import functional as F
from visual_block import visual_block


class ResBlk(nn.Module):
    """
    resnet block
    """

    def __init__(self, ch_in, ch_out, stride=1):
        """

        :param ch_in:
        :param ch_out:
        """
        super(ResBlk, self).__init__()

        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

        self.extra = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(ch_out)
        )
        if ch_out != ch_in:
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self, x):
        """
        :param x: [b, ch, h, w]
        :return:
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.extra(x) + out
        out = F.relu(out)
        return out


class ResNet18(nn.Module):

    def __init__(self):
        super(ResNet18, self).__init__()
        self.visual_block = visual_block(max_row=10, max_column=10)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )
        self.blk1 = ResBlk(64, 128, stride=1)
        self.blk2 = ResBlk(128, 256, stride=1)
        self.blk3 = ResBlk(256, 512, stride=2)
        self.blk4 = ResBlk(512, 512, stride=2)

        self.outlayer = nn.Linear(512 * 1 * 1, 10)

    def forward(self, x):
        """
        :param x:
        :return:
        """
        self.visual_block([x, {'mode': 'source_image', 'layer': 'source_image'}])
        x = F.relu(self.conv1(x))
        self.visual_block([x, {'mode': 'feature_map', 'layer': 'conv_1', 'channel_num': 10}])
        x = self.blk1(x)
        self.visual_block([x, {'mode': 'feature_map', 'layer': 'conv_2', 'channel_num': 10}])
        x = self.blk2(x)
        self.visual_block([x, {'mode': 'feature_map', 'layer': 'conv_3', 'channel_num': 10}])
        x = self.blk3(x)
        self.visual_block([x, {'mode': 'feature_map', 'layer': 'conv_4', 'channel_num': 10, 'end': True}])
        x = self.blk4(x)

        x = F.adaptive_avg_pool2d(x, [1, 1])
        x = x.view(x.size(0), -1)
        x = self.outlayer(x)

        return x


def main():
    blk = ResBlk(64, 128, stride=4)
    tmp = torch.randn(2, 64, 32, 32)
    out = blk(tmp)
    print('block:', out.shape)
    x = torch.randn(2, 3, 32, 32)
    model = ResNet18()
    out = model(x)
    print('resnet:', out.shape)


if __name__ == '__main__':
    main()
