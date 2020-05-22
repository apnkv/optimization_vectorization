import torch
from torch import nn


def conv_bn_relu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def upconv(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        conv_bn_relu(in_channels, out_channels, kernel, padding),
    )


class CrossingRefiner(nn.Module):
    def __init__(self):
        super(CrossingRefiner, self).__init__()

        self.conv1 = conv_bn_relu(1, 8, 3, 1)
        self.conv2 = conv_bn_relu(8, 16, 3, 1)
        self.conv3 = conv_bn_relu(16, 32, 3, 1)

        self.upconv1 = upconv(32, 16, 3, 1)
        self.upconv2 = upconv(32, 8, 3, 1)
        self.upconv3 = upconv(16, 4, 3, 1)

        self.maxpool = nn.MaxPool2d(kernel_size=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        down1 = self.maxpool(self.conv1(x))
        down2 = self.maxpool(self.conv2(down1))
        down3 = self.maxpool(self.conv3(down2))

        x = self.upconv1(down3)
        x = self.upconv2(torch.cat((x, down2), dim=1))
        x = self.upconv3(torch.cat((x, down1), dim=1))

        return x


class CrossingRefinerFull(nn.Module):
    def __init__(self):
        super(CrossingRefinerFull, self).__init__()

        self.downscaling_branch = CrossingRefiner()

        self.conv1 = conv_bn_relu(1, 4, 5, 2)
        self.conv2 = conv_bn_relu(4, 8, 5, 2)
        self.conv3 = conv_bn_relu(8, 8, 5, 2)

        self.conv4 = conv_bn_relu(8, 8, 5, 2)
        self.conv5 = conv_bn_relu(8, 4, 5, 2)
        self.conv6 = conv_bn_relu(8, 1, 5, 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        scaling_channels = self.downscaling_branch(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = torch.cat((x, scaling_channels), dim=1)
        x = self.conv6(x)

        return x

    def apply_convolutions(self, x, n_convolutions=2):
        convolutions = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6]
        for conv in convolutions[:n_convolutions]:
            x = conv(x)

        return x
