import torch.nn as nn
from torch.nn import functional as F
import torch
import numpy as np

def tensor2array(tensors):
    arrays = tensors.detach().to("cpu").numpy()
    return np.transpose(arrays, (0, 2, 3, 1))


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        residual = self.conv(x)
        return x + residual


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, withConvRelu=True):
        super(DownsampleBlock, self).__init__()
        if withConvRelu:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=2),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=2)

    def forward(self, x):
        return self.conv(x)


class ConvBlock(nn.Module):
    def __init__(self, inChannels, outChannels, convNum):
        super(ConvBlock, self).__init__()
        self.inConv = nn.Sequential(
            nn.Conv2d(inChannels, outChannels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        layers = []
        for _ in range(convNum - 1):
            layers.append(nn.Conv2d(outChannels, outChannels, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        x = self.inConv(x)
        x = self.conv(x)
        return x


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class SkipConnection(nn.Module):
    def __init__(self, channels):
        super(SkipConnection, self).__init__()
        self.conv = nn.Conv2d(2 * channels, channels, 1, bias=False)

    def forward(self, x, y):
        x = torch.cat((x, y), 1)
        return self.conv(x)