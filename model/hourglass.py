import torch.nn as nn
from .base_module import ConvBlock, DownsampleBlock, ResidualBlock, SkipConnection, UpsampleBlock


class HourGlass(nn.Module):
    def __init__(self, convNum=4, resNum=4, inChannel=6, outChannel=3):
        super(HourGlass, self).__init__()
        self.inConv = ConvBlock(inChannel, 64, convNum=2)
        self.down1 = nn.Sequential(*[DownsampleBlock(64, 128, withConvRelu=False), ConvBlock(128, 128, convNum=2)])
        self.down2 = nn.Sequential(
            *[DownsampleBlock(128, 256, withConvRelu=False), ConvBlock(256, 256, convNum=convNum)])
        self.down3 = nn.Sequential(
            *[DownsampleBlock(256, 512, withConvRelu=False), ConvBlock(512, 512, convNum=convNum)])
        self.residual = nn.Sequential(*[ResidualBlock(512) for _ in range(resNum)])
        self.up3 = nn.Sequential(*[UpsampleBlock(512, 256), ConvBlock(256, 256, convNum=convNum)])
        self.skip3 = SkipConnection(256)
        self.up2 = nn.Sequential(*[UpsampleBlock(256, 128), ConvBlock(128, 128, convNum=2)])
        self.skip2 = SkipConnection(128)
        self.up1 = nn.Sequential(*[UpsampleBlock(128, 64), ConvBlock(64, 64, convNum=2)])
        self.skip1 = SkipConnection(64)
        self.outConv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, outChannel, kernel_size=1, padding=0)
        )

    def forward(self, x):
        f1 = self.inConv(x)
        f2 = self.down1(f1)
        f3 = self.down2(f2)
        f4 = self.down3(f3)
        r4 = self.residual(f4)
        r3 = self.skip3(self.up3(r4), f3)
        r2 = self.skip2(self.up2(r3), f2)
        r1 = self.skip1(self.up1(r2), f1)
        y = self.outConv(r1)
        return y


class ResidualHourGlass(nn.Module):
    def __init__(self, resNum=4, inChannel=6, outChannel=3):
        super(ResidualHourGlass, self).__init__()
        self.inConv = nn.Conv2d(inChannel, 64, kernel_size=3, padding=1)
        self.residualBefore = nn.Sequential(*[ResidualBlock(64) for _ in range(2)])
        self.down1 = nn.Sequential(
            *[DownsampleBlock(64, 128, withConvRelu=False), ConvBlock(128, 128, convNum=2)])
        self.down2 = nn.Sequential(
            *[DownsampleBlock(128, 256, withConvRelu=False), ConvBlock(256, 256, convNum=2)])
        self.residual = nn.Sequential(*[ResidualBlock(256) for _ in range(resNum)])
        self.up2 = nn.Sequential(*[UpsampleBlock(256, 128), ConvBlock(128, 128, convNum=2)])
        self.skip2 = SkipConnection(128)
        self.up1 = nn.Sequential(*[UpsampleBlock(128, 64), ConvBlock(64, 64, convNum=2)])
        self.skip1 = SkipConnection(64)
        self.residualAfter = nn.Sequential(*[ResidualBlock(64) for _ in range(2)])
        self.outConv = nn.Sequential(
            nn.Conv2d(64, outChannel, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        f1 = self.inConv(x)
        f1 = self.residualBefore(f1)
        f2 = self.down1(f1)
        f3 = self.down2(f2)
        r3 = self.residual(f3)
        r2 = self.skip2(self.up2(r3), f2)
        r1 = self.skip1(self.up1(r2), f1)
        y = self.residualAfter(r1)
        y = self.outConv(y)
        return y
