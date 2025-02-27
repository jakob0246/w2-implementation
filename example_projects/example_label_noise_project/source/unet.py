import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    """
    Simple UNet architecture based on https://github.com/milesial/Pytorch-UNet and https://github.com/RedekopEP/SUE
    """

    def __init__(self, n_channels: int = 3, n_classes: int = 6, activation: bool = False, use_dropout: bool = False):
        super(UNet, self).__init__()
        self.inc = _inconv(n_channels, 64, use_dropout)
        self.down1 = _down(64, 128, use_dropout)
        self.down2 = _down(128, 256, use_dropout)
        self.down3 = _down(256, 512, use_dropout)
        self.down4 = _down(512, 512, use_dropout)
        self.up1 = _up(1024, 256, use_dropout)
        self.up2 = _up(512, 128, use_dropout)
        self.up3 = _up(256, 64, use_dropout)
        self.up4 = _up(128, 64, use_dropout)
        self.outc = _outconv(64, n_classes)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        if self.activation:
            x = torch.sigmoid(x)
        return x


class _double_conv(nn.Module):
    """ (Conv => BatchNorm => ReLU) * 2 """

    def __init__(self, in_ch: int, out_ch: int):
        super(_double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x:  torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return x


class _inconv(nn.Module):

    def __init__(self, in_ch: int, out_ch: int, flag_do: bool, do_p: int = 0.25):
        super(_inconv, self).__init__()
        self.conv = _double_conv(in_ch, out_ch)
        self.flag_do = flag_do
        self.do_p = do_p

    def forward(self, x:  torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.flag_do:
            x = F.dropout2d(x, p=self.do_p, training=True)
        return x


class _down(nn.Module):

    def __init__(self, in_ch: int, out_ch: int, flag_do: bool, do_p: int = 0.25):
        super(_down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            _double_conv(in_ch, out_ch)
        )
        self.flag_do = flag_do
        self.do_p = do_p

    def forward(self, x:  torch.Tensor) -> torch.Tensor:
        x = self.mpconv(x)
        if self.flag_do:
            x = F.dropout2d(x, p=self.do_p, training=True)
        return x


class _up(nn.Module):

    def __init__(self, in_ch: int, out_ch: int, flag_do: bool, do_p: int = 0.25, bilinear: bool = True):
        super(_up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = _double_conv(in_ch, out_ch)
        self.flag_do = flag_do
        self.do_p = do_p

    def forward(self, x1:  torch.Tensor, x2:  torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        if self.flag_do:
            x = F.dropout2d(x, p=self.do_p, training=True)
        return x


class _outconv(nn.Module):

    def __init__(self, in_ch: int, out_ch: int):
        super(_outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x:  torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return x