import torch
from torch import nn
import torch.nn.functional as F


class UNetWithCracks(nn.Module):
    def __init__(self):
        super(UNetWithCracks, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.conv11 = nn.Sequential(
            nn.Conv2d(64,64, kernel_size=3, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.conv12 = nn.Sequential(
            nn.Conv2d(64,64, kernel_size=1, padding = 0),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.conv21 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.conv31 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.conv32 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())

        self.conv41 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.conv42 = self. conv41

        self.conv5 = self.conv41
        self.conv51 = self.conv41
        self.conv52 = self.conv41
        self.conv53 = nn.Sequential(
            nn.Conv2d(512,2048, kernel_size=3, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU())
        self.conv54 = nn.Sequential(
            nn.Conv2d(2048, 2048, kernel_size=3, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU())

        self.Up2 = nn.Sequential(
            nn.UpsamplingBilinear2d(size=(32, 32)),
            nn.ConvTranspose2d(2048, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256))

        self.Up3 = nn.Sequential(
            nn.UpsamplingBilinear2d(size=(64, 64)),
            nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128))

        self.Up4 = nn.Sequential(
            nn.UpsamplingBilinear2d(size=(128, 128)),
            nn.ConvTranspose2d(384, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64))

        self.Up5 = nn.Sequential(
            nn.UpsamplingBilinear2d(size=(256, 256)),
            nn.ConvTranspose2d(192, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128))

        self.Up51 = nn.Sequential(
            nn.UpsamplingBilinear2d(size=(256, 256)),
            nn.ConvTranspose2d(192, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128))

        self.Up52 = nn.Sequential(
            nn.ConvTranspose2d(128, 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(4))

        self.Up53 = nn.ConvTranspose2d(4, 4, kernel_size=1, padding=0)

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x_hBy1 = x

        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = self.conv21(x)
        x_hBy2 = x

        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        x = self.conv31(x)
        x = self.conv32(x)
        x_hBy4 = x

        x = F.max_pool2d(x, 2)
        x = self.conv4(x)
        x = self.conv41(x)
        x = self.conv42(x)
        x_hBy8 = x  # not used

        x=F.max_pool2d(x,2)
        x = self.conv5(x)
        x = self.conv51(x)
        x = self.conv52(x)
        x = F.dropout2d(x, p=0.1, training=True)
        x = self.conv53(x)
        x = F.dropout2d(x, p=0.1, training=True)
        x = self.conv54(x)

        x = self.Up2(x)

        x = self.Up3(x)
        x = torch.cat((x_hBy4, x), dim=1)

        x = self.Up4(x)
        x = torch.cat((x_hBy2, x), dim=1)

        x = self.Up5(x)
        x = torch.cat((x_hBy1, x), dim=1)

        x = self.Up51(x)
        x = self.Up52(x)
        x = self.Up53(x)

        return x


if __name__ == "__main__":
    MODELPATH = "models/synchrotron_tomography/model_e137_l0.6860.pt"
    model = UNetWithCracks()
    model.load_state_dict(torch.load(MODELPATH))
    print(model)