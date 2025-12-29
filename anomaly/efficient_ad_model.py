import torch
import torch.nn as nn
import torch.nn.functional as F


class PDN(nn.Module):
    def __init__(self, out_channels=384, padding=False):
        super(PDN, self).__init__()
        self.conv1 = nn.Conv2d(
            3, 128, kernel_size=4, stride=1, padding=3 if padding else 0
        )
        self.conv2 = nn.Conv2d(
            128, 256, kernel_size=4, stride=1, padding=3 if padding else 0
        )
        self.conv3 = nn.Conv2d(
            256, 256, kernel_size=3, stride=1, padding=1 if padding else 0
        )
        self.conv4 = nn.Conv2d(256, out_channels, kernel_size=4, stride=1, padding=0)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.avgpool1(x)
        x = F.relu(self.conv2(x))
        x = self.avgpool2(x)
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        return x


class AutoEncoder(nn.Module):
    def __init__(self, out_channels=384):
        super(AutoEncoder, self).__init__()
        # Encoder
        self.en_conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=1, padding=1)
        self.en_conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=1)
        self.en_conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=1, padding=1)
        self.en_conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=1)
        self.en_conv5 = nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1)
        self.en_conv6 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=0)

        # Decoder
        self.de_conv1 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1)
        self.de_conv2 = nn.Conv2d(1024, 1024, kernel_size=4, stride=1, padding=1)
        self.de_conv3 = nn.Conv2d(1024, 1024, kernel_size=4, stride=1, padding=1)
        self.de_conv4 = nn.Conv2d(1024, 1024, kernel_size=4, stride=1, padding=1)
        self.de_conv5 = nn.Conv2d(1024, 1024, kernel_size=4, stride=1, padding=1)
        self.de_conv6 = nn.Conv2d(
            1024, out_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        # Encoder
        x = F.relu(self.en_conv1(x))
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.en_conv2(x))
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.en_conv3(x))
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.en_conv4(x))
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.en_conv5(x))
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.en_conv6(x))

        # Decoder
        x = F.interpolate(x, size=(3, 3), mode="bilinear", align_corners=False)
        x = F.relu(self.de_conv1(x))
        x = F.interpolate(x, size=(8, 8), mode="bilinear", align_corners=False)
        x = F.relu(self.de_conv2(x))
        x = F.interpolate(x, size=(15, 15), mode="bilinear", align_corners=False)
        x = F.relu(self.de_conv3(x))
        x = F.interpolate(x, size=(32, 32), mode="bilinear", align_corners=False)
        x = F.relu(self.de_conv4(x))
        x = F.interpolate(x, size=(63, 63), mode="bilinear", align_corners=False)
        x = F.relu(self.de_conv5(x))
        x = F.interpolate(x, size=(127, 127), mode="bilinear", align_corners=False)
        x = self.de_conv6(x)
        return x
