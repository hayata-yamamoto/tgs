import torch
import torch.nn as nn
import torch.functional as F


class Unet(nn.Module):
    def __init__(self):
        self.conv0 = nn.Conv2d()
        self.conv1 = nn.Conv2d()
        self.pool0 = nn.MaxUnpool2d()
        self.drop1 = nn.Dropout()

        self.conv2 = nn.Conv2d()
        self.conv3 = nn.Conv2d()
        self.pool1 = nn.MaxUnpool2d()
        self.drop1 = nn.Dropout()

        self.conv0 = nn.Conv2d()
        self.conv1 = nn.Conv2d()
        self.pool0 = nn.MaxUnpool2d()
        self.drop = nn.Dropout()

    def forward(self, *input):
        pass


    def downsampling(self):
        return nn.Sequential(
            nn.Conv2d(),
            nn.Conv2d(),
            nn.MaxUnpool2d(),
            nn.Dropout()
        )