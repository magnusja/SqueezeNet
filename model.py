from torch import nn, torch
from torch.nn import init


class Fire(nn.Module):
    def __init__(self, input_channels, num_squeeze, num_expand_1x1, num_expand_3x3):
        super().__init__()

        self.squeeze = nn.Sequential(
            nn.Conv2d(input_channels, num_squeeze, kernel_size=1),
            nn.ReLU()
        )

        self.expand_1x1 = nn.Sequential(
            nn.Conv2d(num_squeeze, num_expand_1x1, kernel_size=1),
            nn.ReLU()
        )

        self.expand_3x3 = nn.Sequential(
            nn.Conv2d(num_squeeze, num_expand_3x3, kernel_size=3),
            nn.ReLU()
        )

    def forward(self, x):
        squeezed = self.squeeze(x)
        expanded = torch.cat([
            self.expand_1x1(squeezed),
            self.expand_3x3(squeezed)
        ])
        return expanded


class SqueezeNet(nn.Module):

    def __init__(self, num_classes, input_channels=3):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 96, kernel_size=7, stride=2),
            nn.MaxPool2d(3, stride=2),
            Fire(96, 16, 64, 64),
            Fire(128, 16, 64, 64),
            Fire(128, 32, 128, 128),
            nn.MaxPool2d(3, stride=2),
            Fire(256, 32, 128, 128),
            Fire(256, 48, 192, 192),
            Fire(384, 48, 192, 192),
            Fire(384, 64, 256, 256),
            nn.MaxPool2d(3, stride=2),
            Fire(512, 64, 256, 256),
            nn.Dropout(),
            nn.Conv2d(512, num_classes, 1),
            nn.AvgPool2d(13, stride=1)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        return self.model(x)
