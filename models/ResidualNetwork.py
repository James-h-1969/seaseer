import torch.nn as nn

from .ResidualBlock import ResidualBlock


class ResNet(nn.Module):
    def __init__(
        self,
        in_channels,
        num_blocks,
        hidden_channels,
        num_classes=None,
        stride=1,
        kernel_size=3,
    ):
        super().__init__()
        assert len(num_blocks) == len(hidden_channels)

        self.conv1 = nn.Conv2d(
            in_channels,
            hidden_channels[0],
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(hidden_channels[0])
        self.relu = nn.ReLU(inplace=True)

        # build residual layers
        layers = []
        ch = hidden_channels[0]
        for n_blocks, out_ch in zip(num_blocks, hidden_channels):
            layers.append(self._make_layer(ch, out_ch, n_blocks, stride, kernel_size))
            ch = out_ch
        self.res_layers = nn.Sequential(*layers)

        # optional classification head
        self.head = None
        if num_classes is not None:
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(hidden_channels[-1], num_classes),
            )

    @staticmethod
    def _make_layer(in_ch, out_ch, num_blocks, stride, kernel_size):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        ch = in_ch
        for s in strides:
            layers.append(ResidualBlock(ch, out_ch, stride=s, kernel_size=kernel_size))
            ch = out_ch
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.res_layers(out)
        if self.head is not None:
            out = self.head(out)
        return out
