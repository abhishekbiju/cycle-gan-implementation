"""
CycleGAN generator: encoder (downsampling) → residual blocks → decoder (upsampling).
See Zhu et al., "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks".
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs)
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity(),
        )

    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(
                channels,
                channels,
                down=True,
                use_act=True,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            ConvBlock(
                channels,
                channels,
                down=True,
                use_act=False,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    """
    ResNet-style translator G : X → Y.
    Default depth matches 256×256 inputs (three downsamples, nine residual blocks).
    """

    def __init__(
        self,
        img_channels=3,
        num_features=64,
        num_residuals=9,
    ):
        super().__init__()
        self.initial = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(img_channels, num_features, kernel_size=7),
            nn.InstanceNorm2d(num_features),
            nn.ReLU(inplace=True),
        )
        self.down1 = ConvBlock(
            num_features,
            num_features * 2,
            down=True,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.down2 = ConvBlock(
            num_features * 2,
            num_features * 4,
            down=True,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.down3 = ConvBlock(
            num_features * 4,
            num_features * 8,
            down=True,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.residuals = nn.Sequential(
            *[ResidualBlock(num_features * 8) for _ in range(num_residuals)]
        )
        self.up1 = ConvBlock(
            num_features * 8,
            num_features * 4,
            down=False,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.up2 = ConvBlock(
            num_features * 4,
            num_features * 2,
            down=False,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.up3 = ConvBlock(
            num_features * 2,
            num_features,
            down=False,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.last = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(num_features, img_channels, kernel_size=7),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.residuals(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        return self.last(x)


def _test():
    g = Generator(img_channels=3, num_features=64, num_residuals=9)
    x = torch.randn(2, 3, 256, 256)
    y = g(x)
    print("in ", x.shape, "out", y.shape)


if __name__ == "__main__":
    _test()
