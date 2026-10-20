import torch
import torch.nn as nn


def conv_block(in_channels, out_channels):
    """Two 3x3 convolutions, each followed by ReLU. This is the
    basic repeated unit used all over the U-Net."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
    )


class SimpleUNet(nn.Module):
    """A small U-Net: 3 downsampling steps, few channels.
    Same architecture as the DeepGlobe model - only NUM_CLASSES differs,
    which is passed in from config.py at construction time."""

    def __init__(self, num_classes):
        super().__init__()

        # encoder (shrinks the image, learns WHAT is in it)
        self.enc1 = conv_block(3, 16)
        self.enc2 = conv_block(16, 32)
        self.enc3 = conv_block(32, 64)

        self.pool = nn.MaxPool2d(2)

        # bottleneck (most compressed representation)
        self.bottleneck = conv_block(64, 128)

        # decoder (grows the image back, learns WHERE things are)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = conv_block(128, 64)   # 128 = 64 (upsampled) + 64 (skip connection)

        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = conv_block(64, 32)

        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = conv_block(32, 16)

        # 1x1 conv: turns 16 feature channels into one score per class, per pixel
        self.final = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        # encoder
        e1 = self.enc1(x)                  # full resolution
        e2 = self.enc2(self.pool(e1))      # 1/2 resolution
        e3 = self.enc3(self.pool(e2))      # 1/4 resolution

        # bottleneck
        b = self.bottleneck(self.pool(e3))  # 1/8 resolution

        # decoder, with skip connections bringing back fine detail
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.final(d1)  # shape: (batch, num_classes, H, W) - raw logits
