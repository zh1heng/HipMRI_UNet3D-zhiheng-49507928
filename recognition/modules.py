import torch
import torch.nn as nn


class UNet(nn.Module):
    """2D U-Net architecture for segmentation tasks."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1) -> None:
        super().__init__()
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(256, 128)
        self.dec2 = self.conv_block(128, 64)
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    @staticmethod
    def conv_block(in_c: int, out_c: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d1 = self.up1(e3)
        d1 = torch.cat([d1, e2], dim=1)
        d1 = self.dec1(d1)
        d2 = self.up2(d1)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)
        out = torch.sigmoid(self.out_conv(d2))
        return out


class UNet3D(nn.Module):
    """3D U-Net architecture for volumetric segmentation tasks."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1) -> None:
        super().__init__()
        self.enc1 = self.conv_block(in_channels, 32)
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)
        self.pool = nn.MaxPool3d(2)
        self.up1 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)
        self.dec2 = self.conv_block(64, 32)
        self.out_conv = nn.Conv3d(32, out_channels, kernel_size=1)

    @staticmethod
    def conv_block(in_c: int, out_c: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for UNet3D. The network outputs raw logits rather than
        probabilities. A sigmoid activation should be applied externally when
        computing probabilities or Dice scores.
        """
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d1 = self.up1(e3)
        d1 = torch.cat([d1, e2], dim=1)
        d1 = self.dec1(d1)
        d2 = self.up2(d1)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)
        # return raw logits; apply sigmoid externally when needed
        out = self.out_conv(d2)
        return out