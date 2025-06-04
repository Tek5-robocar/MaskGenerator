import torch
import torch.nn as nn

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class DSConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DSConvBlock, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=stride,
            padding=1, groups=in_channels, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock(out_channels)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.se(x)
        return x

class EfficientLiteSeg(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(EfficientLiteSeg, self).__init__()
        self.enc1 = DSConvBlock(in_channels, 16, stride=2)
        self.enc2 = DSConvBlock(16, 32, stride=2)
        self.enc3 = DSConvBlock(32, 64, stride=2)
        self.enc4 = DSConvBlock(64, 128, stride=1)
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.dec3 = nn.Conv2d(64 + 32, 64, kernel_size=3, padding=1)  # Fixed: 64 + 32 = 96
        self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.dec2 = nn.Conv2d(32 + 16, 32, kernel_size=3, padding=1)   # 32 + 16 = 48
        self.upconv1 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.dec1 = nn.Conv2d(16 + in_channels, 16, kernel_size=3, padding=1)  # 16 + 3 = 19
        self.conv_last = nn.Conv2d(16, out_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        e1 = self.enc1(x)   # [B, 16, H/2, W/2]
        e2 = self.enc2(e1)  # [B, 32, H/4, W/4]
        e3 = self.enc3(e2)  # [B, 64, H/8, W/8]
        e4 = self.enc4(e3)  # [B, 128, H/8, W/8]
        d3 = self.upconv3(e4)  # [B, 64, H/4, W/4]
        d3 = torch.cat([d3, e2], dim=1)  # [B, 64 + 32, H/4, W/4]
        d3 = self.relu(self.dec3(d3))    # [B, 64, H/4, W/4]
        d2 = self.upconv2(d3)            # [B, 32, H/2, W/2]
        d2 = torch.cat([d2, e1], dim=1)  # [B, 32 + 16, H/2, W/2]
        d2 = self.relu(self.dec2(d2))    # [B, 32, H/2, W/2]
        d1 = self.upconv1(d2)            # [B, 16, H, W]
        d1 = torch.cat([d1, x], dim=1)   # [B, 16 + in_channels, H, W]
        d1 = self.relu(self.dec1(d1))    # [B, 16, H, W]
        out = self.conv_last(d1)         # [B, out_channels, H, W]
        return out