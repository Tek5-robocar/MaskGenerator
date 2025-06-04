import torch
import torch.nn as nn
import torchvision.models as models

class MobileNetV3Seg(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(MobileNetV3Seg, self).__init__()
        # Use MobileNetV3-Small as backbone
        self.backbone = models.mobilenet_v3_small(pretrained=True).features
        # Decoder: simple upsampling + conv
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(48, 24, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(24, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_channels, kernel_size=1),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.decoder(x)
        # Upsample to match input resolution
        x = nn.functional.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        return x