import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class BiSeNetSeg(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(BiSeNetSeg, self).__init__()
        self.model = smp.BiSeNet(
            encoder_name='resnet18',  # Lightweight backbone
            in_channels=in_channels,
            classes=out_channels,
        )

    def forward(self, x):
        return self.model(x)