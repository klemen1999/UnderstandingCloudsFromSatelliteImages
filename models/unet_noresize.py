import torch.nn as nn
import segmentation_models_pytorch as smp

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
             stride=stride, padding=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.relu(x)

        return x


class PreprocessBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = ConvBlock(3, 8, kernel_size=5, stride=1)
        self.block2 = ConvBlock(8, 8, kernel_size=5, stride=2)
        self.block3 = ConvBlock(8, 8, kernel_size=5, stride=1)
        self.block4 = ConvBlock(8, 3, kernel_size=5, stride=2)

    def forward(self, inputs):
        x = self.block1(inputs)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        return x


class UnetNoResize(nn.Module):
    def __init__(self, encoder, encoder_weights):
        super().__init__()
        self.preprocess_model = PreprocessBlock()
        self.main_model = smp.Unet(
            encoder_name=encoder, 
            encoder_weights=encoder_weights, 
            classes=4, 
            activation=None,
        )
    
    def forward(self, inputs):
        x = self.preprocess_model(inputs)
        x = self.main_model(x)

        return x
