import torch.nn as nn
from torchvision import models


class GrayscaleToColorModel(nn.Module):
    def __init__(self, size=128):
        super(GrayscaleToColorModel, self).__init__()

        # TODO: add num_classes argument
        resnet = models.resnet18(num_classes=365)

        # Change the weight of the first layer so that it accepts single channel grayscale input.
        resnet.conv1.weight = nn.Parameter(resnet.conv1.weight.sum(dim=1).unsqueeze(1))

        # TODO: Check resnet.children(), does increasing the layers improve the accuracy?
        # Use only the first 6 layers of ResNet18
        self.resnet_layers = nn.Sequential(*list(resnet.children())[0:6])

        # TODO: Why to upsample?
        # Upsample the output from the last layer of ResNet
        self.upsample_layers = nn.Sequential(
            # TODO: Try padding 1.
            nn.Conv2d(size, 128, kernel_size=3, padding=1),
            # TODO
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # TODO Try varying the scale_factor
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),

            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 2, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2)
        )

        def forward(input_frame):
            features_from_resnet_layers = self.resnet_layers(input_frame)
            a_b_channel_output = self.upsample_layers(features_from_resnet_layers)
            return a_b_channel_output

