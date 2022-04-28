import torch
import torchvision
from PIL import Image
from torch import nn
from torchvision import datasets, transforms

from model import GrayscaleToColorModel

from grayscalefolder import GrayscaleImageFolder


def main():
    model = GrayscaleToColorModel()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    train_transforms = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()])
    train_imagefolder = GrayscaleImageFolder('/Users/anishsatalkar/Downloads/images/train', train_transforms)
    train_loader = torch.utils.data.DataLoader(train_imagefolder, batch_size=64, shuffle=True)

    validation_transforms = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
    validation_imagefolder = GrayscaleImageFolder('/Users/anishsatalkar/Downloads/images/val', validation_transforms)
    validation_loader = torch.utils.data.DataLoader(validation_imagefolder, batch_size=64, shuffle=False)








    # image = Image.open(image_file)
    #
    # # random resize crop and save image
    # randresizecrop = RandomResizedCrop(128)
    # image = randresizecrop(image)
    # lab_image = convertToLAB(image)
    # plot_LAB(image, lab_image)


if __name__ == '__main__':
    main()
