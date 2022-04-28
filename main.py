import torch
from torch import nn
from torchvision import transforms

from grayscalefolder import GrayscaleImageFolder
from model import GrayscaleToColorModel, Trainer


def main():
    model = GrayscaleToColorModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    train_transforms = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()])
    train_imagefolder = GrayscaleImageFolder('/Users/anishsatalkar/Downloads/images/train', train_transforms)
    train_loader = torch.utils.data.DataLoader(train_imagefolder, batch_size=64, shuffle=True)

    validation_transforms = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
    validation_imagefolder = GrayscaleImageFolder('/Users/anishsatalkar/Downloads/images/val', validation_transforms)
    validation_loader = torch.utils.data.DataLoader(validation_imagefolder, batch_size=64, shuffle=False)

    save_images = True
    max_losses = 1e10
    epochs = 5

    for epoch in range(epochs):
        Trainer.train(train_loader, model, criterion, optimizer, epoch)

        with torch.no_grad():
            losses = Trainer.validate(validation_loader, model, criterion, save_images, epoch)

        if losses < max_losses:
            max_losses = losses
            torch.save(model.state_dict(), f'checkpoints/model-epoch-{epoch + 1}-losses-{losses}')

    # image = Image.open(image_file)
    #
    # # random resize crop and save image
    # randresizecrop = RandomResizedCrop(128)
    # image = randresizecrop(image)
    # lab_image = convertToLAB(image)
    # plot_LAB(image, lab_image)


if __name__ == '__main__':
    main()
