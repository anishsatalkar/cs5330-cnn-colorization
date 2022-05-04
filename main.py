# Imports.
import torch
from torch import nn
from torchvision import transforms
import time

from config_reader import ConfigReader
from grayscalefolder import GrayscaleImageFolder
from model import GrayscaleToColorModel, Trainer


def main():
    """
    Loads the training and validation data and trains the model.
    """
    # a = torch.cuda.FloatTensor()
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(device)
    # print(torch.cuda.is_available())
    # print(torch.version.cuda)

    use_gpu = torch.cuda.is_available()
    config = ConfigReader.read()

    model = GrayscaleToColorModel(kernel_size=3)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    if use_gpu:
        criterion = criterion.cuda()
        model = model.cuda()

    train_transforms = transforms.Compose(
        [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()])
    train_imagefolder = GrayscaleImageFolder(
        f'{config["train_val_split_dir"]}/train', train_transforms, False)
    train_loader = torch.utils.data.DataLoader(
        train_imagefolder, batch_size=64, shuffle=True)

    validation_transforms = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(224)])
    validation_imagefolder = GrayscaleImageFolder(
        f'{config["train_val_split_dir"]}/val', validation_transforms, False)
    validation_loader = torch.utils.data.DataLoader(
        validation_imagefolder, batch_size=64, shuffle=False)

    save_images = True
    max_losses = 1e10
    epochs = 5
    path_to_save = {'grayscale': 'outputs/gray/',
                    'color': 'outputs/color/'}
    lr = 0.01
    for lr_count in range(3):
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        lr += 0.05
        for epoch in range(epochs):
            start = time.time()
            Trainer.train_model(train_loader, model, criterion, optimizer, epoch)
            end = time.time()
            print(f'Epoch {epoch + 1} took {end - start} seconds')
            with torch.no_grad():
                losses = Trainer.validate_model(
                    validation_loader, model, criterion, save_images, path_to_save, epoch)

            if losses < max_losses:
                max_losses = losses
                torch.save(model.state_dict(),
                           f'ckpt/model-epoch-{epoch + 1}-losses-{losses}')

    # image = Image.open(image_file)
    #
    # # random resize crop and save image
    # randresizecrop = RandomResizedCrop(128)
    # image = randresizecrop(image)
    # lab_image = convertToLAB(image)
    # plot_LAB(image, lab_image)


if __name__ == '__main__':
    main()
