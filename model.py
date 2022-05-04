import time

import torch.cuda
import torch.nn as nn
from torchvision import models

from helper import State, ConvertToRGB


class GrayscaleToColorModel(nn.Module):
    def __init__(self, size=128, kernel_size=3, activation=nn.ReLU()):
        super(GrayscaleToColorModel, self).__init__()

        # TODO: add num_classes argument
        resnet = models.resnet18(num_classes=365)

        # Change the weight of the first layer so that it accepts single channel grayscale input.
        resnet.conv1.weight = nn.Parameter(
            resnet.conv1.weight.sum(dim=1).unsqueeze(1))

        # TODO: Check resnet.children(), does increasing the layers improve the accuracy?
        # Use only the first 6 layers of ResNet18
        self.resnet_layers = nn.Sequential(*list(resnet.children())[0:6])

        # Upsample the output from the last layer of ResNet
        padding = 3
        self.upsample_layers = nn.Sequential(
            nn.Conv2d(size, 128, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(128),
            activation,
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(64),

            activation,
            nn.Conv2d(64, 64, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(64),
            activation,
            nn.Upsample(scale_factor=2),

            nn.Conv2d(64, 32, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(32),
            activation,

            nn.Conv2d(32, 2, kernel_size=kernel_size, padding=padding),
            nn.Upsample(scale_factor=2)
        )

    def forward(self, input_frame):
        features_from_resnet_layers = self.resnet_layers(input_frame)
        a_b_channel_output = self.upsample_layers(features_from_resnet_layers)
        return a_b_channel_output


class Trainer(object):
    @staticmethod
    def validate_model(validate_loader, model, criterion, save_imgs, path_to_save, epoch):
        model.eval()

        batch_time, data_time, losses = State(), State(), State()

        end_time = time.time()
        is_image_saved = False

        use_gpu = False
        if torch.cuda.is_available():
            use_gpu = True

        for idx, (gray, ab_img, target) in enumerate(validate_loader):
            data_time.update_state(time.time() - end_time)

            if use_gpu:
                gray, ab_img, target = gray.cuda(), ab_img.cuda(), target.cuda()

            print(f"Predicting {idx} image")
            predicted_ab_img = model(gray)
            loss = criterion(predicted_ab_img, ab_img)
            losses.update_state(loss.item(), gray.size(0))

            if epoch == "":
                for jdx in range(min(len(predicted_ab_img), 10)):
                    print(f"Saving {idx} image")
                    save_name = f"img-{idx * validate_loader.batch_size + jdx}-epoch-{epoch}.jpg"
                    ConvertToRGB.convert_to_rgb(gray[jdx].cpu(), ab_img=predicted_ab_img[jdx].detach().cpu(),
                                                path_to_save=path_to_save, save_name=save_name)
            else:
                # if save_imgs:
                if save_imgs and not is_image_saved:
                    is_image_saved = True
                    for jdx in range(min(len(predicted_ab_img), 10)):
                        print(f"Saving {idx} image")
                        save_name = f"img-{idx * validate_loader.batch_size + jdx}-epoch-{epoch}.jpg"
                        ConvertToRGB.convert_to_rgb(gray[jdx].cpu(), ab_img=predicted_ab_img[jdx].detach().cpu(),
                                                    path_to_save=path_to_save, save_name=save_name)

            batch_time.update_state(time.time() - end_time)
            end_time = time.time()

            if idx % 50 == 0:
                print(f'Validation: [{idx}/{len(validate_loader)}]\t'
                      f'Time {batch_time.value:.3f} ({batch_time.average:.3f})\t'
                      f'Loss {losses.value:.4f} ({losses.average:.4f})\t')

        print("Done with validation.")
        return losses.average

    @staticmethod
    def train_model(train_loader, model, criterion, optimizer, epoch):
        print(f"Training epoch {epoch}")

        model.train()

        batch_time, data_time, losses = State(), State(), State()

        end_time = time.time()

        use_gpu = False
        if torch.cuda.is_available():
            use_gpu = True

        for idx, (gray, ab_img, target) in enumerate(train_loader):
            data_time.update_state(time.time(), end_time)

            if use_gpu:
                gray, ab_img, target = gray.cuda(), ab_img.cuda(), target.cuda()

            predicted_ab_img = model(gray)
            loss = criterion(predicted_ab_img, ab_img)
            losses.update_state(loss.item(), gray.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update_state(time.time(), end_time)
            end_time = time.time()

            if idx % 50 == 0:
                print(f'Epoch: [{epoch}][{idx}/{len(train_loader)}]\t'
                      f'Time {batch_time.value:.3f} ({batch_time.average:.3f})\t'
                      f'Data {data_time.value:.3f} ({data_time.average:.3f})\t'
                      f'Loss {losses.value:.4f} ({losses.average:.4f})\t')

        print(f'Trained epoch {epoch}')
