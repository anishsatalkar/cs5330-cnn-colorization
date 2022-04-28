import time

import torch.cuda
import torch.nn as nn
from torchvision import models

from helper import State, ConvertToRGB


class GrayscaleToColorModel(nn.Module):
    def __init__(self, size=128):
        super(GrayscaleToColorModel, self).__init__()

        # TODO: add num_classes argument
        resnet = models.resnet18(num_classes=365)

        # Change the weight of the first layer so that it accepts single channel grayscale input.
        resnet.conv1.weight = nn.Parameter(
            resnet.conv1.weight.sum(dim=1).unsqueeze(1))

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

    def forward(self, input_frame):
        features_from_resnet_layers = self.resnet_layers(input_frame)
        a_b_channel_output = self.upsample_layers(features_from_resnet_layers)
        return a_b_channel_output


class Trainer(object):
    @staticmethod
    def validate(validate_loader, model, criterion, save_imgs, epoch):
        model.eval()

        batch_time, data_time, losses = State(), State(), State()

        end_time = time.time()
        is_image_saved = False

        path_to_save = {'grayscale': 'outputs/gray/',
                        'color': 'outputs/color/'}

        use_gpu = False
        if torch.cuda.is_available():
            use_gpu = True

        for idx, (gray, ab_img, target) in enumerate(validate_loader):
            data_time.update(time.time() - end_time)

            if use_gpu:
                gray, ab_img, target = gray.cuda(), ab_img.cuda(), target.cuda()

            predicted_ab_img = model(gray)
            loss = criterion(predicted_ab_img, ab_img)
            losses.update(loss.item(), gray.size(0))

            if save_imgs and not is_image_saved:
                is_image_saved = True
                for jdx in range(min(len(predicted_ab_img), 10)):
                    save_name = f"img-{idx * validate_loader.batch_size + jdx}-epoch-{epoch}.jpg"
                    ConvertToRGB.convert_to_rgb(gray[jdx].cpu(), ab_img=predicted_ab_img[jdx].detach().cpu(),
                                                path_to_save=path_to_save, save_name=save_name)

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            if idx % 50 == 0:
                print(f'Validation: [{idx}/{len(validate_loader)}]\t'
                      f'Time {batch_time.val:.3f} ({batch_time.average:.3f})\t'
                      f'Loss {losses.val:.4f} ({losses.average:.4f})\t')

        print("Done with validation.")
        return losses.average

    @staticmethod
    def train(train_loader, model, criterion, optimizer, epoch):
        print(f"Training epoch {epoch}")

        model.train()

        batch_time, data_time, losses = State(), State(), State()

        end_time = time.time()

        use_gpu = False
        if torch.cuda.is_available():
            use_gpu = True

        for idx, (gray, ab_img, target) in enumerate(train_loader):
            data_time.update(time.time(), end_time)

            if use_gpu:
                gray, ab_img, target = gray.cuda(), ab_img.cuda(), target.cuda()

            predicted_ab_img = model(gray)
            loss = criterion(predicted_ab_img, ab_img)
            losses.update(loss.item(), gray.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time(), end_time)
            end_time = time.time()

            if idx % 50 == 0:
                print(f'Epoch: [{epoch}][{idx}/{len(train_loader)}]\t'
                      f'Time {batch_time.val:.3f} ({batch_time.average:.3f})\t'
                      f'Data {data_time.val:.3f} ({data_time.average:.3f})\t'
                      f'Loss {losses.val:.4f} ({losses.average:.4f})\t')

        print(f'Trained epoch {epoch}')
