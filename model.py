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
    def validate_model(validate_loader, cnn_model, loss_criterion, should_save, save_path, epoch):
        """
        Validates the current state of the model.
        :param validate_loader: The loader object that specifies parameters like batch size, shuffle behavior, etc.
        :param cnn_model: The CNN model.
        :param loss_criterion: The current_loss function.
        :param should_save: Flag that specifies whether while validation should the images be saved or not.
        :param save_path: Path where the predicted image should be saved.
        :param epoch: Current value of the training epoch.
        :return: Average current_loss.
        """
        cnn_model.eval()

        batch_time, data_time, accumulated_losses = State(), State(), State()

        end_time = time.time()
        is_image_saved = False

        use_gpu = False
        if torch.cuda.is_available():
            use_gpu = True

        for idx, (gray_image, ab_image, target_image) in enumerate(validate_loader):
            data_time.update_state(time.time() - end_time)

            if use_gpu:
                gray_image, ab_image, target_image = gray_image.cuda(), ab_image.cuda(), target_image.cuda()

            print(f"Predicting {idx} image")
            predicted_ab_image = cnn_model(gray_image)
            current_loss = loss_criterion(predicted_ab_image, ab_image)
            accumulated_losses.update_state(current_loss.item(), gray_image.size(0))

            if epoch=="":
                for jdx in range(min(len(predicted_ab_image), 10)):
                    print(f"Saving {idx} image")
                    save_name = f"img-{idx * validate_loader.batch_size + jdx}-epoch-{epoch}.jpg"
                    ConvertToRGB.convert_to_rgb(gray_image[jdx].cpu(), ab_img=predicted_ab_image[jdx].detach().cpu(),
                                                path_to_save=save_path, save_name=save_name)
            else:
                # if save_imgs:
                if should_save and not is_image_saved:
                    is_image_saved = True
                    for jdx in range(min(len(predicted_ab_image), 10)):
                        print(f"Saving {idx} image")
                        save_name = f"img-{idx * validate_loader.batch_size + jdx}-epoch-{epoch}.jpg"
                        ConvertToRGB.convert_to_rgb(gray_image[jdx].cpu(), ab_img=predicted_ab_image[jdx].detach().cpu(),
                                                    path_to_save=save_path, save_name=save_name)

            batch_time.update_state(time.time() - end_time)
            end_time = time.time()

            if idx % 50 == 0:
                print(f'Validation: [{idx}/{len(validate_loader)}]\t'
                      f'Time {batch_time.value:.3f} ({batch_time.average:.3f})\t'
                      f'Loss {accumulated_losses.value:.4f} ({accumulated_losses.average:.4f})\t')

        print("Done with validation.")
        return accumulated_losses.average

    @staticmethod
    def train_model(train_loader, model, criterion, cnn_optimizer, epoch):
        """
        Trains the model.
        :param train_loader: Train loader object that specifies parameters like batch size, shuffle behavior, etc.
        :param model: The CNN model.
        :param criterion: The loss function.
        :param cnn_optimizer: The CNN optimizer.
        :param epoch: The current epoch.
        """
        print(f"Training epoch {epoch}")

        model.train_model()

        batch_time, data_time, accumulated_losses = State(), State(), State()

        end_time = time.time()

        use_gpu = False
        if torch.cuda.is_available():
            use_gpu = True

        for idx, (gray_image, ab_image, target) in enumerate(train_loader):
            data_time.update_state(time.time(), end_time)

            if use_gpu:
                gray_image, ab_image, target = gray_image.cuda(), ab_image.cuda(), target.cuda()

            predicted_ab_image = model(gray_image)
            current_loss = criterion(predicted_ab_image, ab_image)
            accumulated_losses.update_state(current_loss.item(), gray_image.size(0))

            cnn_optimizer.zero_grad()
            current_loss.backward()
            cnn_optimizer.step()

            batch_time.update_state(time.time(), end_time)
            end_time = time.time()

            if idx % 50 == 0:
                print(f'Epoch: [{epoch}][{idx}/{len(train_loader)}]\t'
                      f'Time {batch_time.value:.3f} ({batch_time.average:.3f})\t'
                      f'Data {data_time.value:.3f} ({data_time.average:.3f})\t'
                      f'Loss {accumulated_losses.value:.4f} ({accumulated_losses.average:.4f})\t')

        print(f'Trained epoch {epoch}')
