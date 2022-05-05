import torch
import torch.nn as nn
from torch.utils import data
from torchvision import transforms

from grayscalefolder import GrayscaleImageFolder
from model import GrayscaleToColorModel, Trainer


class Test_Data():
    @staticmethod
    def test_image(ckpt_path, path_to_image_folder, path_to_save, video):
        """
        Colorizes the given set of images present in the image folder.
        :param ckpt_path: Saved model file path.
        :param path_to_image_folder: Path that contains the images to color.
        :param path_to_save: Path where the colorized images should be saved.
        :param video: Specified whether images is being colorized or a video.
        """
        if video:
            test_transforms = transforms.Compose([transforms.Resize(224)])
            batch_size = 1

        else:
            test_transforms = transforms.Compose(
                [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()])
            batch_size = 64

        test_imagefolder = GrayscaleImageFolder(
            path_to_image_folder, test_transforms, video)

        test_loader = data.DataLoader(
            test_imagefolder, batch_size, shuffle=False)

        model = GrayscaleToColorModel()
        model.load_state_dict(torch.load(
            ckpt_path, map_location=torch.device('cpu')))
        model.eval()

        criterion = nn.MSELoss()
        save_images = True

        with torch.no_grad():
            losses = Trainer.validate_model(
                test_loader, model, criterion, save_images, path_to_save, "")
