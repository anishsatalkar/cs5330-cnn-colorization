import numpy as np
import torch
from skimage.color import rgb2gray, rgb2lab
from torchvision import datasets


class GrayscaleImageFolder(datasets.ImageFolder):
    """
    This class extends the ImageFolder class so that once the images are read, they are separated
    into original and AB components.
    """

    def __init__(self, path_to_img, transform, test_video):
        self.test_video = test_video
        # super(GrayscaleImageFolder, self).__init__()
        super(GrayscaleImageFolder, self).__init__(path_to_img, transform)
        self.counter = 1

    def __getitem__(self, item):
        """
        Reads images from the given path and splits it into original and AB components.
        :param item: Internally passed.
        :return: original image, AB component of the image and the target.
        """
        path_to_img, target = self.imgs[item]

        if self.test_video:
            path_to_img = path_to_img.split(
                "root/")[0]+f"root/{self.counter}.jpg"
            self.counter += 1

        image_original = None
        image_ab_component = None
        image = self.loader(path_to_img)

        if self.transform is not None:
            image_original = self.transform(image)
            image_original = np.asarray(image_original)

            image_lab_format = rgb2lab(image_original)
            image_lab_format = (image_lab_format + 128) / 255

            image_original = rgb2gray(image_original)
            image_original = torch.from_numpy(
                image_original).unsqueeze(0).float()

            image_ab_component = image_lab_format[:, :, 1:3]
            image_ab_component = torch.from_numpy(
                image_ab_component.transpose((2, 0, 1))).float()

        if self.target_transform is not None:
            target = self.target_transform(target)

        return image_original, image_ab_component, target
