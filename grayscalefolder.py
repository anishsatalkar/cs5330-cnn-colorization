import numpy as np
import torch
from torchvision import datasets
from skimage.color import rgb2lab, rgb2gray


class GrayscaleImageFolder(datasets.ImageFolder):
    def __init__(self, path_to_img, transform, test_video):
        self.test_video = test_video
        # super(GrayscaleImageFolder, self).__init__()
        super(GrayscaleImageFolder, self).__init__(path_to_img, transform)
        self.counter = 1

    def __getitem__(self, item):
        path, target = self.imgs[item]
        if self.test_video:
            path = path.split("root/")[0]+f"root/{self.counter}.jpg"
            self.counter += 1
        img = self.loader(path)
        img_original = None
        img_ab = None
        if self.transform is not None:
            img_original = self.transform(img)
            img_original = np.asarray(img_original)

            img_lab = rgb2lab(img_original)
            img_lab = (img_lab + 128) / 255

            img_ab = img_lab[:, :, 1:3]
            img_ab = torch.from_numpy(img_ab.transpose((2, 0, 1))).float()

            img_original = rgb2gray(img_original)
            img_original = torch.from_numpy(img_original).unsqueeze(0).float()

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img_original, img_ab, target

