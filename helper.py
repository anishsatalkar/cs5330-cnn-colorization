import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.color import lab2rgb


class State(object):
    """
    Maintains the state of a training session.
    """

    def __init__(self):
        self.value = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def update_state(self, val, factor=1):
        self.value = val
        self.sum += val * factor
        self.count += factor
        self.average = self.sum / self.count


class ConvertToRGB(object):
    @staticmethod
    def convert_to_rgb(gray_image, ab_img, path_to_save=None, save_name=None):
        """
        Combines the given gray and AB components into an RGB image.
        :param gray_image: Grayscale image.
        :param ab_img: AB component of an image.
        :param path_to_save: Location to save the combined image.
        :param save_name: The file name used to save the image.
        """
        plt.clf()

        color_image = torch.cat((gray_image, ab_img), 0).numpy()
        color_image = color_image.transpose((1, 2, 0))
        color_image[:, :, 0:1] = color_image[:, :, 0:1] * 100
        color_image[:, :, 1:3] = color_image[:, :, 1:3] * 255 - 128
        color_image = lab2rgb(color_image.astype(np.float64))

        gray_image = gray_image.squeeze().numpy()

        if path_to_save is not None and save_name is not None:
            plt.imsave(
                arr=gray_image, fname=f"{path_to_save['grayscale']}{save_name}", cmap='gray')
            plt.imsave(arr=color_image,
                       fname=f"{path_to_save['color']}{save_name}")
