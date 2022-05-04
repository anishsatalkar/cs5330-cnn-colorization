import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.color import lab2rgb


class State(object):
    def __init__(self):
        self.val = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def update(self, val, factor=1):
        self.val = val
        self.sum += val * factor
        self.count += factor
        self.average = self.sum / self.count


class ConvertToRGB(object):
    @staticmethod
    def convert_to_rgb(gray, ab_img, path_to_save=None, save_name=None):
        plt.clf()
        color_img = torch.cat((gray, ab_img), 0).numpy()
        color_img = color_img.transpose((1, 2, 0))
        color_img[:, :, 0:1] = color_img[:, :, 0:1] * 100
        color_img[:, :, 1:3] = color_img[:, :, 1:3] * 255 - 128
        color_img = lab2rgb(color_img.astype(np.float64))
        # color_img = lab2rgb(color_img.squeeze().numpy())
        gray = gray.squeeze().numpy()
        if path_to_save is not None and save_name is not None:
            plt.imsave(arr=gray, fname=f"{path_to_save['grayscale']}{save_name}", cmap='gray')
            plt.imsave(arr=color_img, fname=f"{path_to_save['color']}{save_name}")
