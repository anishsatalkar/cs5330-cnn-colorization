# Imports
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision.transforms import RandomResizedCrop


def convertToLAB(image):
    """
    Converts a given image from BGR to LAB.
    :param image: Input BGR image.
    :return: Converted LAB image.
    """
    # split the image into LAB color space
    return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2LAB)


def plot_LAB(image, lab_image):
    """
    Plots the original and LAB image.
    :param image: Original image.
    :param lab_image: LAB image.
    :return: Figure that has these plots.
    """
    figure = plt.figure(figsize=(1, 4))

    L, A, B = cv2.split(lab_image)
    # plot the LAB image
    figure.add_subplot(1, 4, 1)
    plt.imshow(image)
    plt.title('Original')
    plt.axis('off')
    figure.add_subplot(1, 4, 2)
    plt.imshow(L, cmap='gray')
    plt.title('l')
    plt.axis('off')
    figure.add_subplot(1, 4, 3)
    plt.imshow(A, cmap='RdYlGn')
    plt.title('a')
    plt.axis('off')
    figure.add_subplot(1, 4, 4)
    plt.imshow(B, cmap='YlGnBu')
    plt.title('b')
    plt.axis('off')

    return figure


def save_images(lab_image):
    """
    Saves the given LAB image. Save location is './images'
    :param lab_image: Given LAB image.
    """
    # save the images
    L, A, B = cv2.split(lab_image)
    cv2.imwrite('.images/L.png', L)
    cv2.imwrite('.images/A.png', A)
    cv2.imwrite('.images/B.png', B)


def combine(L, A, B):
    """
    Combines L, A and B components of an image into a single image.
    :param L: Lightness component.
    :param A: A component of the image.
    :param B: B component of the image.
    :return: Combined LAB image.
    """
    # combine the LAB images
    lab_image = cv2.merge((L, A, B))
    # convert back to RGB
    return cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)


def plot_predicted_image(original_image, predicted_image):
    """
    Plots the original and predicted image.
    :param original_image: Original image data.
    :param predicted_image: Predicted image data.
    """
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(original_image)
    ax[0].set_title('Original')
    ax[0].axis('off')
    ax[1].imshow(predicted_image)
    ax[1].set_title('Predicted')
    ax[1].axis('off')
    plt.show()


def main(argv):
    if len(argv) != 2:
        print("Usage: python read_image.py <image_file>")
        return 1
    # fetch the image file
    image_file = argv[1]
    # read the image
    image = Image.open(image_file)

    # resize to 256x256 and then center crop to 224x224
    image = cv2.resize(image, (256, 256))[16:240, 16:240]
    lab_image = convertToLAB(image)
    merged_image = combine(
        lab_image[:, :, 0], lab_image[:, :, 1], lab_image[:, :, 2])
    # plot_LAB(image, lab_image, merged_image)
    # plot the original and predicted images
    plot_predicted_image(image, merged_image)
    # save the images
    save_images(lab_image)
    return 0


if __name__ == "__main__":
    main(sys.argv)
