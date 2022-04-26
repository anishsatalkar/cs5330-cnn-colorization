import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision.transforms import RandomResizedCrop


def convertToLAB(image):
    # split the image into LAB color space
    return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2LAB)


def plot_LAB(image, lab_image):
    L, A, B = cv2.split(lab_image)
    # plot the LAB image
    plt.subplot(1, 4, 1)
    plt.imshow(image)
    plt.title('Original')
    plt.axis('off')
    plt.subplot(1, 4, 2)
    plt.imshow(L, cmap='gray')
    plt.title('L')
    plt.axis('off')
    plt.subplot(1, 4, 3)
    plt.imshow(A, cmap='RdYlGn')
    plt.title('A')
    plt.axis('off')
    plt.subplot(1, 4, 4)
    plt.imshow(B, cmap='YlGnBu')
    plt.title('B')
    plt.axis('off')
    plt.show()

    # save the images
    cv2.imwrite('./L.png', L)
    cv2.imwrite('./A.png', A)
    cv2.imwrite('./B.png', B)


def combine(L, A, B):
    # combine the LAB images
    lab_image = cv2.merge((L, A, B))
    # convert back to RGB
    return cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)


def plot_predicted_image(original_image, predicted_image):
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(original_image)
    ax[0].set_title('Original')
    ax[1].imshow(predicted_image)
    ax[1].set_title('Predicted')
    plt.show()


def main(argv):
    if len(argv) != 2:
        print("Usage: python read_image.py <image_file>")
        return 1
    # fetch the image file
    image_file = argv[1]
    # read the image
    image = Image.open(image_file)

    # random resize crop and save image
    randresizecrop = RandomResizedCrop(128)
    image = randresizecrop(image)
    lab_image = convertToLAB(image)
    merged_image = combine(
        lab_image[:, :, 0], lab_image[:, :, 1], lab_image[:, :, 2])
    # plot_LAB(image, lab_image, merged_image)
    # plot the original and predicted images
    plot_predicted_image(image, merged_image)

    return 0


if __name__ == "__main__":
    main(sys.argv)
