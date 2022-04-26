import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
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
    plot_LAB(image, lab_image)
    return 0


if __name__ == "__main__":
    main(sys.argv)
