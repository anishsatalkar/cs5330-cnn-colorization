import numpy as np
import cv2
import os
import sys
from matplotlib import pyplot as plt
from numpy.linalg import norm


def compare_images(original, predicted):
    # function to calculate the distance between two images
    # convert the images to Lab color space
    original_lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)
    predicted_lab = cv2.cvtColor(predicted, cv2.COLOR_BGR2LAB)
    L1, a1, b1 = cv2.split(original_lab)
    L2, a2, b2 = cv2.split(predicted_lab)
    # calculate the L2 norm between a and b
    L2_norm = np.sqrt(np.sum((a2 - a1) ** 2)) + np.sqrt(np.sum((b2 - b1) ** 2))
    # calculate the saturation differene netween the two images
    # convert to HSV and get the saturation channel
    sat1 = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)[:, :, 1]
    sat2 = cv2.cvtColor(predicted, cv2.COLOR_BGR2HSV)[:, :, 1]
    sat_diff = np.abs(np.sum(sat2)-np.sum(sat1))/np.sum(sat1)

    return L2_norm, sat_diff


def read_images(path):
    # read all images in the folder
    images = []
    for filename in os.listdir(path):
        if filename.endswith(".jpg"):
            image = cv2.imread(os.path.join(path, filename))
            # resize to 256x256 and then center crop to 224x224
            image = cv2.resize(image, (256, 256))[16:240, 16:240]
            images.append(image)
    return images


def compare_folders(original_path, predicted_path):
    # read the images in the folders
    original_images = read_images(original_path)
    predicted_images = read_images(predicted_path)
    # compare the images and sum the L2 norm and the saturation difference
    L2_norm = 0
    sat_diff = 0
    for original, predicted in zip(original_images, predicted_images):
        L2, sat_diff = compare_images(original, predicted)
        # print(f"L2 norm: {L2}, saturation difference: {sat_diff}")
        L2_norm += L2
        sat_diff += sat_diff

    return L2_norm/len(original_images), sat_diff/len(original_images)


def plot_images(original_path, predicted_path):
    # read the images in the folders
    original_images = read_images(original_path)
    # convert to RGB
    original_images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                       for image in original_images]
    predicted_images = read_images(predicted_path)
    # convert to RGB
    predicted_images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        for image in predicted_images]

    # show original image against predicted image using plt
    for i in range(1, 20, 2):
        plt.subplot(5, 4, i)
        plt.imshow(original_images[i // 2])
        plt.title(f"Original image {i//2}")
        plt.axis('off')
        plt.subplot(5, 4, i+1)
        plt.imshow(predicted_images[i // 2])
        plt.title(f"Predicted image {i//2}")
        plt.axis('off')
    plt.show()


def main(argv):
    # validate the arguments
    if len(argv) != 3:
        print("Usage: python3 evaluation.py <original_path> <predicted_path>")
        sys.exit(1)
    # get the paths to the original and predicted images
    original_path = argv[1]
    predicted_path = argv[2]
    # compare the images in the folders
    L2_norm, sat_diff = compare_folders(original_path, predicted_path)
    print(f"L2 norm: {L2_norm}, saturation difference: {sat_diff}")

    plot_images(original_path, predicted_path)

    # close the program when the user presses q
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(sys.argv)
