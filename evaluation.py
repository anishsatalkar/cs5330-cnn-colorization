import cv2
import numpy as np

# function to calculate the distance between two images


def compare_images(original, predicted):
    L1, a1, b1 = cv2.split(original)
    L2, a2, b2 = cv2.split(predicted)
    # calculate the L2 norm between a and b
    L2_norm = np.linalg.norm(a2-a1) + np.linalg.norm(b2-b1)
    # calculate the saturation differene netween the two images
    # convert to HSV and get the saturation channel
    sat1 = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)[:, :, 1]
    sat2 = cv2.cvtColor(predicted, cv2.COLOR_BGR2HSV)[:, :, 1]
    sat_diff = np.abs(np.sum(sat2)-np.sum(sat1))/np.sum(sat1)

    return L2_norm, sat_diff
