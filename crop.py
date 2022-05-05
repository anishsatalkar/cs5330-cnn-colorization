import cv2
import sys
import numpy as np


def main(argv):
    # load the image
    image = cv2.imread(argv[1])
    # crop the image
    cropped_image = image[60:754-70:, :]
    cropped_image = cropped_image[:, 240:1536-200]
    new_image = np.concatenate(
        (cropped_image[:, :160], cropped_image[:, 150*2:160*3]), axis=1)
    new_image2 = np.concatenate(
        (new_image, cropped_image[:, 155*4:160*5]), axis=1)
    new_image3 = np.concatenate((new_image2, cropped_image[:, 155*6:]), axis=1)

    # show the image
    cv2.imshow('image', new_image3)

    # write the image
    cv2.imwrite('.\Presentation_Report_Images\{}_cropped.png'.format(
        argv[1]), new_image3)

    # wait for a keypress
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
