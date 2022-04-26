import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import read_image
from torchvision.transforms import RandomResizedCrop


def main(argv):
    if len(argv) != 2:
        print("Usage: python read_video.py <video_file>")
    # read the video
    video_reader = cv2.VideoCapture(argv[1])
    height, width, channels = video_reader.read()[1].shape
    video_writer = cv2.VideoWriter(
        './output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (height, width))

    print("Video", argv[1], "read")

    # read the frames
    success = True
    while success:
        success, image = video_reader.read()
        print(type(image))
        lab_image = read_image.convertToLAB(image)
        video_writer.write(lab_image)  # write the frame to the video file
        cv2.imshow('image', lab_image)
    # display the video
    video_writer.release()
    video_reader.release()


if __name__ == "__main__":
    main(sys.argv)
