import sys
import io
import cv2
from cv2 import cvtColor
import imutils
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import read_image
from torchvision.transforms import RandomResizedCrop

image_size = 256


def main(argv):
    if len(argv) != 2:
        print("Usage: python read_video.py <video_file>")
    # read the video
    cap = cv2.VideoCapture(argv[1])
    # get the frame size
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        "./videos/output.avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (image_size, image_size))
    # display the video
    while True:
        success, frame = cap.read()
        if success:
            # add a border to the frame
            frame = cv2.copyMakeBorder(
                frame, int((width-height)/2), int((width-height)/2), 0, 0, cv2.BORDER_CONSTANT, None, value=0)
            # resize to 128x128
            frame = cv2.resize(frame, (image_size, image_size))
            # convert to LAB
            lab_image = read_image.convertToLAB(frame)
            L, A, B = cv2.split(lab_image)
            # display the LAB frames
            cv2.imshow('Original', frame)
            cv2.imshow('L', L)
            cv2.imshow('a', A)
            cv2.imshow('b', B)

            # combine the LAB images
            merged_image = read_image.combine(L, A, B)
            # save the frames
            writer.write(merged_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(sys.argv)
