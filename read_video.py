import os
import sys

import cv2

from config_reader import ConfigReader
from test_data import Test_Data

image_size = 256


def main(argv):
    """
    Reads a video file, extracts frames, writes them in a folder and then feeds this folder for colorization to the CNN.
    :param argv: Video file name.
    """
    if len(argv) != 2:
        print("Usage: python read_video.py <video_file>")

    # read the video
    config = ConfigReader.read()
    cap = cv2.VideoCapture(config["test_video_directory"])

    # Write each video frame
    counter = 0
    while True:
        counter += 1
        success, frame = cap.read()
        if success:
            frame = cv2.resize(frame, (image_size, image_size))
            cv2.imwrite(config["video_frames_directory"] +
                        f"/root/{counter}.jpg", frame)
        else:
            break

    # Evaluate each frame using the model-> convert each frame from grayscale to color
    ckpt_path = "/Users/ankithaudupa/Desktop/Final Project CVPR/cs5330-cnn-colorization-master/checkpoints/model-epoch-95-losses-0.0024831692744046448"
    path_to_img = config["video_frames_directory"]
    path_to_save = {'grayscale': 'outputs/video_gray/',
                    'color': 'outputs/video_color/'}
    Test_Data.test_image(ckpt_path, path_to_img, path_to_save, True)
    cv2.destroyAllWindows()


def make_video():
    """
    Reads the folder that has predicted color images and creates a video using these images.
    """
    video_image_folder = "outputs/video_color"
    save_video_name = "output.avi"
    counter = -1
    images = []

    # glob images
    for image in os.listdir(video_image_folder):
        counter += 1
        if (image.endswith(".jpg") or image.endswith(".jpeg") or image.endswith(".png")):
            images.append(image)

    # Sort the images in order of frames
    def split_name(img_name):
        return int(img_name.split("-")[1])

    images.sort(key=split_name)

    # read each frame
    frame = cv2.imread(os.path.join(video_image_folder, images[0]))

    # setting the width and height of the frame
    h, w, l = frame.shape

    # create an instance of VideoWriter with 30 fps
    vid = cv2.VideoWriter(save_video_name, cv2.VideoWriter_fourcc(
        'M', 'J', 'P', 'G'), 30, (w, h))

    # Generating the video from each frame
    for img in images:
        vid.write(cv2.imread(os.path.join(video_image_folder, img)))

    cv2.destroyAllWindows()
    vid.release()


if __name__ == "__main__":
    main(sys.argv)
    make_video()
