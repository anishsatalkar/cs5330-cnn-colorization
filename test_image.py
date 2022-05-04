import os
import sys
import cv2
from config_reader import ConfigReader
from test_data import Test_Data

def main(argv):
    config = ConfigReader.read()
    ckpt_path = "/Users/ankithaudupa/Desktop/Final Project CVPR/cs5330-cnn-colorization-master/checkpoints/model-epoch-95-losses-0.0024831692744046448"
    path_to_img = config["test_data_directory"]
    path_to_save = {'grayscale': 'outputs/gray/', 'color': 'outputs/color/'}
    Test_Data.test_image(ckpt_path, path_to_img, path_to_save, False)


if __name__ == "__main__":
    main(sys.argv)