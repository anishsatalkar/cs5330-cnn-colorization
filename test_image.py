import sys

from config_reader import ConfigReader
from test_data import Test_Data


def main(argv):
    """
    Driver to colorize images.
    :param argv:
    """
    config = ConfigReader.read()
    ckpt_path = "checkpoints\model-epoch-95-losses-0.0024831692744046448"
    print("model loaded")
    path_to_img = config["test_data_directory"]
    path_to_save = {'grayscale': 'outputs_test/gray/',
                    'color': 'outputs_test/color/'}
    Test_Data.test_image(ckpt_path, path_to_img, path_to_save, False)


if __name__ == "__main__":
    main(sys.argv)
