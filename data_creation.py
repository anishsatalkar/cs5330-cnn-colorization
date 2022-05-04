# Imports
import os
import shutil

from config_reader import ConfigReader

# Base directory where the training images are present.
BASE_DIRECTORY = "testSet_resize"


def main():
    """
    Splits the training data into 40k training images and 1k validation images.
    """
    config = ConfigReader.read()
    for idx, filename in enumerate(os.listdir(f'{config["training_data_directory"]}/{BASE_DIRECTORY}')):
        if idx < 1000:
            shutil.copy2(f'{config["training_data_directory"]}/{BASE_DIRECTORY}/{filename}',
                         f"{config['train_val_split_dir']}/val/class/{filename}")
        else:
            shutil.copy2(f'{config["training_data_directory"]}/{BASE_DIRECTORY}/{filename}',
                         f"{config['train_val_split_dir']}/train/class/{filename}")


if __name__ == '__main__':
    main()
