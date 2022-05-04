import os

from config_reader import ConfigReader


def main():
    """
    Creates the required directories in the project.
    """
    config = ConfigReader.read()
    os.makedirs('outputs/color', exist_ok=True)
    os.makedirs('outputs/gray', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs(f"{config['train_val_split_dir']}/train/class", exist_ok=True)
    os.makedirs(f"{config['train_val_split_dir']}/val/class", exist_ok=True)


if __name__ == '__main__':
    main()
