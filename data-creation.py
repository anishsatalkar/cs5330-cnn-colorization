import os
import shutil

BASE_DIRECTORY_PATH = "/Users/anishsatalkar/Downloads/testSetPlaces205_resize"
BASE_DIRECTORY = "testSet_resize"


def main():
    for idx, filename in enumerate(os.listdir(f'{BASE_DIRECTORY_PATH}/{BASE_DIRECTORY}')):
        if idx < 1000:
            shutil.copy2(f"{BASE_DIRECTORY_PATH}/{BASE_DIRECTORY}/{filename}", f"/Users/anishsatalkar/Downloads/images/val/class/{filename}")
        else:
            shutil.copy2(f"{BASE_DIRECTORY_PATH}/{BASE_DIRECTORY}/{filename}", f"/Users/anishsatalkar/Downloads/images/train/class/{filename}")


if __name__ == '__main__':
    main()
