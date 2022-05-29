import os
from shutil import copyfile
import math
import random


def main():
    curr_dirname = os.path.dirname(os.path.abspath(__file__))
    source = os.path.join(curr_dirname, os.pardir, "images/raw")
    dest = os.path.join(curr_dirname, os.pardir, "images")

    train_dir = os.path.join(dest, "train")
    test_dir = os.path.join(dest, "test")

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    files = os.listdir(source)

    num_files = len(files)
    num_test_files = math.ceil(0.1 * num_files)

    for i in range(num_test_files):
        idx = random.randint(0, len(files) - 1)
        filename = files[idx]
        copyfile(
            os.path.join(source, filename),
            os.path.join(test_dir, filename)
        )
        files.remove(files[idx])

    for file in files:
        copyfile(
            os.path.join(source, file),
            os.path.join(train_dir, file)
        )


if __name__ == "__main__":
    main()
