import re
import os

import cv2
import pandas as pd


def txt_to_csv(annotations_path: str, images_dir: str):
    dirname = os.getcwd()
    result = []

    for txt_file in os.listdir(annotations_path):
        gt_filename, _ = os.path.splitext(txt_file)
        img_filename = f"{gt_filename.split('_', 1)[1]}.jpg"
        img_path = os.path.join(dirname, images_dir, img_filename)
        im = cv2.imread(img_path)
        height, width, _ = im.shape

        with open(os.path.join(annotations_path, txt_file), mode="r") as file:
            while line := file.readline().rstrip():
                left, top, right, bottom, _ = re.split(' |, ', line)

                value = (
                    img_filename,
                    width,
                    height,
                    "Text",
                    int(left),
                    int(top),
                    int(right),
                    int(bottom),
                )

                result.append(value)

    column_name = [
        "filename",
        "width",
        "height",
        "class",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
    ]

    return pd.DataFrame(result, columns=column_name)
