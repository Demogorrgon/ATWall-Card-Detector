import os
import io
from typing import Callable
from collections import namedtuple

import tensorflow as tf
from PIL import Image
from object_detection.utils import dataset_util, label_map_util


def class_text_to_int(row_label: str, label_map_file_name: str):
    curr_dir = os.getcwd()
    labels_path = os.path.join(curr_dir, os.pardir, f"annotations/{label_map_file_name}.pbtxt")
    label_map = label_map_util.load_labelmap(labels_path)
    label_map_dict = label_map_util.get_label_map_dict(label_map)

    return label_map_dict[row_label]


def split(df, group):
    data = namedtuple("data", ["filename", "object"])
    gb = df.groupby(group)

    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path, label_map_file_name):
    with tf.io.gfile.GFile(os.path.join(path, "{}".format(group.filename)), "rb") as fid:
        encoded_jpg = fid.read()

    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode("utf8")
    image_format = b"png"
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        # quickfix: do not add "Number" class for now
        if row["class"] == "Number":
            continue

        xmins.append(row["xmin"] / width)
        xmaxs.append(row["xmax"] / width)
        ymins.append(row["ymin"] / height)
        ymaxs.append(row["ymax"] / height)
        classes_text.append(row["class"].encode("utf8"))
        classes.append(class_text_to_int(row["class"], label_map_file_name))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        "image/height": dataset_util.int64_feature(height),
        "image/width": dataset_util.int64_feature(width),
        "image/filename": dataset_util.bytes_feature(filename),
        "image/source_id": dataset_util.bytes_feature(filename),
        "image/encoded": dataset_util.bytes_feature(encoded_jpg),
        "image/format": dataset_util.bytes_feature(image_format),
        "image/object/bbox/xmin": dataset_util.float_list_feature(xmins),
        "image/object/bbox/xmax": dataset_util.float_list_feature(xmaxs),
        "image/object/bbox/ymin": dataset_util.float_list_feature(ymins),
        "image/object/bbox/ymax": dataset_util.float_list_feature(ymaxs),
        "image/object/class/text": dataset_util.bytes_list_feature(classes_text),
        "image/object/class/label": dataset_util.int64_list_feature(classes),
    }))

    return tf_example


def create_tf_record(
    images_dir: str,
    tf_record_filename: str,
    annotations_path: str,
    label_map_file_name: str,
    processor: Callable[[str, str], tf.train.Example],
):
    curr_dir = os.getcwd()
    annotations_full_path = os.path.join(curr_dir, os.pardir, annotations_path)
    tfrecord_path = os.path.join(curr_dir, os.pardir, f"annotations/{tf_record_filename}.tfrecord")

    writer = tf.io.TFRecordWriter(tfrecord_path)
    images_raw_path = os.path.join(curr_dir, os.pardir, images_dir)
    examples = processor(annotations_full_path, images_dir)

    grouped = split(examples, "filename")

    for group in grouped:
        tf_example = create_tf_example(group, images_raw_path, label_map_file_name)
        writer.write(tf_example.SerializeToString())

    writer.close()
