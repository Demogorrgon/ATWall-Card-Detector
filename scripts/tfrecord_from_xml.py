import os
import glob
import pandas as pd
import io
import xml.etree.ElementTree as ET

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"    # Suppress TensorFlow logging (1)
import tensorflow as tf
from PIL import Image
from object_detection.utils import dataset_util, label_map_util
from collections import namedtuple

curr_dirname = os.path.dirname(os.path.abspath(__file__))


def xml_to_csv(annotations_path, images_dir):
    xml_list = []

    for xml_file in glob.glob(annotations_path + "/*.xml"):
        tree = ET.parse(xml_file)
        images_data = tree.iter("image")

        files = os.listdir(os.path.join(curr_dirname, os.pardir, images_dir))

        for idxx, file in enumerate(files):
            for idx, image_data in enumerate(images_data):
                name = image_data.attrib["name"]

                if file == name:
                    print(idxx, file)

                    try:
                        width = image_data.attrib["width"]
                        height = image_data.attrib["height"]

                        for box in image_data.findall("box"):
                            xbr = int(float(box.attrib["xbr"]))
                            xtl = int(float(box.attrib["xtl"]))
                            ybr = int(float(box.attrib["ybr"]))
                            ytl = int(float(box.attrib["ytl"]))

                            if xbr < xtl:
                                xmax = xtl
                                xmin = xbr
                            else:
                                xmin = xtl
                                xmax = xbr

                            if ybr < ytl:
                                ymax = ytl
                                ymin = ybr
                            else:
                                ymin = ytl
                                ymax = ybr

                            value = (
                                name,
                                width,
                                height,
                                box.attrib["label"],
                                int(xmin),
                                int(ymin),
                                int(xmax),
                                int(ymax),
                            )

                            xml_list.append(value)
                    except AttributeError as e:
                        print(f"Error {e}")
                    break

    column_name = ["filename", "width", "height",
                   "class", "xmin", "ymin", "xmax", "ymax"]

    xml_df = pd.DataFrame(xml_list, columns=column_name)

    return xml_df


def class_text_to_int(row_label):
    labels_path = os.path.join(curr_dirname, os.pardir, "annotations/label_map.pbtxt")
    label_map = label_map_util.load_labelmap(labels_path)
    label_map_dict = label_map_util.get_label_map_dict(label_map)

    return label_map_dict[row_label]


def split(df, group):
    data = namedtuple("data", ["filename", "object"])
    gb = df.groupby(group)

    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
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
        xmins.append(row["xmin"] / width)
        xmaxs.append(row["xmax"] / width)
        ymins.append(row["ymin"] / height)
        ymaxs.append(row["ymax"] / height)
        classes_text.append(row["class"].encode("utf8"))
        classes.append(class_text_to_int(row["class"]))

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


def create_tf_record(images_dir, tf_record_filename):
    annotations_path = os.path.join(curr_dirname, os.pardir, "annotations")
    tfrecord_path = os.path.join(curr_dirname, os.pardir, f"annotations/{tf_record_filename}.tfrecord")

    writer = tf.io.TFRecordWriter(tfrecord_path)
    images_raw_path = os.path.join(curr_dirname, os.pardir, images_dir)
    examples = xml_to_csv(annotations_path, images_dir)

    grouped = split(examples, "filename")

    for group in grouped:
        tf_example = create_tf_example(group, images_raw_path)
        writer.write(tf_example.SerializeToString())

    writer.close()


def main():
    create_tf_record(images_dir="images/train", tf_record_filename="1_342_train")
    create_tf_record(images_dir="images/test", tf_record_filename="1_342_test")


if __name__ == "__main__":
    main()
