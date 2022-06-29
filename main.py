import time

import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

PATH_TO_SAVED_MODEL_DIR = "exported-models/my_model/saved_model"
PATH_TO_LABELS = "annotations/label_map.pbtxt"
IMAGE_PATHS = [
    "example/images/img.png",
    # "example/images/img_1.png",
    # "example/images/img_2.png",
    # "example/images/img_3.png",
    # "example/images/img_4.png",
    # "example/images/img_5.png",
    # "example/images/img_6.png",
]


def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(path))


for image_path in IMAGE_PATHS:
    print("Loading model...", end="")
    start_time = time.time()

    detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL_DIR)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Done! Took {} seconds".format(elapsed_time))

    print("Running inference for {}... ".format(image_path), end="")

    image_np = load_image_into_numpy_array(image_path)
    image_np = image_np[:, :, :3]

    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop("num_detections"))
    detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}
    detections["num_detections"] = num_detections

    detections["detection_classes"] = detections["detection_classes"].astype(np.int64)

    image_np_with_detections = image_np.copy()

    category_index = label_map_util.create_category_index_from_labelmap(
        PATH_TO_LABELS,
        use_display_name=True
    )

    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detections["detection_boxes"],
          detections["detection_classes"],
          detections["detection_scores"],
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=200,
          min_score_thresh=.30,
          agnostic_mode=False)

    cv2.imshow("image", image_np_with_detections)
    cv2.waitKey(0)
    print("Done")
