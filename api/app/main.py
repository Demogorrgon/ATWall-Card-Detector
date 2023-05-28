from typing import TypedDict, List

import os
import numpy as np
from io import BytesIO

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import tensorflow as tf


# TODO: place into env vars
CARD_DETECTOR_MODEL_PATH = "models/my_model/saved_model"
TEXT_DETECTOR_MODEL_PATH = "models/my_text_detector/saved_model"


class Coordinates(TypedDict):
    box: List[float]
    score: List[float]


class RecognitionResult(TypedDict):
    bounding_boxes: List[Coordinates]


def get_bounding_boxes(detections) -> List[Coordinates]:
    """Takes raw prediction result of an object detection model
    and returns most probable collection of bounding boxes"""

    # This is the way I'm getting my coordinates
    boxes = detections['detection_boxes'][0]
    # get all boxes from an array
    max_boxes_to_draw = boxes.shape[0]
    # get scores to get a threshold
    scores = detections['detection_scores'][0]
    # this is set as a default but feel free to adjust it to your needs
    min_score_thresh = .5
    # # iterate over all objects found
    coordinates: List[Coordinates] = []
    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        if scores[i] > min_score_thresh:
            coordinates.append(
                Coordinates(
                    box=boxes[i],
                    score=scores[i],
                )
            )

    return coordinates


async def predict_and_get_bb(file, model_path) -> List[Coordinates]:
    """Loads the model by its path, then runs prediction on an image file. Returns bounding boxes data"""

    current_file_path = os.path.abspath(__file__)

    current_directory = os.path.dirname(current_file_path)

    model_directory = os.path.join(current_directory, model_path)

    model_fn = tf.saved_model.load(model_directory)

    contents = await file.read()

    image_np = Image.open(BytesIO(contents))
    image_np = np.array(image_np)
    image_np = image_np[:, :, :3]

    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = model_fn(input_tensor)

    bounding_boxes = get_bounding_boxes(detections)

    return tf.nest.map_structure(lambda x: x.numpy().tolist() if isinstance(x, tf.Tensor) else x, bounding_boxes)


def create_app():
    app = FastAPI()

    origins = [
        "http://localhost:3000",
    ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/recognize_card")
    async def recognize_card(file: UploadFile = File(...)) -> RecognitionResult:
        data = await predict_and_get_bb(file=file, model_path=CARD_DETECTOR_MODEL_PATH)

        return {"bounding_boxes": data}

    @app.post("/recognize_text")
    async def recognize_text(file: UploadFile = File(...)) -> RecognitionResult:
        data = await predict_and_get_bb(file=file, model_path=TEXT_DETECTOR_MODEL_PATH)

        return {"bounding_boxes": data}

    return app
