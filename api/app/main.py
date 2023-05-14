import os
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from io import BytesIO
import tensorflow as tf

CARD_DETECTOR_MODEL_PATH = "models/my_model/saved_model"
TEXT_DETECTOR_MODEL_PATH = "models/my_text_detector/saved_model"


def get_bounding_boxes(detections, width, height):
    # This is the way I'm getting my coordinates
    boxes = detections['detection_boxes'][0]
    # get all boxes from an array
    max_boxes_to_draw = boxes.shape[0]
    # get scores to get a threshold
    scores = detections['detection_scores'][0]
    # this is set as a default but feel free to adjust it to your needs
    min_score_thresh = .5
    # # iterate over all objects found
    coordinates = []
    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        if scores[i] > min_score_thresh:
            coordinates.append({
                "box": boxes[i],
                "score": scores[i]
            })

    return coordinates


def create_app():
    app = FastAPI()

    origins = [
        "http://localhost:3000",  # Add the origin of your React app
        # Add more origins if needed
    ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/recognize_card")
    async def recognize_card(file: UploadFile = File(...)):
        current_file_path = os.path.abspath(__file__)

        current_directory = os.path.dirname(current_file_path)

        card_detector_directory = os.path.join(current_directory, CARD_DETECTOR_MODEL_PATH)

        detect_fn = tf.saved_model.load(card_detector_directory)

        contents = await file.read()

        image_np = Image.open(BytesIO(contents))
        image_np = np.array(image_np)
        image_np = image_np[:, :, :3]

        print("DEBUG", image_np.shape)

        input_tensor = tf.convert_to_tensor(image_np)
        input_tensor = input_tensor[tf.newaxis, ...]
        detections = detect_fn(input_tensor)

        bounding_boxes = get_bounding_boxes(detections)

        data = tf.nest.map_structure(lambda x: x.numpy().tolist() if isinstance(x, tf.Tensor) else x, bounding_boxes)

        return {"bounding_boxes": data}

    @app.post("/recognize_text")
    async def recognize_text(file: UploadFile = File(...)):
        current_file_path = os.path.abspath(__file__)

        current_directory = os.path.dirname(current_file_path)

        text_detector_directory = os.path.join(current_directory, TEXT_DETECTOR_MODEL_PATH)

        detect_fn = tf.saved_model.load(text_detector_directory)

        contents = await file.read()

        image_np = Image.open(BytesIO(contents))
        image_np = np.array(image_np)
        image_np = image_np[:, :, :3]

        (height, width, _) = image_np.shape

        input_tensor = tf.convert_to_tensor(image_np)
        input_tensor = input_tensor[tf.newaxis, ...]
        detections = detect_fn(input_tensor)

        bounding_boxes = get_bounding_boxes(detections, width, height)

        data = tf.nest.map_structure(lambda x: x.numpy().tolist() if isinstance(x, tf.Tensor) else x, bounding_boxes)

        return {"bounding_boxes": data}

    return app
