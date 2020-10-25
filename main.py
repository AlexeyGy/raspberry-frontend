import logging
import os
import datetime

import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory, render_template

import cv2 as cv

# threshold for detections to show up
CONFIDENCE_THRESHOLD = .5
OUTPUT_FOLDER = "static/images"

# flask app
app = Flask(__name__, static_url_path='')

# Load the model.
# options: person-detection-retail-0013 face-detection-adas-0001
model_name = 'person-detection-retail-0013'
input_folder = 'model/'
net = cv.dnn.readNet(input_folder + model_name + '.bin',
                     input_folder + model_name + '.xml')
IMAGE_FORMAT = (544, 320)

# Specify target device, cv.dnn.DNN_TARGET_MYRIAD for the Movidius and cv2.dnn.DNN_TARGET_CPU for the CPU
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)


def _recognize(img: np.ndarray) -> Response:
    """
    recognizes and transforms an image
    """
    # Read an image.
    if img is None:
        logging.INFO("no image to process")
        return []
    # Prepare input blob and perform inference.
    net.setInput(
        cv.dnn.blobFromImage(
            img, size=IMAGE_FORMAT, ddepth=cv.CV_8U
        )
    )
    net_output = net.forward()

    rectangles = []
    # see https://docs.openvinotoolkit.org/2019_R3.1/_models_intel_face_detection_adas_0001_description_face_detection_adas_0001.html

    detections = net_output.reshape(-1, 7)
    logging.info(f"detected #{len(detections)} people")
    for detection in detections:
        confidence = float(detection[2])
        logging.info(
            f"filtering out a detection, confidence {confidence} is below the threshold of {CONFIDENCE_THRESHOLD}")
        if confidence >= CONFIDENCE_THRESHOLD:
            xmin = int(detection[3] * img.shape[1])
            ymin = int(detection[4] * img.shape[0])
            xmax = int(detection[5] * img.shape[1])
            ymax = int(detection[6] * img.shape[0])
            rectangles.append((xmin, ymin, xmax, ymax))
    return rectangles


def _save_detection(rectangles, img: np.array):
    # save the image
    print(rectangles)
    if len(rectangles) > 0:
        for rectangle in rectangles:
            cv.rectangle(img, (rectangle[0], rectangle[1]),
                         (rectangle[2], rectangle[3]), color=(0, 255, 0))
            path_to_save_to = os.path.join(
                OUTPUT_FOLDER, f'{datetime.datetime.now().isoformat()}.jpg')

            logging.info(f'saving detection to {path_to_save_to}')
            cv.imwrite(path_to_save_to, img)


@app.route('/upload', methods=['POST'])
def process():
    img = cv.imdecode(
        np.fromstring(
            request.files['webcam'].read(), np.uint8
        ),
        cv.IMREAD_COLOR
    )
    rectangles = _recognize(img)
    _save_detection(rectangles, img)
    return jsonify(rectangles)


@app.route('/', methods=['GET'])
def main():
    return send_from_directory('static', 'index.html')


@app.route('/images')
def hello():
    return render_template('images.html', images=os.listdir('static/images'))
