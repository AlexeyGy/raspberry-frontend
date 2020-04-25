# import cv2 as cv
import numpy as np
from flask import Flask, request, jsonify, Response

import cv2 as cv

# global stuff
app = Flask(__name__)
net = None

# Load the model.
net = cv.dnn.readNet('face-detection-adas-0001.xml',
                     'face-detection-adas-0001.bin')
# Specify target device.
net.setPreferableTarget(cv.dnn.DNN_TARGET_MYRIAD)


def _recognize(img: np.ndarray) -> Response:
    """
    recognizes and transforms an image
    """
    # Read an image.
    if img is not None:

        rectangles = []
        # Prepare input blob and perform an inference.
        blob = cv.dnn.blobFromImage(
            img, size=(672, 384), ddepth=cv.CV_8U)
        net.setInput(blob)
        out = net.forward()
        # Draw detected faces on the img.
        for detection in out.reshape(-1, 7):
            confidence = float(detection[2])
            xmin = int(detection[3] * img.shape[1])
            ymin = int(detection[4] * img.shape[0])
            xmax = int(detection[5] * img.shape[1])
            ymax = int(detection[6] * img.shape[0])
            if confidence > 0.5:
                cv.rectangle(img, (xmin, ymin),
                             (xmax, ymax), color=(0, 255, 0))
                rectangles.append((xmin, ymin, xmax, ymax))
        # Save the img to an image file.
        # cv.imwrite('out.png', img)
        # print(cv.imencode('.jpeg', img))
        # return cv.imencode('.jpeg', img)
        return jsonify(rectangles)


@app.route('/upload', methods=['POST'])
def main():
    # _recognize_write_bounding_boxes(f)
    # resp = Response()
    resp = _recognize(cv.imdecode(np.fromstring(request.files['img'].read(), np.uint8),cv.IMREAD_COLOR))
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp
