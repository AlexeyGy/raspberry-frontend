from logging import info
import os
import datetime
from typing import List
import numpy as np
from flask import (
    Flask,
    Response,
    jsonify,
    request,
    send_from_directory,
    render_template,
)

import cv2 as cv

app = Flask(__name__, static_url_path="")

# ---  application parameters ---
CONFIDENCE_THRESHOLD = 0.5  # when we make the cutoff on the bounding boxes
OUTPUT_FOLDER = "static/images"  # where we output images to
INPUT_FOLDER = "model/"  # where we read the neural network from

# global variable containing the neural network state
NET = None


def _save_detection(rectangles, img: np.array, folder=OUTPUT_FOLDER):
    # save the image
    print(rectangles)
    if len(rectangles) > 0:
        for rectangle in rectangles:
            cv.rectangle(
                img,
                (rectangle[0], rectangle[1]),
                (rectangle[2], rectangle[3]),
                color=(0, 255, 0),
            )
            path_to_save_to = os.path.join(
                folder, f"{datetime.datetime.now().isoformat()}.jpg"
            )

            logging.info(f"saving detection to {path_to_save_to}")
            cv.imwrite(path_to_save_to, img)


@app.route("/upload", methods=["POST"])
def process():
    img = cv.imdecode(
        np.fromstring(request.files["webcam"].read(),
                      np.uint8), cv.IMREAD_COLOR
    )
    rectangles = recognize(img, NET)
    _save_detection(rectangles, img)
    return jsonify(rectangles)


@app.route("/", methods=["GET"])
def main():
    return send_from_directory("static", "index.html")


@app.route("/images")
def show_images():
    return render_template("images.html", images=os.listdir("static/images"))


if __name__ == '__main__':
    global net
    NET = set_up_inference()
    app.run()
