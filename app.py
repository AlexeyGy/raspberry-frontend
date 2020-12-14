import datetime
import os
import logging
from logging import info

from typing import List

import cv2 as cv
import numpy as np
from flask import (
    Flask,
    jsonify,
    request,
    send_from_directory,
    render_template,
)

from model import recognize, set_up_inference

app = Flask(__name__, static_url_path="")
logging.basicConfig(level=logging.INFO)

# ---  application parameters ---
OUTPUT_FOLDER = "static/images"  # where we output images to

# --- initialize the globally scoped neural network --
NET = set_up_inference()


def _save_detection(rectangles: List, img: np.array, folder=OUTPUT_FOLDER):
    if not rectangles:
        return

    info(f"received {rectangles} to process")
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

        info(f"saving detection to {path_to_save_to}")
        cv.imwrite(path_to_save_to, img)


@app.route("/upload", methods=["POST"])
def process():

    info("processing")
    img = cv.imdecode(
        np.fromstring(request.files["webcam"].read(), np.uint8), cv.IMREAD_COLOR
    )
    rectangles = recognize(img, NET)
    _save_detection(rectangles, img)
    return jsonify(rectangles)


@app.route("/", methods=["GET"])
def main():
    return send_from_directory("static", "index.html")


@app.route("/images")
def show_images():
    return render_template("images.html", images=os.listdir(OUTPUT_FOLDER))


if __name__ == "__main__":
    app.run()
