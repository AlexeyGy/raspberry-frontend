import os
from logging import info
from typing import List

import numpy as np
import cv2 as cv

INPUT_FOLDER = "model/"  # where we read the neural network from

# the image size that the neural network is trained to work on.
# Should only be changed if the model is switched.
IMAGE_FORMAT = (544, 320)

# a pretrained model from OpenVino, see https://docs.openvinotoolkit.org/2018_R5/_docs_Retail_object_detection_pedestrian_rmnet_ssd_0013_caffe_desc_person_detection_retail_0013.html
NETWORK_NAME = "person-detection-retail-0013"
CONFIDENCE_THRESHOLD = 0.5  # when we make the cutoff on the bounding boxes


def set_up_inference():
    net = cv.dnn.readNet(
        os.path.join(INPUT_FOLDER, NETWORK_NAME + ".bin"),
        os.path.join(INPUT_FOLDER, NETWORK_NAME + ".xml"),
    )
    net.setPreferableTarget(cv.dnn.DNN_TARGET_MYRIAD)

    return net


def recognize(img: np.ndarray, net) -> List:
    """
    recognizes and transforms an image
    """
    # Read an image.
    if img is None:
        info("no image to process")
        return []
    # Prepare input blob and perform inference.
    net.setInput(cv.dnn.blobFromImage(img, size=IMAGE_FORMAT, ddepth=cv.CV_8U))
    net_output = net.forward()

    rectangles = []
    # see https://docs.openvinotoolkit.org/2019_R3.1/_models_intel_face_detection_adas_0001_description_face_detection_adas_0001.html

    detections = net_output.reshape(-1, 7)
    info(f"detected #{len(detections)} people")
    for detection in detections:
        confidence = float(detection[2])
        info(
            f"filtering out a detection, confidence {confidence} is below the threshold of {CONFIDENCE_THRESHOLD}"
        )
        if confidence >= CONFIDENCE_THRESHOLD:
            xmin = int(detection[3] * img.shape[1])
            ymin = int(detection[4] * img.shape[0])
            xmax = int(detection[5] * img.shape[1])
            ymax = int(detection[6] * img.shape[0])
            rectangles.append((xmin, ymin, xmax, ymax))
    return rectangles
