# this script performs a testwise detection of the content of test-images

import cv2 as cv
import glob
from main import _recognize, _save_detection
import logging
# Load the model.
logging.basicConfig(level="INFO")


model_name = 'person-detection-retail-0013'  # options: person-detection-retail-0013 face-detection-adas-0001
input_folder = 'model/'

net = cv.dnn.readNet(input_folder + model_name + '.bin',
                     input_folder + model_name + '.xml')

# Specify target device, cv.dnn.DNN_TARGET_MYRIAD for the Movidius and cv2.dnn.DNN_TARGET_CPU for the CPU
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# Read an image.
for img_name in glob.glob('test-images/*'):
    img = cv.imread(img_name)
    if img is None:
        raise Exception('Image not found!')
    _save_detection(_recognize(img), img)
