# this script performs a testwise detection of the content of test-images
import unittest

import cv2 as cv
from main import recognize


class TestDetection(unittest.TestCase):

    def setUp(self):
        model_name = 'person-detection-retail-0013'
        input_folder = 'model/'

        net = cv.dnn.readNet(input_folder + model_name + '.bin',
                             input_folder + model_name + '.xml')

        # Specify target device, cv.dnn.DNN_TARGET_MYRIAD for the Movidius and cv2.dnn.DNN_TARGET_CPU for the CPU
        net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

    def test_detect_givenFullBody_expectDetection(self):
        img = cv.imread('test-images/person-exists.jpg')
        self.assertEqual(len(recognize(img)), 1)

    def test_detect_givenJustFace_expectNoDetections(self):
        img = cv.imread('test-images/face-exists.jpg')
        self.assertEqual(recognize(img), [])
