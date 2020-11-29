# this script performs a testwise detection of the content of test-images
import unittest

import cv2 as cv
from model import recognize, set_up_inference


class TestDetection(unittest.TestCase):
    def setUp(self):
        self.net = set_up_inference()

    def test_detect_givenFullBody_expectDetection(self):
        img = cv.imread("test-images/person-exists.jpg")
        self.assertEqual(len(recognize(img, self.net)), 1)

    def test_detect_givenJustFace_expectNoDetections(self):
        img = cv.imread("test-images/face-exists.jpg")
        self.assertEqual(recognize(img, self.net), [])
