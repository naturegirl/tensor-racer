import argparse
from io import BytesIO
import cv2
try:
    from gopigo import *
except ImportError:
    print("gopigo not found")
import numpy as np
import os
from sklearn.externals import joblib
from time import sleep


LABELS = {'left': 0, 'right': 1, 'straight': 2}
RESIZE = 30     # resize images to this size
RESOLUTION = 100  # original image resolution
DEFAULT_SPEED = 50

class Driver(object):
    def __init__(self, imagefile=None, nodrive=False):
        """
        imagefile: when given, don't predict via cam, but only with given imagefile
        nodrive: when True, do not drive, only print out prediction
        """
        self.imagefile = imagefile
        self.nodrive = nodrive
        self.resize = RESIZE
        self.resolution = RESOLUTION
        print("in driver")
        if not self.imagefile:
            self.cam = self._setup_cam()

    def load_model(self, modelfile):
        """implemented by subclass. Will load model and safe as self.model"""
        return None

    def predict(self, x):
        """implemented by subclass. Will make prediction with
        self.model and x, and return a y."""
        return 0

    def _setup_cam(self):
        import picamera
        cam = picamera.PiCamera()
        cam.resolution = (self.resolution, self.resolution)
        sleep(1)
        return cam

    def _scale_to_zero_one(self, img):
        """convert pixel values from [0, 255] -> [0.0, 1.0]"""
        if img.dtype == np.uint8:
            img = img.astype(np.float32)
            return np.multiply(img, 1.0 / 255.0)
        else:
            print("image values already seem to be float")
            return img

    def _postprocess(self, img):
        """common postprocessing, whether img is captured or from file"""
        img = self._scale_to_zero_one(img)
        img = img.reshape(1, -1)  # to avoid a scikit-learn deprecation warning later
        return img

    def _read_image(self):
        if not os.path.isfile(self.imagefile):
            raise Exception("image file does not exist")
        img = cv2.imread(self.imagefile)
        if img.shape[:2] != (self.resize, self.resize):
            img = cv2.resize(img, (self.resize, self.resize),
                             interpolation=cv2.INTER_AREA)
        return img.flatten()

    def run(self):
        """call this function after the constructor"""
        if self.imagefile:
            img = self._postprocess(self._read_image())
            self.predict(img)
        else:
            self.drive()

    def drive(self):
        if not self.nodrive:
            fwd()
            set_speed(DEFAULT_SPEED)
        while True:
            img = self._postprocess(self._capture())
            print img
            y = self.predict(img)
            if y == LABELS['left']:
                print "left"
                if not self.nodrive:
                    set_left_speed(0)
                    set_right_speed(DEFAULT_SPEED)
            elif y == LABELS['right']:
                print "right"
                if not self.nodrive:
                    set_right_speed(0)
                    set_left_speed(DEFAULT_SPEED)
            elif y == LABELS['straight']:
                print "straight"
                if not self.nodrive:
                    fwd()
                    set_speed(DEFAULT_SPEED)


    def _capture(self):
        """capture one image and return as 1D numpy array"""
        stream = BytesIO()
        cam.capture(stream, 'jpeg')
        data = np.fromstring(stream.getvalue(), dtype=np.uint8)
        # "Decode" the image preserving color
        img = cv2.imdecode(data, 1)
        # switch BGR order to RGB order
        img = img[:, :, ::-1]

        # resize image to match training size
        img = cv2.resize(img, (args.resize, args.resize), interpolation=cv2.INTER_AREA)
        print("done resizing")

    #    cv2.imshow('image',img)
    #    cv2.waitKey(0)
    #    cv2.destroyAllWindows()
        return img.flatten()
