import abc
import argparse
import cv2
from io import BytesIO
import numpy as np
import os
from sklearn.externals import joblib
from time import sleep

try:
    from gopigo import *
except ImportError:
    print("gopigo not found")

LABELS = {'left': 0, 'right': 1, 'straight': 2}
RESIZE = 30     # resize images to this size
RESOLUTION = 100  # original image resolution
DEFAULT_SPEED = 50

class Driver(object):
    __metaclass__ = abc.ABCMeta
    def __init__(self, modelfile, imagefile=None, nodrive=False):
        """
        modelfile: path to model file, used by load_model()
        imagefile: when given, don't predict via cam, but only with given imagefile
        nodrive: when True, do not drive, only print out prediction
        """
        self.imagefile = imagefile
        self.nodrive = nodrive
        self.resize = RESIZE
        self.resolution = RESOLUTION
        self.model = self.load_model(modelfile)
        if not self.imagefile:
            self.cam = self._setup_cam()

    @abc.abstractmethod
    def load_model(self, modelfile):
        """Load model and return"""
        pass

    @abc.abstractmethod
    def predict(self, x):
        """Make prediction with self.model and x and return single value y"""
        pass

    def run(self):
        """call this function after the constructor"""
        if self.imagefile:
            img = self._postprocess(self._read_image())
            self.predict(img)
        else:
            self.drive()

    def drive(self):
        """when not predicting on a single image,
        we will go into this function to continuously drive."""
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


    def _capture(self):
        """capture one image and return as 1D numpy array"""
        stream = BytesIO()
        self.cam.capture(stream, 'jpeg')
        data = np.fromstring(stream.getvalue(), dtype=np.uint8)
        # "Decode" the image preserving color
        img = cv2.imdecode(data, 1)
        # switch BGR order to RGB order
        img = img[:, :, ::-1]

        # resize image to match training size
        img = cv2.resize(img, (self.resize, self.resize), interpolation=cv2.INTER_AREA)

    #    cv2.imshow('image',img)
    #    cv2.waitKey(0)
    #    cv2.destroyAllWindows()
        return img.flatten()
