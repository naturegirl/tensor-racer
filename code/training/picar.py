"""Functions for reading training data and surfacing via a Dataset class"""

# Example Usage:
#    picar = read_data_sets("/Users/naturegirl/code/tensor-racer/data/round1)
#    print picar.train.labels
#    print picar.validation.images
#
# The API is heavily borrowed from the mnist dataset interface at:
#   https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/datasets/mnist.py

import collections
import cv2
import numpy as np
import os
from tensorflow.python.framework import dtypes

ORIGINAL_SIZE = 100 # original images are 100x100
LABELS = {'left': 0, 'right': 1, 'straight': 2}

class DataSet(object):
    # TODO: add boolean: greyscale?
    def __init__(self, images, labels, dtype=dtypes.float32):
        if dtype == dtypes.float32:
            # Convert from [0, 255] -> [0.0, 1.0].
            images = images.astype(np.float32)
            images = np.multiply(images, 1.0 / 255.0)
        if len(labels) != images.shape[0]:
            raise Exception("number of labels and images doesn't match")
        self._images = images
        self._labels = labels
        self._num_examples = len(labels)
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        # TODO
        return 0

Datasets = collections.namedtuple('Datasets', ['train', 'validation'])

def _load_images(data_dir):
    """read images from data_dir and return list of labels and images as matrix.
    For the matrix each row is one training data sample, in the order the
    training data was taken."""
    files = _list_files(data_dir)
    labels = [triple[2] for triple in files]
    num_rows = len(labels)
    images = [_read_image(os.path.join(data_dir, fpath)) for _, fpath, _ in files]
    matrix = np.concatenate(images).reshape(num_rows, 30*30*3)
    return labels, matrix

def _read_image(path, resize=True, size=30, rgb=True):
    """read one image and return it as a one dimensional numpy array
    path: path to image:
    resize: whether the image should be resized
    size: image will be resized to size x size
    rgb: if True, will return colored image, otherwise grayscale.
    Returns: one dimensional numpy array of length 30 x 30 x 3 (rgb case)
    """
    if not os.path.isfile(path):
        raise Exception("image file does not exist")
    if size > ORIGINAL_SIZE or size <= 0:
        raise Exception("invalid size for resizing image")
    img = cv2.imread(path)
    if resize:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    if not rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.flatten()

def _list_files(data_dir):
    """return files as list of triple:
    (index, filepath relative to data_dir, label) sorted by index."""
    l = []
    for root, subdirs, files in os.walk(data_dir):
        for f in files:
            if f.endswith(".jpg"):
                label = os.path.basename(root)
                index = int(f.strip(".jpg"))
                l.append((index, os.path.join(label, f), LABELS[label]))
    l.sort()
    return l

def read_data_sets(data_dir):
    """Use this public function to read the dataset"""
    # TODO: add shuffling parameter?
    labels, images = _load_images(data_dir)
    VALIDATION_SIZE = 100

    train_labels = labels[VALIDATION_SIZE:]
    train_images = images[VALIDATION_SIZE:]
    validation_labels = labels[:VALIDATION_SIZE]
    validation_images = images[:VALIDATION_SIZE]

    train = DataSet(train_images, train_labels)
    validation = DataSet(validation_images, validation_labels)
    # Currently don't have a separate test set
    return Datasets(train=train, validation=validation)
