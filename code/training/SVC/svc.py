"""SVC implementation from scikit-learn"""

import argparse
import os, sys
from sklearn import datasets, svm, metrics

# add parent dir to be able to retrieve picar.py
parent_dir = os.path.abspath("..")
sys.path.insert(0, parent_dir)

from picar import read_data_sets

def main():
    picar = read_data_sets(FLAGS.data_dir)

    model = svm.SVC(gamma=0.001)
    model.fit(picar.train.images, picar.train.labels)

    predicted = model.predict(picar.validation.images)
    expected = picar.validation.labels

    print("Accuracy:", metrics.accuracy_score(expected, predicted))
    print("Classification report for classifier %s:\n%s\n"
          % (model, metrics.classification_report(expected, predicted)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/Users/naturegirl/code/tensor-racer/data/round1',
                      help='Directory for storing data')
  FLAGS = parser.parse_args()
  main()