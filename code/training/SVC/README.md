training with scikit-learn SVC implementation.

Based on tutorial: http://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html

## Training Results

For training the model, I used the training and validation set as given by the `picar.py` data reader. The validation set contains the last 100 samples, and does not shuffle the data. (The validation set includes 33 'left' labels, 45 'right' labels and 22 'straight' labels)

I compared accuracy metrics on the validation set when I vary the size of the image inputs. Specifically I trained with `SVC(gamma=0.0001)` and tested resizing images to 30x30, 60x60 and using the original size of 100x100. All images are in RGB colors.

| image size         | train accuracy | test accuracy |
|--------------------|----------------|---------------|
| 30x30              | 0.8357         | 0.5799        |
| 60x60              | 0.9068         | 0.7299        |
| 100x100 (original) | 0.9534         | 0.79          |
