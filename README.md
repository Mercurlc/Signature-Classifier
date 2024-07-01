# Signature-Classifier
A module for quick learning and using the signature classification model

![Static Badge](https://img.shields.io/badge/Signature-Classifier)
![GitHub top language](https://img.shields.io/github/languages/top/Mercurlc/Signature-Classifier)
![GitHub](https://img.shields.io/github/license/Mercurlc/Signature-Classifier)
![GitHub Repo stars](https://img.shields.io/github/stars/Mercurlc/Signature-Classifier)

<img src="https://github.com/Mercurlc/Signature-Classifier/assets/110380879/fdbc56af-45a0-4d30-8745-ae38ead19548" alt="icon" width="45%">

## Usage (Linux)

1. Create a virtual environment

```Python
python -m venv venv
```

2. Activate the virtual environment

```Python
source venv/bin/activate
```

3. Install dependencies

```Python
pip install Signature-Classifier
```

4. Run the script to demonstrate the capabilities of the library

Check out the [usage examples](https://github.com/Mercurlc/Signature-Classifier/tree/main/example-usage) to learn how to properly use the library. Ensure that your [dataset](https://github.com/Mercurlc/Signature-Classifier/tree/main/example-usage/dataset) is in the same format and has the same class names. Use the built-in or [online documentation](https://signature-classifier.readthedocs.io/en/latest/).

## Training

Training the final model can be done either after optimizing hyperparameters or directly by adding custom values in the following format:

```Python
custom_params = {
    'resize_shape': (128, 256),
    'nns': 3,
    'orientations': 12,
    'cells_per_block': 3,
    'pixels_per_cell': 16,
}
```

You can also save plots that visualize the heatmap and confusion matrix. Example:
<p align="center">
  <img src="https://github.com/Mercurlc/Signature-Classifier/assets/110380879/0cf7edc1-c7b9-4f6f-8222-90b23b75d6c6" alt="classification_report_heatmap" width="45%">
  <img src="https://github.com/Mercurlc/Signature-Classifier/assets/110380879/c822b600-6cf5-42f6-bb4b-48318afe6980" alt="confusion_matrix_heatmap" width="45%">
</p>

---

After training, all parameters, including custom ones, and the trained model are saved. You can then use only the predict method.

## Prediction

The predict method allows you to select the number of nearest neighbors to return and whether to return the forgery probability (use only if there are such data in the training set). It returns a list of tuples, where each tuple consists of:

```Python
[('003', 1.0, 0.5347718688045583)]
```

1. Class number.
2. Probability of belonging to this class.
3. Probability of forgery from 0 to 1, where 0 is a forgery and 1 is a genuine signature.

## Why these methods?

**HOG** (Histogram of Oriented Gradients) - is a feature extraction method that transforms an image into a set of histograms of gradients, allowing the capture of shape and structure.

- Robustness to Scale and Rotation: Signatures can be written at different angles and sizes, but HOG retains the essential characteristics of the image regardless of these changes.
- Focus on Edge Features: Signatures consist of lines and curves, which are well represented by gradients. HOG effectively captures these edge features, crucial for accurate recognition.

<p align="center">
  <img src="https://github.com/Mercurlc/Signature-Classifier/assets/110380879/32383b65-e4f5-4e0c-8147-59187cebea6c" alt="hog example" width="100%">
</p>

**KNN** (k-Nearest Neighbors) - is a simple and efficient machine learning algorithm for classification tasks. It is based on the idea that objects close to each other in feature space are likely to belong to the same class. Since there is likely a small amount of data and signatures are relatively easy to distinguish, KNN is well-suited for this task.

**Thank you for using Signature Classifier and star this project!**
