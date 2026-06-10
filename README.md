# handwritten-digit-recognition
Semester project for my AI course. Built a CNN on MNIST and deployed it as a desktop app.
# Handwritten Digit Recognition

A university AI semester project. Draw a digit on screen and the model predicts it.

## Setup

```bash
pip install tensorflow pillow
```

Place `digit_model.keras` in the same folder as `digit_recognizer_app.py`, then run:

```bash
python digit_recognizer_app.py
```

## How to get digit_model.keras

Train the model on [Google Colab](https://colab.research.google.com) using the training code, then download the saved file and put it in this folder.

## Files

- `digit_recognizer_app.py` — the desktop drawing app
- `digit_model.keras` — the trained model

## Dataset

MNIST — 70,000 handwritten digit images, loaded automatically through Keras.

## Accuracy

~98.5% on the MNIST test set.

=



