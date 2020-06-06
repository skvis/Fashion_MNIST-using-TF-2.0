import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import classification_report

import config
import data_preprocess
import sys


def prediction(model):

    X_test, _ = data_preprocess.data_preprocessing(config.TEST_DF)
    y_true = np.array(config.TEST_DF.iloc[:, 0])

    predicted_classes = model.predict_classes(X_test)
    p = predicted_classes[:10000]
    y = y_true[:10000]
    correct = np.nonzero(p == y)[0]
    incorrect = np.nonzero(p != y)[0]

    print('Correct predicted classes:', correct.shape[0])
    print('Incorrect predicted classes:', incorrect.shape[0])

    target_names = ["Class {} ({}) :".format(i, config.LABELS[i]) for i in range(config.NUM_CLASSES)]
    print(classification_report(y_true, predicted_classes, target_names=target_names))


if __name__ == '__main__':
    model = tf.keras.models.load_model(f"{config.MODEL_PATH}my_model.h5")
    prediction(model)
