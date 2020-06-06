import config
import pandas as pd
from tensorflow import keras

# Explore dataset


def dataset_shape(train_data, test_data):
    print(f"Fashion Mnist train - rows: {train_data.shape[0]} columns: {train_data.shape[1]}")  # columns name is not present
    print(f"Fashion Mnist test - rows: {test_data.shape[0]} columns: {test_data.shape[1]}")


def get_class_distribution(data):
    # get the count for each label
    label_counts = data.iloc[:, 0].value_counts()
    # get total number of samples
    total_samples = len(data)
    # count the number items in each class
    for i in range(len(label_counts)):
        label = config.LABELS[label_counts.index[i]]
        count = label_counts.values[i]
        percent = (count / total_samples) * 100
        print(f"{label:<20s}: {count} or {percent:.2f}%")


def data_preprocessing(raw):
    num_images = raw.shape[0]
    x_as_array = raw.values[:, 1:]
    x_shapped_array = x_as_array.reshape(num_images, config.IMG_WIDTH, config.IMG_HEIGHT, config.IMG_CHANNELS)
    out_x = x_shapped_array / 255
    out_y = keras.utils.to_categorical(raw.iloc[:, 0], config.NUM_CLASSES)
    return out_x, out_y


if __name__ == '__main__':

    dataset_shape(config.TRAIN_DF, config.TEST_DF)

    get_class_distribution(config.TRAIN_DF)
    get_class_distribution(config.TEST_DF)

    data_preprocessing(config.TRAIN_DF)
