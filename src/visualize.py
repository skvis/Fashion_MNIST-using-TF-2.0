import config
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot_class_distribution(data):
    f, ax = plt.subplots(1, 1, figsize=(12, 4))
    g = sns.countplot(data.iloc[:, 0], order=data.iloc[:, 0].value_counts().index)
    g.set(xlabel='label', ylabel='count', title='Number of labels for each class')
    for p, label in zip(g.patches, data.iloc[:, 0].value_counts().index):
        g.annotate(config.LABELS[label], (p.get_x(), p.get_height()+0.1))
    plt.show()


def sample_images_data(data):
    # an empty list to collect some samples
    sample_images = []
    sample_labels = []

    # iterate over the keys of the labels
    for k in config.LABELS.keys():
        # get four samples for each category
        samples = data[data.iloc[:, 0] == k].head(4)
        # append the samples to the sample list
        for j, s in enumerate(samples.values):
            # first column contain labels, hence index should start from 1
            img = np.array(samples.iloc[j, 1:]).reshape(28, 28)
            sample_images.append(img)
            sample_labels.append(samples.iloc[j, 0])

    print('Total number of sample images to plot: ', len(sample_images))
    return sample_images, sample_labels


def plot_sample_images(data_sample_images, data_sample_labels, cmap='Blues'):
    f, ax = plt.subplots(5, 8, figsize=(16, 10))
    for i, img in enumerate(data_sample_images):
        ax[i//8, i % 8].imshow(img, cmap=cmap)
        ax[i//8, i % 8].axis('off')
        ax[i//8, i % 8].set_title(config.LABELS[data_sample_labels[i]])
    plt.show()


def plot_graphs(history, string):
    plt.plot(history[string])
    plt.plot(history['val_'+string])
    plt.xlabel('Epochs')
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()


if __name__ == '__main__':
    plot_class_distribution(config.TRAIN_DF)
    plot_class_distribution(config.TEST_DF)

    train_sample_images, train_sample_labels = sample_images_data(config.TRAIN_DF)
    test_sample_images, test_sample_labels = sample_images_data(config.TEST_DF)

    plot_sample_images(train_sample_images, train_sample_labels, 'Greens')
    plot_sample_images(test_sample_images, test_sample_labels)

    history = np.load(f'{config.MODEL_PATH}my_history.npy', allow_pickle=True).item()
    plot_graphs(history, 'accuracy')
    plot_graphs(history, 'loss')
