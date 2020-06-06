import os
import pandas as pd

DATA_PATH = '../input/'
MODEL_PATH = '../models/'

TRAIN_FILE = os.path.join(DATA_PATH, 'fashion_mnist_train.csv')
TEST_FILE = os.path.join(DATA_PATH, 'fashion_mnist_test.csv')

TRAIN_DF = pd.read_csv(TRAIN_FILE)
TEST_DF = pd.read_csv(TEST_FILE)

LABELS = {0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
          5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle Boot"}

TEST_SIZE = 0.2

# parameter
IMG_WIDTH = 28
IMG_HEIGHT = 28
IMG_CHANNELS = 1

NUM_CLASSES = 10

# Model
NUM_EPOCHS = 3
BATCH_SIZE = 64
LEARNING_RATE = 1e-2
