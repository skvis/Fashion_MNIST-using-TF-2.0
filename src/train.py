from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger
import numpy as np
from sklearn import model_selection

import model
import config
import data_preprocess


def train_fn():

    X, y = data_preprocess.data_preprocessing(config.TRAIN_DF)
    X_test, y_test = data_preprocess.data_preprocessing(config.TEST_DF)

    X_train, X_val, y_train, y_val = model_selection.train_test_split(X, y,
                                                                      test_size=config.TEST_SIZE,
                                                                      random_state=100,
                                                                      stratify=y)

    cnn_model = model.CNN()

    cnn_model.compile(optimizer=RMSprop(lr=config.LEARNING_RATE),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

    # print(model.summary())

    earlystop = EarlyStopping(patience=10)
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                                patience=2,
                                                verbose=1,
                                                factor=0.5,
                                                min_lr=0.00001)
    csv_logger = CSVLogger(f'{config.MODEL_PATH}training.log', separator=',', append=False)

    callbacks = [earlystop, learning_rate_reduction, csv_logger]

    history = cnn_model.fit(X_train, y_train,
                            validation_data=(X_val, y_val),
                            epochs=config.NUM_EPOCHS,
                            verbose=2,
                            callbacks=callbacks)

    score = cnn_model.evaluate(X_test, y_test, verbose=0)
    print(score)

    cnn_model.save(f"{config.MODEL_PATH}my_model.h5")
    np.save(f'{config.MODEL_PATH}my_history.npy', history.history)


if __name__ == '__main__':
    train_fn()
