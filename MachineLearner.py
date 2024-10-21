import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflowjs as tfjs
from sklearn.metrics import r2_score

NUM_DATA = 1000

class TfCallback(tf.keras.callbacks.Callback):
    def __init__(self, pbar):
        super().__init__()
        self.pbar = pbar
        self.curStep = 1

    def on_epoch_end(self, epoch, logs=None):
        self.pbar.setValue(self.curStep)
        self.curStep += 1

class MachineLearner:
    def __init__(self):
        self.modelLoaded = False

    def set(self, nnList, batchSize, epoch, learningRate, splitPercentage, earlyStopping, verb, callback):
        self.batchSize = batchSize
        self.epoch = epoch
        self.learningRate = learningRate
        self.splitPercentage = splitPercentage
        self.earlyStopping = earlyStopping
        self.verbose = verb
        self.callback = callback
        self.model = self.createModel(nnList)
        self.modelLoaded = True

    def fit(self, x_data, y_data):
        _callbacks = [self.callback]
        if self.earlyStopping == True:
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=500)
            _callbacks.append(early_stopping)

        if self.splitPercentage > 0:
            training_history = self.model.fit(x_data, y_data, batch_size=self.batchSize, epochs=self.epoch,
                                              validation_split=self.splitPercentage, verbose=self.verbose,
                                              callbacks=[_callbacks])
        else:
            training_history = self.model.fit(x_data, y_data, batch_size=self.batchSize, epochs=self.epoch,
                                              verbose=self.verbose, callbacks=_callbacks)

        return training_history

    def fitWithValidation(self, x_train_data, y_train_data, x_valid_data, y_valid_data):
        _callbacks = [self.callback]
        if self.earlyStopping == True:
            early_stopping = tf.keras.callbacks.EarlyStopping()
            _callbacks.append(early_stopping)

        training_history = self.model.fit(x_train_data, y_train_data, batch_size=self.batchSize, epochs=self.epoch,
                                          verbose=self.verbose, validation_data=(x_valid_data, y_valid_data),
                                          callbacks=_callbacks)

        return training_history

    def predict(self, x_data):
        y_predicted = self.model.predict(x_data)

        return y_predicted

    def saveModel(self, foldername):
        if self.modelLoaded == True:
            self.model.save(foldername, save_format="h5")

    def saveModelJS(self, filename):
        if self.modelLoaded == True:
            tfjs.converters.save_keras_model(self.model, filename)

    def loadModel(self, foldername):
        self.model = keras.models.load_model(foldername)
        self.modelLoaded = True

    def showResultValid(self, training_history, y_train_data, y_train_pred, y_valid_data, y_valid_pred,
                        y_test_data, y_test_pred):
        r2Train = r2_score(y_train_data, y_train_pred)
        r2Valid = r2_score(y_valid_data, y_valid_pred)
        if len(y_test_data) > 0:
            r2Test = r2_score(y_test_data, y_test_pred)
        else:
            r2Test = 0

        fig, axs = plt.subplots(2, 2, figsize=(12, 12))
        title = 'Sensors' + 'Height'
        fig.suptitle(title)

        datasize = len(y_train_data)
        x_display2 = np.zeros((datasize, 1))
        for j in range(datasize):
            x_display2[j][0] = j

        datasize = len(y_valid_data)
        x_display3 = np.zeros((datasize, 1))
        for j in range(datasize):
            x_display3[j][0] = j

        datasize = len(y_test_data)
        x_display = np.zeros((datasize, 1))
        for j in range(datasize):
            x_display[j][0] = j

        axs[0, 0].scatter(x_display2, y_train_data, color="red", s=1)
        axs[0, 0].scatter(x_display2, y_train_pred, color='blue', s=1)
        title = f'Train Data (R2 = {r2Train})'
        axs[0, 0].set_title(title)
        axs[0, 0].grid()

        axs[0, 1].scatter(x_display3, y_valid_data, color="red", s=1)
        axs[0, 1].scatter(x_display3, y_valid_pred, color='blue', s=1)
        title = f'Validation Data (R2 = {r2Valid})'
        axs[0, 1].set_title(title)
        axs[0, 1].grid()

        axs[1, 0].scatter(x_display, y_test_data, color="red", s=1)
        axs[1, 0].plot(x_display, y_test_pred, color='blue')
        title = f'Test Data (R2 = {r2Test})'
        axs[1, 0].set_title(title)
        axs[1, 0].grid()

        lossarray = training_history.history['loss']
        axs[1, 1].plot(lossarray, label='Loss')
        axs[1, 1].set_title('Loss')
        axs[1, 1].grid()

        plt.show()
