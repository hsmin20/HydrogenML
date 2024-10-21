import sys
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras import layers
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QGridLayout, QLabel, QLineEdit, QCheckBox, QProgressBar, QMessageBox
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
import time
from MachineLearner import MachineLearner, TfCallback
from NuclearCFDLSTM import LSTMWindow

class TransformerMachineLearner(MachineLearner):
   def setWindowSize(self, windowSize):
        self.windowSize = windowSize

   def set(self, batchSize, epoch, learningRate, splitPercentage, earlyStopping, verb, inputShape, headSize, numHeads,
           ffDim, numTransBlocks, mlpUnits, dropout, mlpDropout, callback):
        self.batchSize = batchSize
        self.epoch = epoch
        self.learningRate = learningRate
        self.splitPercentage = splitPercentage
        self.earlyStopping = earlyStopping
        self.verbose = verb
        self.callback = callback
        self.model = self.createModel(inputShape, headSize, numHeads, ffDim, numTransBlocks, mlpUnits, dropout, mlpDropout)
        self.modelLoaded = True

   def transformerEncoder(self, inputs, head_size, num_heads, ff_dim, dropout):
        # Normalization and Attention
        x = layers.LayerNormalization(epsilon=1e-6)(inputs)
        x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
        x = layers.Dropout(dropout)(x)
        res = x + inputs

        # Feed Forward Part
        x = layers.LayerNormalization(epsilon=1e-6)(res)
        x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
        x = layers.Dropout(dropout)(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)

        return x + res

   def createModel(self, inputShape, headSize, numHeads, ffDim, numTransBlocks, mlpUnits, dropout=0, mlpDropout=0):
        adamOpt = tf.keras.optimizers.Adam(learning_rate=self.learningRate)

        inputs = keras.Input(shape=inputShape)
        x = inputs
        for _ in range(numTransBlocks):
            x = self.transformerEncoder(x, headSize, numHeads, ffDim, dropout)

        x = layers.GlobalAveragePooling2D(data_format="channels_last")(x)
        for dim in mlpUnits:
            x = layers.Dense(dim, activation="relu")(x)
            x = layers.Dropout(mlpDropout)(x)
        outputs = layers.Dense(1)(x)

        model = keras.Model(inputs, outputs)

        model.compile(loss='mse', optimizer=adamOpt, metrics=RootMeanSquaredError())

        if self.verbose:
            model.summary()

        return model

class TransformerWindow(LSTMWindow):
    def __init__(self):
        super().__init__()

        self.modelLearner = TransformerMachineLearner()

    def initMLOption(self):
        layout = QGridLayout()

        batchLabel = QLabel('Batch Size')
        batchLabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.editBatch = QLineEdit('32')
        self.editBatch.setFixedWidth(100)
        epochLabel = QLabel('Epoch')
        epochLabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.editEpoch = QLineEdit('100')
        self.editEpoch.setFixedWidth(100)
        lrLabel = QLabel('Learning Rate')
        self.editLR = QLineEdit('0.0002')
        self.editLR.setFixedWidth(100)
        self.cbVerbose = QCheckBox('Verbose')
        self.cbVerbose.setChecked(True)

        splitLabel = QLabel('Split for Validation (0 means no split-data for validation)')
        self.editSplit = QLineEdit('0.2')
        self.editSplit.setFixedWidth(100)
        self.cbSKLearn = QCheckBox('use sklearn split')
        self.cbSKLearn.setChecked(True)
        self.cbEarlyStop = QCheckBox('Use Early Stopping (validation data)')

        widsizeLabel = QLabel('Window Size')
        self.editWidSize = QLineEdit('10')
        self.editWidSize.setFixedWidth(100)
        tmMultiLabel = QLabel('Time Multiplier')
        self.editTmMulti = QLineEdit('7.5')
        self.editTmMulti.setFixedWidth(100)
        distMultiLabel = QLabel('Dist Multiplier')
        self.editDistMulti = QLineEdit('19')
        self.editDistMulti.setFixedWidth(100)

        self.cbMinMax = QCheckBox('Use Min/Max of  All data')
        self.cbMinMax.setChecked(True)
        self.cbTestData = QCheckBox('Use Partially checked  sensors as Test')
        self.cbTestData.setChecked(True)
        self.epochPbar = QProgressBar()

        layout.addWidget(batchLabel, 0, 0, 1, 1)
        layout.addWidget(self.editBatch, 0, 1, 1, 1)
        layout.addWidget(epochLabel, 0, 2, 1, 1)
        layout.addWidget(self.editEpoch, 0, 3, 1, 1)
        layout.addWidget(lrLabel, 0, 4, 1, 1)
        layout.addWidget(self.editLR, 0, 5, 1, 1)
        layout.addWidget(self.cbVerbose, 0, 6, 1, 1)

        layout.addWidget(splitLabel, 1, 0, 1, 2)
        layout.addWidget(self.editSplit, 1, 2, 1, 1)
        layout.addWidget(self.cbSKLearn, 1, 3, 1, 1)
        layout.addWidget(self.cbEarlyStop, 1, 4, 1, 2)

        layout.addWidget(widsizeLabel, 2, 0, 1, 1)
        layout.addWidget(self.editWidSize, 2, 1, 1, 1)
        layout.addWidget(tmMultiLabel, 2, 2, 1, 1)
        layout.addWidget(self.editTmMulti, 2, 3, 1, 1)
        layout.addWidget(distMultiLabel, 2, 4, 1, 1)
        layout.addWidget(self.editDistMulti, 2, 5, 1, 1)

        layout.addWidget(self.cbMinMax, 3, 0, 1, 2)
        layout.addWidget(self.cbTestData, 3, 2, 1, 2)
        layout.addWidget(self.epochPbar, 3, 4, 1, 4)

        return layout

    def initTransformerOption(self):
        layout = QGridLayout()

        headSizeLabel = QLabel('Head Size')
        headSizeLabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.editHeadSize = QLineEdit('512')
        self.editHeadSize.setFixedWidth(100)
        numHeadsLabel = QLabel('Number of Heads')
        numHeadsLabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.editNumHeads = QLineEdit('4')
        self.editNumHeads.setFixedWidth(100)
        ffDimLabel = QLabel('Feed-Forward Dimension')
        ffDimLabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.editFFDim = QLineEdit('4')
        self.editFFDim.setFixedWidth(100)

        numTransBlocksLabel = QLabel('Number of Transformer Blocks')
        numTransBlocksLabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.editNumTransBlocks = QLineEdit('4')
        self.editNumTransBlocks.setFixedWidth(100)
        mlpUnitsLabel = QLabel('MLP Units[')
        mlpUnitsLabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.editMLPUnits = QLineEdit('256')
        self.editMLPUnits.setFixedWidth(100)
        mlpUnitsLabelR = QLabel(']')
        mlpUnitsLabelR.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        dropoutLabel = QLabel('Dropout')
        dropoutLabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.editDropout = QLineEdit('0')
        self.editDropout.setFixedWidth(100)
        mlpDropoutLabel = QLabel('MLP Dropout')
        mlpDropoutLabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.editMLPDropout = QLineEdit('0')
        self.editMLPDropout.setFixedWidth(100)

        layout.addWidget(headSizeLabel, 0, 0, 1, 1)
        layout.addWidget(self.editHeadSize, 0, 1, 1, 1)
        layout.addWidget(numHeadsLabel, 0, 2, 1, 1)
        layout.addWidget(self.editNumHeads, 0, 3, 1, 1)
        layout.addWidget(ffDimLabel, 0, 4, 1, 1)
        layout.addWidget(self.editFFDim, 0, 5, 1, 1)

        layout.addWidget(numTransBlocksLabel, 1, 0, 1, 1)
        layout.addWidget(self.editNumTransBlocks, 1, 1, 1, 1)
        layout.addWidget(mlpUnitsLabel, 1, 2, 1, 1)
        layout.addWidget(self.editMLPUnits, 1, 3, 1, 1)
        layout.addWidget(mlpUnitsLabelR, 1, 4, 1, 1)

        layout.addWidget(dropoutLabel, 2, 0, 1, 1)
        layout.addWidget(self.editDropout, 2, 1, 1, 1)
        layout.addWidget(mlpDropoutLabel, 2, 2, 1, 1)
        layout.addWidget(self.editMLPDropout, 2, 3, 1, 1)

        return layout

    def initUI(self):
        self.setWindowTitle('Machine Learning Curve Fitting/Interpolation')
        self.setWindowIcon(QIcon('web.png'))

        self.initMenu()

        layout = QVBoxLayout()

        sensorLayout = self.initSensor()
        readOptLayout = self.initReadOption()
        fileLayout = self.initCSVFileReader()
        cmdLayout = self.initCommand()
        mlOptLayout = self.initMLOption()
        transOptLayout = self.initTransformerOption()
        tableLayout = self.initGridTable()

        layout.addLayout(fileLayout)
        layout.addLayout(sensorLayout)
        layout.addLayout(readOptLayout)
        layout.addLayout(mlOptLayout)
        layout.addLayout(transOptLayout)
        layout.addLayout(cmdLayout)
        layout.addLayout(tableLayout)

        self.centralWidget().setLayout(layout)

        self.resize(900, 800)
        self.center()
        self.show()

    def _doMachineLearning(self, x_train_data, y_train_data, x_valid_data, y_valid_data, x_test_data, y_test_data):
        epoch = int(self.editEpoch.text())
        batchSize = int(self.editBatch.text())
        learningRate = float(self.editLR.text())
        splitPercentage = float(self.editSplit.text())
        earlyStopping = self.cbEarlyStop.isChecked()
        verb = self.cbVerbose.isChecked()

        inputShape = x_train_data.shape[1:]
        headSize = int(self.editHeadSize.text())
        numHeads = int(self.editNumHeads.text())
        ffDim = int(self.editFFDim.text())
        numTransBlocks = int(self.editNumTransBlocks.text())
        mlpUnits = int(self.editMLPUnits.text())
        dropout = float(self.editDropout.text())
        mlpDropout = float(self.editMLPDropout.text())

        self.epochPbar.setMaximum(epoch)

        if self.cbResume.isChecked() == False or self.modelLearner.modelLoaded == False:
            self.modelLearner.set(batchSize, epoch, learningRate, splitPercentage, earlyStopping, verb, inputShape,
                                  headSize, numHeads, ffDim, numTransBlocks, [mlpUnits], dropout, mlpDropout,
                                  TfCallback(self.epochPbar))

        training_history = self.modelLearner.fitWithValidation(x_train_data, y_train_data, x_valid_data, y_valid_data)

        y_train_pred = self.modelLearner.predict(x_train_data)
        y_valid_pred = self.modelLearner.predict(x_valid_data)
        if len(x_test_data) > 0:
            y_test_pred = self.modelLearner.predict(x_test_data)
        else:
            y_test_pred = []

        self.modelLearner.showResultValid(training_history, y_train_data, y_train_pred,
                                          y_valid_data, y_valid_pred, y_test_data, y_test_pred)

    def _processForMachineLearning(self, xdata):
        window = [[x] for x in xdata]
        return window

    def _processForCheckVal(self, x_data, windowSize):
        x_data_2 = [[x] for x in x_data]
        x_final = [x_data_2]
        x_data_3 = np.array(x_final)

        return x_data_3

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = TransformerWindow()
    sys.exit(app.exec_())