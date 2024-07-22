import sys
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras import layers
from PyQt5.QtWidgets import QApplication, QMainWindow, QDesktopWidget, QAction, QFileDialog, \
    QVBoxLayout, QWidget, QPushButton, QGridLayout, QLabel, QInputDialog, \
    QLineEdit, QComboBox, QMessageBox, QCheckBox, QProgressBar, QHBoxLayout, QTableWidget, QTableWidgetItem, \
    QAbstractItemView, QHeaderView, QDialogButtonBox, QDialog, QGroupBox, QRadioButton, QButtonGroup
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
import time

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

def R_squared(y, y_pred):
    residual = tf.reduce_sum(tf.square(tf.subtract(y, y_pred)))
    total = tf.reduce_sum(tf.square(tf.subtract(y, tf.reduce_mean(y))))
    r2 = tf.subtract(1.0, tf.divide(residual, total))

    return r2

def correctValue(values):
    corrected = (values - 101325) / 1000
    return corrected

def correctValueList(valueList):
    arr = np.array(valueList)
    corrected = (arr - 101325) / 1000
    return corrected

class MachineLearner:
    def __init__(self):
        self.modelLoaded = False

    def set(self, batchSize, epoch, learningRate, splitPercentage, windowSize, earlyStopping, verb, callback,
            inputShape, headSize, numHeads, ffDim, numTransBlocks, mlpUnits, dropout, mlpDropout):
        self.batchSize = batchSize
        self.epoch = epoch
        self.learningRate = learningRate
        self.windowSize = int(windowSize)
        self.splitPercentage = splitPercentage
        self.earlyStopping = earlyStopping
        self.verbose = verb
        self.callback = callback
        self.model = self.createModel(inputShape, headSize, numHeads, ffDim, numTransBlocks, mlpUnits, dropout, mlpDropout)
        self.modelLoaded = True

    def fit(self, x_data, y_data):
        _callbacks = [self.callback]
        if self.earlyStopping == True:
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=500)
            _callbacks.append(early_stopping)

        if self.splitPercentage > 0:
            training_history = self.model.fit(x_data, y_data, batch_size=self.batchSize, epochs=self.epoch,
                                              validation_split=self.splitPercentage, verbose=self.verbose,
                                              callbacks=_callbacks)
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

    def transformerEncoder(self, inputs, head_size, num_heads, ff_dim, dropout=0):
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

        x = layers.GlobalAveragePooling2D(data_format="channels_first")(x)
        for dim in mlpUnits:
            x = layers.Dense(dim, activation="relu")(x)
            x = layers.Dropout(mlpDropout)(x)
        outputs = layers.Dense(1)(x)

        model = keras.Model(inputs, outputs)

        model.compile(loss='mse', optimizer=adamOpt, metrics=RootMeanSquaredError())

        if self.verbose:
            model.summary()

        return model

    def saveModel(self, foldername):
        if self.modelLoaded == True:
            self.model.save(foldername, save_format="h5")

    def loadModel(self, foldername):
        self.model = keras.models.load_model(foldername)
        self.modelLoaded = True

    def showResult(self, y_data, training_history, y_predicted, sensor_name, height):
        r2All = R_squared(y_data, y_predicted)
        r2AllValue = r2All.numpy()

        pred_y = []
        for pv in y_predicted:
            pred_y.append(pv[0])

        fig, axs = plt.subplots(2, 1, figsize=(12, 12))
        title = sensor_name + height
        fig.suptitle(title)

        datasize = len(y_data)
        x_display = np.zeros((datasize, 1))
        for j in range(datasize):
            x_display[j][0] = j

        if len(pred_y) < datasize:
            for _ in range(datasize - len(pred_y)):
                pred_y.insert(0, 0)

        axs[0].scatter(x_display, y_data, color="red", s=1)
        axs[0].plot(x_display, pred_y, color='blue')
        title = f'All Data (R2 = {r2AllValue})'
        axs[0].set_title(title)
        axs[0].grid()

        lossarray = training_history.history['loss']
        axs[1].plot(lossarray, label='Loss')
        axs[1].set_title('Loss')
        axs[1].grid()

        plt.show()

    def showResultValid(self, y_data, training_history, y_predicted, y_train_data, y_train_pred,
                                          y_valid_data, y_valid_pred, sensor_name, height):
        r2All = R_squared(y_data, y_predicted)
        r2Train = R_squared(y_train_data, y_train_pred)
        r2Valid = R_squared(y_valid_data, y_valid_pred)

        r2AllValue = r2All.numpy()
        r2TrainValue = r2Train.numpy()
        r2ValidValue = r2Valid.numpy()

        fig, axs = plt.subplots(2, 2, figsize=(12, 12))
        title = sensor_name + height
        fig.suptitle(title)

        datasize = len(y_data)
        x_display = np.zeros((datasize, 1))
        for j in range(datasize):
            x_display[j][0] = j

        datasize = len(y_train_data)
        x_display2 = np.zeros((datasize, 1))
        for j in range(datasize):
            x_display2[j][0] = j

        datasize = len(y_valid_data)
        x_display3 = np.zeros((datasize, 1))
        for j in range(datasize):
            x_display3[j][0] = j

        axs[0, 0].scatter(x_display, y_data, color="red", s=1)
        axs[0, 0].plot(x_display, y_predicted, color='blue')
        title = f'All Data (R2 = {r2AllValue})'
        axs[0, 0].set_title(title)
        axs[0, 0].grid()

        lossarray = training_history.history['loss']
        axs[0, 1].plot(lossarray, label='Loss')
        axs[0, 1].set_title('Loss')
        axs[0, 1].grid()

        axs[1, 0].scatter(x_display2, y_train_data, color="red", s=1)
        axs[1, 0].scatter(x_display2, y_train_pred, color='blue', s=1)
        title = f'Train Data (R2 = {r2TrainValue})'
        axs[1, 0].set_title(title)
        axs[1, 0].grid()

        axs[1, 1].scatter(x_display3, y_valid_data, color="red", s=1)
        axs[1, 1].scatter(x_display3, y_valid_pred, color='blue', s=1)
        title = f'Validation Data (R2 = {r2ValidValue})'
        axs[1, 1].set_title(title)
        axs[1, 1].grid()

        plt.show()

class MLWindow(QMainWindow):
    N_FEATURE = 6

    def __init__(self):
        super().__init__()

        self.DEFAULT_LAYER_FILE = 'defaultNuCFDLSTM.nn'
        self.NUM_DATA = 1000

        self.distArrayName = ['S6m', 'S10m', 'S20m']
        self.distList = [6.0, 10.0, 20.0]

        self.barrierPosArrayName = ['B4m (Width 10m)', 'B5m (Width 10m)', 'B6m (Width 10m)',
                                    'B4m (Width 5m)', 'B5m (Width 5m)', 'B6m (Width 5m)']
        self.barrierPosList = [4.0, 5.0, 6.0, 4.0, 5.0, 6.0]

        self.initUI()

        self.indexijs = []
        self.time_data = None
        self.southSensors = []
        self.dataLoaded = False
        self.modelLearner = MachineLearner()

    def initMenu(self):
        exitMenu = QAction(QIcon('exit.png'), 'Exit', self)
        exitMenu.setStatusTip('Exit')
        exitMenu.triggered.connect(self.close)

        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)
        fileMenu = menubar.addMenu('&File')
        fileMenu.addSeparator()
        fileMenu.addAction(exitMenu)

        self.statusBar().showMessage('Welcome to Machine Learning Transformer')
        self.central_widget = QWidget()  # define central widget
        self.setCentralWidget(self.central_widget)  # set QMainWindow.centralWidget

    def initCSVFileReader(self):
        layout = QHBoxLayout()

        fileLabel = QLabel('csv file')
        self.editFile = QLineEdit('Please load /DataRepositoy/total_data_horizontal.csv')
        self.editFile.setFixedWidth(700)
        openBtn = QPushButton('...')
        openBtn.clicked.connect(self.showFileDialog)

        layout.addWidget(fileLabel)
        layout.addWidget(self.editFile)
        layout.addWidget(openBtn)

        return layout

    def initSensor(self):
        layout = QGridLayout()

        rows = len(self.barrierPosArrayName)
        cols = len(self.distArrayName)

        self.distArray = []
        for i in range(cols):
            cbDist = QCheckBox(self.distArrayName[i])
            cbDist.setTristate()
            cbDist.stateChanged.connect(self.distClicked)
            self.distArray.append(cbDist)

        self.barrierPosArray = []
        for i in range(rows):
            cbBarrierPos = QCheckBox(self.barrierPosArrayName[i])
            cbBarrierPos.setTristate()
            cbBarrierPos.stateChanged.connect(self.barrierPosClicked)
            self.barrierPosArray.append(cbBarrierPos)

        self.cbArray = []
        for i in range(rows):
            col = []
            for j in range(cols):
                col.append(QCheckBox(''))
            self.cbArray.append(col)

        for i in range(len(self.distArray)):
            cbDist = self.distArray[i]
            layout.addWidget(cbDist, 0, i + 1)

        for i in range(len(self.barrierPosArray)):
            cbBarrierPos = self.barrierPosArray[i]
            layout.addWidget(cbBarrierPos, i + 1, 0)

        for i in range(rows):
            for j in range(cols):
                qcheckbox = self.cbArray[i][j]
                qcheckbox.setTristate()
                layout.addWidget(qcheckbox, i + 1, j + 1)

        return layout

    def initReadOption(self):
        layout = QHBoxLayout()

        loadButton = QPushButton('Load Data')
        loadButton.clicked.connect(self.loadData)
        showButton = QPushButton('Show Graph')
        showButton.clicked.connect(self.showGraphs)

        layout.addWidget(loadButton)
        layout.addWidget(showButton)

        return layout

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
        self.editWidSize = QLineEdit('20')
        self.editWidSize.setFixedWidth(100)
        tmMultiLabel = QLabel('Time Multiplier')
        self.editTmMulti = QLineEdit('7.5')
        self.editTmMulti.setFixedWidth(100)
        distMultiLabel = QLabel('Dist Multiplier')
        self.editDistMulti = QLineEdit('19')
        self.editDistMulti.setFixedWidth(100)

        self.cbMinMax = QCheckBox('Use Min/Max of  All data')
        self.cbMinMax.setChecked(True)
        self.cbValidData = QCheckBox('Use Partially checked  sensors as validation')
        self.cbValidData.stateChanged.connect(self.validDataClicked)
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
        layout.addWidget(self.cbValidData, 3, 2, 1, 2)
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

    def initCommand(self):
        layout = QGridLayout()

        mlWithDataBtn = QPushButton('ML with Data')
        mlWithDataBtn.clicked.connect(self.doMachineLearningWithData)
        self.cbResume = QCheckBox('Resume Learning')
        # self.cbResume.setChecked(True)
        saveModelBtn = QPushButton('Save Model')
        saveModelBtn.clicked.connect(self.saveModel)
        loadModelBtn =  QPushButton('Load Model')
        loadModelBtn.clicked.connect(self.loadModel)
        checkValBtn = QPushButton('Check Trained')
        checkValBtn.clicked.connect(self.checkVal)

        layout.addWidget(mlWithDataBtn, 0, 0, 1, 1)
        layout.addWidget(self.cbResume, 0, 1, 1, 1)
        layout.addWidget(saveModelBtn, 0, 2, 1, 1)
        layout.addWidget(loadModelBtn, 0, 3, 1, 1)
        layout.addWidget(checkValBtn, 0, 4, 1, 1)

        return layout

    def initGridTable(self):
        layout = QGridLayout()

        self.tableGridWidget = QTableWidget()

        self.tableGridWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tableGridWidget.setColumnCount(1)
        item = QTableWidgetItem('')
        self.tableGridWidget.setHorizontalHeaderItem(0, item)

        # barrier distance and hegith
        bHeightLabel = QLabel('Barrier Height')
        bHeightLabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.editBHeight = QLineEdit('2')
        self.editBHeight.setFixedWidth(100)
        bWidthLabel = QLabel('Barrier Width')
        bWidthLabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.editBWidth = QLineEdit('10')
        self.editBWidth.setFixedWidth(100)

        # Buttons
        btnAddDist = QPushButton('Add Distance')
        btnAddDist.setToolTip('Add a Distance')
        btnAddDist.clicked.connect(self.addDistance)
        btnAddHeight = QPushButton('Add BarrierPos')
        btnAddHeight.setToolTip('Add a BarrierPos')
        btnAddHeight.clicked.connect(self.addBarrierPos)
        btnRemoveDist = QPushButton('Remove Distance')
        btnRemoveDist.setToolTip('Remove a Distance')
        btnRemoveDist.clicked.connect(self.removeDistance)
        btnRemoveHeight = QPushButton('Remove BarrierPos')
        btnRemoveHeight.setToolTip('Remove a BarrierPos')
        btnRemoveHeight.clicked.connect(self.removeBarrierPos)
        btnLoadDistHeight = QPushButton('Load')
        btnLoadDistHeight.setToolTip('Load predefined Distance/BarrierPos structure')
        btnLoadDistHeight.clicked.connect(self.loadBarrierPos)
        predictBtn = QPushButton('Predict')
        predictBtn.clicked.connect(self.predict)

        layout.addWidget(self.tableGridWidget, 0, 0, 9, 6)
        layout.addWidget(bHeightLabel, 9, 0)
        layout.addWidget(self.editBHeight, 9, 1)
        layout.addWidget(bWidthLabel, 9, 2)
        layout.addWidget(self.editBWidth, 9, 3)
        layout.addWidget(btnAddDist, 10, 0)
        layout.addWidget(btnAddHeight, 10, 1)
        layout.addWidget(btnRemoveDist, 10, 2)
        layout.addWidget(btnRemoveHeight, 10, 3)
        layout.addWidget(btnLoadDistHeight, 10, 4)
        layout.addWidget(predictBtn, 10, 5)

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

    def addDistance(self):
        sDist, ok = QInputDialog.getText(self, 'Input Distance', 'Distance to add:')
        if ok:
            cc = self.tableGridWidget.columnCount()
            self.tableGridWidget.setColumnCount(cc + 1)
            item = QTableWidgetItem('S' + sDist + 'm')
            self.tableGridWidget.setHorizontalHeaderItem(cc, item)

    def addBarrierPos(self):
        sHeight, ok = QInputDialog.getText(self, 'Input BarrierPos', 'BarrierPos to add:')
        if ok:
            rc = self.tableGridWidget.rowCount()
            self.tableGridWidget.setRowCount(rc + 1)
            item = QTableWidgetItem('B' + sHeight + 'm')
            self.tableGridWidget.setVerticalHeaderItem(rc, item)

    def removeDistance(self):
        col = self.tableGridWidget.currentColumn()
        if col == -1:
            QMessageBox.warning(self, 'Warning', 'Select any cell')
            return
        if col == 0:
            QMessageBox.warning(self, 'Warning', 'First column cannot be removed')
            return

        self.tableGridWidget.removeColumn(col)

    def removeBarrierPos(self):
        row = self.tableGridWidget.currentRow()
        if row == -1:
            QMessageBox.warning(self, 'Warning', 'Select any cell')
            return
        self.tableGridWidget.removeRow(row)

    def loadBarrierPos(self):
        fname = QFileDialog.getOpenFileName(self, 'Open distance/barrier-pos data file', '/srv/MLData',
                                            filter="CSV file (*.csv);;All files (*)")
        with open(fname[0], "r") as f:
            self.tableGridWidget.setRowCount(0)
            self.tableGridWidget.setColumnCount(0)

            lines = f.readlines()

            distAr = lines[0].split(',')
            for sDist in distAr:
                sDist = sDist.rstrip()
                cc = self.tableGridWidget.columnCount()
                self.tableGridWidget.setColumnCount(cc + 1)
                item = QTableWidgetItem('S' + sDist + 'm')
                self.tableGridWidget.setHorizontalHeaderItem(cc, item)

            barrierPosAr = lines[1].split(',')
            for sBarrierPos in barrierPosAr:
                sBarrierPos = sBarrierPos.rstrip()
                rc = self.tableGridWidget.rowCount()
                self.tableGridWidget.setRowCount(rc + 1)
                item = QTableWidgetItem('B' + sBarrierPos + 'm')
                self.tableGridWidget.setVerticalHeaderItem(rc, item)

    def distClicked(self, state):
        senderName = self.sender().text()
        col = self.distArrayName.index(senderName)
        rows = len(self.cbArray)
        for i in range(rows):
            qcheckbox = self.cbArray[i][col]
            if state == Qt.Unchecked:
                qcheckbox.setCheckState(Qt.Unchecked)
            elif state == Qt.Checked:
                qcheckbox.setCheckState(Qt.Checked)
            else:
                qcheckbox.setCheckState(Qt.PartiallyChecked)

    def barrierPosClicked(self, state):
        senderName = self.sender().text()
        row = self.barrierPosArrayName.index(senderName)
        cols = len(self.cbArray[0])
        for i in range(cols):
            qcheckbox = self.cbArray[row][i]
            if state == Qt.Unchecked:
                qcheckbox.setCheckState(Qt.Unchecked)
            elif state == Qt.Checked:
                qcheckbox.setCheckState(Qt.Checked)
            else:
                qcheckbox.setCheckState(Qt.PartiallyChecked)

    def validDataClicked(self, state):
        if state == Qt.Checked:
            self.editSplit.setEnabled(False)
            self.cbSKLearn.setEnabled(False)
        else:
            self.editSplit.setEnabled(True)
            self.cbSKLearn.setEnabled(True)

    def showFileDialog(self):
        fname = QFileDialog.getOpenFileName(self, 'Open data file', '/srv/MLData', filter="CSV file (*.csv);;All files (*)")
        if fname[0]:
            self.editFile.setText(fname[0])
            self.df = pd.read_csv(fname[0], dtype=float)

    def loadData(self):
        filename = self.editFile.text()
        if filename == '' or filename.startswith('Please'):
            QMessageBox.about(self, 'Warining', 'No CSV Data')
            return

        self.df = pd.read_csv(filename, dtype=float)

        rows = len(self.barrierPosArrayName)
        cols = len(self.distArrayName)

        listSelected = []
        for i in range(rows):
            for j in range(cols):
                qcheckbox = self.cbArray[i][j]
                if qcheckbox.checkState() != Qt.Unchecked:
                    listSelected.append((i, j))

        if len(listSelected) == 0:
            QMessageBox.information(self, 'Warning', 'Select sensor(s) first..')
            return

        try:
            self.readCSV(listSelected)

        except ValueError:
            QMessageBox.information(self, 'Error', 'There is some error...')
            return

        self.dataLoaded = True
        QMessageBox.information(self, 'Done', 'Data is Loaded')

    def readCSV(self, listSelected):
        self.indexijs.clear()
        self.time_data = None
        self.southSensors.clear()

        self.time_data = self.df.values[0:1000, 4:5].flatten()

        for index_ij in listSelected:
            sensorName, data = self.getPressureData(index_ij)
            self.indexijs.append(index_ij)
            self.southSensors.append(data)

    def showGraphs(self):
        if self.dataLoaded == False:
            QMessageBox.information(self, 'Warning', 'Load Data First')
            return

        numSensors = len(self.southSensors)

        plt.figure()
        for i in range(numSensors):
            t_data = self.time_data
            s_data_raw = self.southSensors[i]

            s_data = correctValue(s_data_raw)

            max_val = max(s_data)
            smax_val = format(max_val, '.2f')
            index_at_max = max(range(len(s_data)), key=s_data.__getitem__)
            impulse = self.getImpulse(s_data)
            simpulse = format(impulse, '.2f')
            index_at_zero = index_at_max
            for index in range(index_at_max, len(s_data)-1):
                val = s_data[index]
                val_n = s_data[index+1]
                if val >=0 and val_n <=0:
                    index_at_zero = index
                    break
            time_at_zero = t_data[index_at_zero]
            stime_at_zero = format(time_at_zero, '.6f')

            index_ij = self.indexijs[i]
            i = index_ij[0]
            j = index_ij[1]
            sensorName = self.distArrayName[j] + self.barrierPosArrayName[i] + ' (max:' + smax_val + ')'

            plt.scatter(t_data, s_data, label=sensorName, s=1)

        plt.title('Pressure Graph')
        plt.xlabel('time (ms)')
        plt.ylabel('pressure (kPa)')
        plt.legend(loc='upper right', markerscale=4.)
        plt.grid()

        plt.show()

    def getPressureData(self, index_ij):
        i = index_ij[0]
        j = index_ij[1]

        col_no = (i * len(self.distArray) * self.N_FEATURE) + (j * self.N_FEATURE) + 5
        sensorName = self.distArrayName[j] + self.barrierPosArrayName[i]

        data = self.df.values[:, col_no:col_no + 1].flatten()

        return sensorName, data

    def _prepareForMachineLearning(self, windowSize):
        numSensors = len(self.southSensors)

        df_normalized = self._normalize()

        # separate data to train & validation
        trainIndex = []

        for i in range(numSensors):
            indexij = self.indexijs[i]
            trainIndex.append(indexij)

        if len(trainIndex) == 0:
            QMessageBox.warning(self, 'warning', 'No Training Data!')
            return

        x_train_data = []
        y_train_data = []

        for i in range(len(trainIndex)):
            indexij = trainIndex[i]

            row_num = indexij[0]
            col_num = indexij[1]
            startNum = (row_num * len(self.distArray) * self.N_FEATURE) + (col_num * self.N_FEATURE)

            df_first_temp = df_normalized.iloc[:, startNum:startNum + self.N_FEATURE]
            df_first = df_first_temp.to_numpy()
            first_list = df_first.tolist()
            for i in range(self.NUM_DATA - windowSize):
                # row = [a for a in df_first[i:i + windowSize]]
                window = first_list[i:(i + windowSize)]
                window = [[x] for x in window]

                x_train_data.append(window)
                label = df_first[i + windowSize, 5:6]
                y_train_data.append(label)

        return np.array(x_train_data), np.array(y_train_data)

    def _prepareForMachineLearningSKLearn(self, windowSize, splitPecentage):
        numSensors = len(self.southSensors)

        df_normalized = self._normalize()

        allIndex = []

        for i in range(numSensors):
            indexij = self.indexijs[i]
            allIndex.append(indexij)

        if len(allIndex) == 0:
            QMessageBox.warning(self, 'warning', 'No Training Data!')
            return

        x_all_data = []
        y_all_data = []

        for i in range(len(allIndex)):
            indexij = allIndex[i]

            row_num = indexij[0]
            col_num = indexij[1]
            startNum = (row_num * len(self.distArray) * self.N_FEATURE) + (col_num * self.N_FEATURE)

            df_first_temp = df_normalized.iloc[:, startNum:startNum + self.N_FEATURE]
            df_first = df_first_temp.to_numpy()
            first_list = df_first.tolist()
            for i in range(self.NUM_DATA - windowSize):
                # row = [a for a in df_first[i:i + windowSize]]
                window = first_list[i:(i + windowSize)]
                window = [[x] for x in window]

                x_all_data.append(window)
                label = df_first[i + windowSize, 5:6]
                y_all_data.append(label)

        x_train_data, x_valid_data, y_train_data, y_valid_data = train_test_split(x_all_data, y_all_data,
                                                                                  test_size=splitPecentage,
                                                                                  random_state=42)

        return np.array(x_all_data), np.array(y_all_data), np.array(x_train_data), np.array(y_train_data), \
            np.array(x_valid_data), np.array(y_valid_data)

    def _prepareForMachineLearningManually(self, windowSize):
        numSensors = len(self.southSensors)

        df_normalized = self._normalize()

        # separate data to train & validation
        trainIndex = []
        validIndex = []

        allIndex = []

        x_all_data = []
        y_all_data = []

        for i in range(numSensors):
            indexij = self.indexijs[i]
            allIndex.append(indexij)

        for i in range(len(allIndex)):
            indexij = allIndex[i]

            row_num = indexij[0]
            col_num = indexij[1]
            startNum = (row_num * len(self.distArray) * self.N_FEATURE) + (col_num * self.N_FEATURE)

            df_first_temp = df_normalized.iloc[:, startNum:startNum + self.N_FEATURE]
            df_first = df_first_temp.to_numpy()
            first_list = df_first.tolist()
            for i in range(self.NUM_DATA - windowSize):
                window = first_list[i:(i + windowSize)]
                window = [[x] for x in window]

                x_all_data.append(window)
                label = df_first[i + windowSize, 5:6]
                y_all_data.append(label)

        # Train & Validation
        for i in range(numSensors):
            indexij = self.indexijs[i]
            row_num = indexij[0]
            col_num = indexij[1]

            if self.cbArray[row_num][col_num].checkState() == Qt.PartiallyChecked:
                validIndex.append(indexij)
            elif self.cbArray[row_num][col_num].checkState() == Qt.Checked:
                trainIndex.append(indexij)

        if len(trainIndex) == 0:
            QMessageBox.warning(self, 'warning', 'No Training Data!')
            return

        x_train_data = []
        y_train_data = []

        x_valid_data = []
        y_valid_data = []

        for i in range(len(trainIndex)):
            indexij = trainIndex[i]

            row_num = indexij[0]
            col_num = indexij[1]
            startNum = (row_num * len(self.distArray) * self.N_FEATURE) + (col_num * self.N_FEATURE)

            df_first_temp = df_normalized.iloc[:, startNum:startNum + self.N_FEATURE]
            df_first = df_first_temp.to_numpy()
            first_list = df_first.tolist()
            for i in range(self.NUM_DATA - windowSize):
                # row = [a for a in df_first[i:i + windowSize]]
                window = first_list[i:(i + windowSize)]
                window = [[x] for x in window]

                x_train_data.append(window)
                label = df_first[i + windowSize, 5:6]
                y_train_data.append(label)

        for i in range(len(validIndex)):
            indexij = validIndex[i]

            row_num = indexij[0]
            col_num = indexij[1]
            startNum = (row_num * len(self.distArray) * self.N_FEATURE) + (col_num * self.N_FEATURE)

            df_first_temp = df_normalized.iloc[:, startNum:startNum + self.N_FEATURE]
            df_first = df_first_temp.to_numpy()
            for i in range(self.NUM_DATA - windowSize):
                # row = [a for a in df_first[i:i + windowSize]]
                window = first_list[i:(i + windowSize)]
                window = [[x] for x in window]

                x_valid_data.append(window)
                label = df_first[i + windowSize, 5:6]
                y_valid_data.append(label)

        return np.array(x_all_data), np.array(y_all_data), np.array(x_train_data), np.array(y_train_data), \
            np.array(x_valid_data), np.array(y_valid_data)

    def doMachineLearningWithData(self):
        if not self.indexijs or not self.southSensors:
            QMessageBox.about(self, 'Warining', 'Load Data First')
            return

        batchSize = int(self.editBatch.text())
        epoch = int(self.editEpoch.text())
        if epoch < 1:
            QMessageBox.warning(self, 'warning', 'Epoch shall be greater than 0')
            return

        splitPercentage = float(self.editSplit.text())
        if splitPercentage < 0 or splitPercentage > 1.0:
            QMessageBox.warning(self, 'warning', 'splitPercentage shall be between 0 and 1')
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)

        windowSize = int(self.editWidSize.text())

        learningRate = float(self.editLR.text())
        verbose = self.cbVerbose.isChecked()

        splitPercentage = float(self.editSplit.text())
        useSKLearn = self.cbSKLearn.isChecked()
        earlyStopping = self.cbEarlyStop.isChecked()
        useValidation = self.cbValidData.isChecked()

        if useValidation:
            x_all_data, y_all_data, x_train_data, y_train_data, x_valid_data, y_valid_data = \
                self._prepareForMachineLearningManually(windowSize)
        else:
            if splitPercentage > 0.0 and useSKLearn:
                x_all_data, y_all_data, x_train_data, y_train_data, x_valid_data, y_valid_data = \
                    self._prepareForMachineLearningSKLearn(windowSize, splitPercentage)
            else:
                x_train_data, y_train_data = self._prepareForMachineLearning(windowSize)

        if useValidation or (splitPercentage > 0.0 and useSKLearn):
            self.doMachineLearningWithValidation(x_all_data, y_all_data, x_train_data, y_train_data, x_valid_data,
                                                 y_valid_data, batchSize, epoch, learningRate, splitPercentage,
                                                 windowSize, earlyStopping, verbose)
        else:
            self.doMachineLearning(x_train_data, y_train_data, batchSize, epoch, learningRate, splitPercentage,
                                   windowSize, earlyStopping, verbose)

        QApplication.restoreOverrideCursor()

    def saveModel(self):
        suggestion = '/srv/MLData/tfModel.h5'
        filename = QFileDialog.getSaveFileName(self, 'Save Model File', suggestion, filter="h5 file (*.h5)")
        if filename[0] != '':
            self.modelLearner.saveModel(filename[0])

        QMessageBox.information(self, 'Saved', 'Model is saved.')

    def loadModel(self):
        fname = QFileDialog.getOpenFileName(self, 'Open h5 model file', '/srv/MLData',
                                            filter="h5 file (*.h5);;All files (*)")
        if fname[0]:
            self.modelLearner.loadModel(fname[0])

        QMessageBox.information(self, 'Loaded', 'Model is loaded.')

    def checkVal(self):
        if not self.indexijs or not self.southSensors:
            QMessageBox.about(self, 'Warining', 'Load Data First')
            return
        if self.modelLearner.modelLoaded == False:
            QMessageBox.about(self, 'Warning', 'Model is not created/loaded')
            return

        start_time = time.time()

        QApplication.setOverrideCursor(Qt.WaitCursor)

        df_normalized = self._normalize()

        windowSize = int(self.editWidSize.text())
        y_data = []
        y_pred = []

        totalSize = len(self.indexijs) * (self.NUM_DATA - windowSize)
        self.epochPbar.setMaximum(totalSize)
        for i in range(len(self.indexijs)):
            indexij = self.indexijs[i]

            row_num = indexij[0]
            col_num = indexij[1]
            startNum = (row_num * len(self.distArray) * self.N_FEATURE) + (col_num * self.N_FEATURE)

            df_first_temp = df_normalized.iloc[:, startNum:startNum + self.N_FEATURE]
            df_first = df_first_temp.to_numpy()
            first_list = df_first.tolist()

            x_data = first_list[0:windowSize]
            time_diff = x_data[1][4] - x_data[0][4]

            for j in range(self.NUM_DATA - windowSize):
                x_data_2 = [[x] for x in x_data]
                x_final = [x_data_2]
                x_data_3 = np.array(x_final)
                y_predicted = self.modelLearner.predict(x_data_3)

                p_predicted = y_predicted[0][0]
                y_pred.append(p_predicted)

                y_data.append(df_first_temp.iloc[j + windowSize - 1, 5])

                x_data.pop(0)
                x_data.append(x_data[-1].copy())
                x = x_data[windowSize-1][4]
                x_data[windowSize-1][4] = x + time_diff
                x_data[windowSize-1][5] = p_predicted

                self.epochPbar.setValue(i * (self.NUM_DATA - windowSize) + j)

        QApplication.restoreOverrideCursor()

        print("--- %s seconds ---" % (time.time() - start_time))

        datasize = len(y_data)
        x_display = np.zeros((datasize, 1))
        for j in range(datasize):
            x_display[j][0] = j

        r2All = R_squared(y_data, y_pred)
        r2AllValue = r2All.numpy()
        title = f'Transfomer Validation (R2 = {r2AllValue})'

        plt.figure()
        plt.scatter(x_display, y_data, label='original data', color="red", s=1)
        plt.scatter(x_display, y_pred, label='predicted', color="blue", s=1)
        plt.title(title)
        plt.xlabel('time (ms)')
        plt.ylabel('pressure (kPa)')
        plt.legend(loc='upper right')
        plt.grid()

        plt.show()

    def predict(self):
        if self.modelLearner.modelLoaded == False:
            QMessageBox.about(self, 'Warning', 'Model is not created/loaded')
            return

        distCount = self.tableGridWidget.columnCount()
        barrierPosCount = self.tableGridWidget.rowCount()
        if distCount < 1 or barrierPosCount < 1:
            QMessageBox.warning(self, 'Warning', 'You need to add distance or barrier pos to predict')
            return

        bposmax, bposmin, sdistmax, sdistmin, bheightmax, bheightmin, bwidthmax, bwidthmin, timemax, timemin, \
        pressuremax, pressuremin = self._getMaxMin()

        start_time = time.time()

        QApplication.setOverrideCursor(Qt.WaitCursor)

        sDistBarrierPosArray = []
        distBarrierPosArray = []

        for i in range(distCount):
            dist = self.tableGridWidget.horizontalHeaderItem(i).text()
            distOnly = dist[1:len(dist)-1]

            for j in range(barrierPosCount):
                barrierPos = self.tableGridWidget.verticalHeaderItem(j).text()
                barrierPosOnly = barrierPos[1:len(barrierPos)-1]

                distf = float(distOnly)
                barrierPosf = float(barrierPosOnly)

                sDistBarrierPosArray.append(dist+barrierPos)
                distBarrierPosArray.append((distf, barrierPosf))

        windowSize = int(self.editWidSize.text())
        bHeight = float(self.editBHeight.text())
        bWidth = float(self.editBWidth.text())
        y_array = []

        totalSize = len(distBarrierPosArray) * (self.NUM_DATA - windowSize)
        self.epochPbar.setMaximum(totalSize)

        i = 0
        for distBarrierPos in distBarrierPosArray:
            x_data = self._prepareOneSensorForPredict(distBarrierPos, windowSize, bHeight, bWidth)
            time_diff = x_data[1][4] - x_data[0][4]

            y_pred_arr = []
            for wi in range(windowSize):
                y_pred_arr.append(x_data[wi][5])
            for j in range(self.NUM_DATA - windowSize):
                # x_input = np.array(x_data)
                # x_input = x_input.reshape((1, windowSize, self.N_FEATURE))
                x_data_2 = [[x] for x in x_data]
                x_final = [x_data_2]
                x_data_3 = np.array(x_final)
                y_predicted = self.modelLearner.predict(x_data_3)

                p_predicted = y_predicted[0][0]
                y_pred_arr.append(p_predicted)

                x_data.pop(0)
                x_data.append(x_data[-1].copy())
                x = x_data[windowSize - 1][4]
                x_data[windowSize - 1][4] = x + time_diff
                x_data[windowSize - 1][5] = p_predicted

                self.epochPbar.setValue(i * (self.NUM_DATA - windowSize) + j)

            i += 1
            self.unnormalize(y_pred_arr, pressuremax, pressuremin)
            y_array.append(y_pred_arr)

        QApplication.restoreOverrideCursor()

        print("--- %s seconds ---" % (time.time() - start_time))

        resultArray = self.showPredictionGraphs(sDistBarrierPosArray, distBarrierPosArray, y_array)

        reply = QMessageBox.question(self, 'Message', 'Do you want to save overpressure and impulse to a file?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            suggestion = '/srv/MLData/opAndImpulse.csv'
            filename = QFileDialog.getSaveFileName(self, 'Save File', suggestion, "CSV Files (*.csv)")

            if filename[0] != '':
                file = open(filename[0], 'w')

                column1 = 'distance,barrierPos,indexAtMax,overpressure,indexAtZero,impulse\n'
                file.write(column1)

                for col in resultArray:
                    file.write(col+'\n')

    def unnormalize(self, data, max, min):
        for i in range(len(data)):
            data[i] = data[i] * (max - min) + min

    def _prepareOneSensorForPredict(self, distBarrierPos, windowSize, bHeight, bWidth):
        bposmax, bposmin, sdistmax, sdistmin, bheightmax, bheightmin, bwidthmax, bwidthmin, timemax, timemin, \
            pressuremax, pressuremin = self._getMaxMin()

        distance = distBarrierPos[0]
        barrierPos = distBarrierPos[1]

        dist_multiplier = float(self.editDistMulti.text())

        distance_n = dist_multiplier * (distance - sdistmin) / (sdistmax - sdistmin)
        barrierPos_n = (barrierPos - bposmin) / (bposmax - bposmin)
        height_n = (bHeight - bheightmin) / (bheightmax - bheightmin)
        width_n = (bWidth - bwidthmin) / (bwidthmax - bwidthmin)

        df_normalized = self._normalize()

        df_first_temp = df_normalized.iloc[:, 0:self.N_FEATURE]
        df_first = df_first_temp.to_numpy()

        x_data = [a for a in df_first[0:windowSize]]
        for i in range(len(x_data)):
            x_data[i][0] = barrierPos_n
            x_data[i][1] = distance_n
            x_data[i][2] = height_n
            x_data[i][3] = width_n

        return x_data

    def showPredictionGraphs(self, sDistBarrierPosArray, distBarrierPosArray, y_array):
        # numSensors = len(y_array)
        resultArray = []

        plt.figure()
        for i in range(len(y_array)):
            t_data = self.time_data
            s_data_raw = y_array[i]

            s_data = correctValueList(s_data_raw)

            distBarrierPos = distBarrierPosArray[i]
            lab = sDistBarrierPosArray[i]

            distance = distBarrierPos[0]
            barrierPos = distBarrierPos[1]

            # index_at_max = max(range(len(s_data)), key=s_data.__getitem__)
            overpressure = max(s_data)
            # impulse, index_at_zero = self.getImpulseAndIndexZero(s_data)

            dispLabel = lab # + '/op=' + format(overpressure[0], '.2f') + '/impulse=' + format(impulse, '.2f')

            resultArray.append(str(distance) + ',' + str(barrierPos)) # + ',' + ',' +
                               # format(overpressure[0], '.6f')) + ',' + str(index_at_zero) + ',' + format(impulse, '.6f'))

            plt.scatter(t_data, s_data, label=dispLabel, s=1)

        plt.title('Pressure Graph')
        plt.xlabel('time (s)')
        plt.ylabel('pressure (kPa)')
        plt.legend(loc='upper right', markerscale=4.)
        plt.grid()

        plt.show()

        return resultArray

    def getImpulse(self, data):
        sumImpulse = 0
        impulseArray = []
        for i in range(len(data)):
            cur_p = data[i]
            sumImpulse += cur_p
            impulseArray.append(sumImpulse)

        impulse = max(impulseArray)

        return impulse

    def getImpulseAndIndexZero(self, data):
        index_at_max = max(range(len(data)), key=data.__getitem__)

        sumImpulse = 0
        impulseArray = []
        initP = data[0][0]
        for i in range(len(data)):
            cur_p = data[i][0]
            if cur_p > 0 and cur_p <= initP :
                cur_p = 0

            sumImpulse += cur_p * 0.000002
            impulseArray.append(sumImpulse)

        # index_at_zero = max(range(len(impulseArray)), key=impulseArray.__getitem__)
        impulse = max(impulseArray)
        index_at_zero = impulseArray.index(impulse)

        return impulse, index_at_zero

    def checkDataGraph(self, sensorName, time_data, rawdata, filetered_data, data_label, iterNum):
        impulse_original = self.getImpulse(rawdata)
        impulse_filtered = self.getImpulse(filetered_data)

        plt.figure()

        rawLabel = 'Raw-Normalized (impulse=' + format(impulse_original, '.4f') + ')'
        plt.scatter(time_data, rawdata, label=rawLabel, color="red", s=1)
        filterLabel = data_label + ' (iter=' + str(iterNum) + ', impulse=' + format(impulse_filtered, '.4f') + ')'
        plt.scatter(time_data, filetered_data, label=filterLabel, color="blue", s=1)
        plt.title(sensorName)
        plt.xlabel('time (ms)')
        plt.ylabel('pressure (kPa)')
        plt.legend(loc='upper right')
        plt.grid()

        plt.show()

    def checkDataGraph2(self, sensorName, time_data, rawdata, filetered_data, data_label, index_at_max, overpressure, index_at_zero):
        impulse_original = self.getImpulse(rawdata)
        impulse_filtered = self.getImpulse(filetered_data)

        plt.figure()

        rawLabel = 'Raw-Normalized (impulse=' + format(impulse_original, '.4f') + ')'
        plt.scatter(time_data, rawdata, label=rawLabel, color="red", s=1)
        filterLabel = data_label + ', impulse=' + format(impulse_filtered, '.4f') + ',indexMax=' + str(index_at_max) \
                      + ',Overpressure=' + str(overpressure) + ',indexZero=' + str(index_at_zero)
        plt.scatter(time_data, filetered_data, label=filterLabel, color="blue", s=1)
        plt.title(sensorName)
        plt.xlabel('time (ms)')
        plt.ylabel('pressure (kPa)')
        plt.legend(loc='upper right')
        plt.grid()

        plt.show()

    def _getMaxMin(self):
        bposmax = -999999
        bposmin = 999999
        sdistmax = -999999
        sdistmin = 999999
        bheightmax = -999999
        bheightmin = 999999
        bwidthmax = -999999
        bwidthmin = 999999
        timemax = -999999
        timemin = 999999
        pressuremax = -999999
        pressuremin = 999999

        if self.cbMinMax.isChecked() == True:
            numSensors = len(self.distArray) * len(self.barrierPosArray)
        else:
            numSensors = len(self.southSensors)

        for i in range(numSensors):
            if self.cbMinMax.isChecked() == True:
                firstcol = i * self.N_FEATURE
            else:
                indexij = self.indexijs[i]
                row_num = indexij[0]
                col_num = indexij[1]
                firstcol = (row_num * len(self.distArray) * self.N_FEATURE) + (col_num * self.N_FEATURE)

            df_temp = self.df.iloc[:, firstcol:firstcol + self.N_FEATURE]

            df_minmax = df_temp.agg(['max', 'min'])
            if df_minmax.iloc[0][0] > bposmax:
                bposmax = df_minmax.iloc[0][0]
            if df_minmax.iloc[1][0] < bposmin:
                bposmin = df_minmax.iloc[1][0]
            if df_minmax.iloc[0][1] > sdistmax:
                sdistmax = df_minmax.iloc[0][1]
            if df_minmax.iloc[1][1] < sdistmin:
                sdistmin = df_minmax.iloc[1][1]
            if df_minmax.iloc[0][2] > bheightmax:
                bheightmax = df_minmax.iloc[0][2]
            if df_minmax.iloc[1][2] < bheightmin:
                bheightmin = df_minmax.iloc[1][2]
            if df_minmax.iloc[0][3] > bwidthmax:
                bwidthmax = df_minmax.iloc[0][3]
            if df_minmax.iloc[1][3] < bwidthmin:
                bwidthmin = df_minmax.iloc[1][3]
            if df_minmax.iloc[0][4] > timemax:
                timemax = df_minmax.iloc[0][4]
            if df_minmax.iloc[1][4] < timemin:
                timemin = df_minmax.iloc[1][4]
            if df_minmax.iloc[0][5] > pressuremax:
                pressuremax = df_minmax.iloc[0][5]
            if df_minmax.iloc[1][5] < pressuremin:
                pressuremin = df_minmax.iloc[1][5]

        return bposmax, bposmin, sdistmax, sdistmin, bheightmax, bheightmin, bwidthmax, bwidthmin, timemax, timemin, \
            pressuremax, pressuremin

    def _normalize(self):
        bposmax, bposmin, sdistmax, sdistmin, bheightmax, bheightmin, bwidthmax, bwidthmin, timemax, timemin, \
            pressuremax, pressuremin = self._getMaxMin()

        time_multiplier = float(self.editTmMulti.text())
        dist_multiplier = float(self.editDistMulti.text())

        df_normalized = self.df.copy()

        if self.cbMinMax.isChecked() == True:
            numSensors = len(self.distArray)*len(self.barrierPosArray)
        else:
            numSensors = len(self.southSensors)

        for i in range(numSensors):
            if self.cbMinMax.isChecked() == True:
                firstcol = i * self.N_FEATURE
            else:
                indexij = self.indexijs[i]
                row_num = indexij[0]
                col_num = indexij[1]
                firstcol = (row_num * len(self.distArray) * self.N_FEATURE) + (col_num * self.N_FEATURE)

            noData = -1
            df_temp = df_normalized.iloc[:, firstcol:firstcol + self.N_FEATURE]
            for col in df_temp.columns:
                noData += 1
                if noData % self.N_FEATURE == 0:
                    df_temp[col] = (df_temp[col] - bposmin) / (bposmax - bposmin)
                if noData % self.N_FEATURE == 1:
                    df_temp[col] = dist_multiplier * (df_temp[col] - sdistmin) / (sdistmax - sdistmin)
                if noData % self.N_FEATURE == 2:
                    df_temp[col] = (df_temp[col] - bheightmin) / (bheightmax - bheightmin)
                if noData % self.N_FEATURE == 3:
                    df_temp[col] = (df_temp[col] - bwidthmin) / (bwidthmax - bwidthmin)
                if noData % self.N_FEATURE == 4:
                    df_temp[col] = time_multiplier * (df_temp[col] - timemin) / (timemax - timemin)
                if noData % self.N_FEATURE == 5:
                    df_temp[col] = (df_temp[col] - pressuremin) / (pressuremax - pressuremin)

            df_normalized.update(df_temp)

        return df_normalized

    def doMachineLearning(self, x_data, y_data, batchSize, epoch, learningRate, splitPercentage, windowSize,
                          earlyStopping, verb):
        self.epochPbar.setMaximum(epoch)

        inputShape = x_data.shape[1:]
        headSize = int(self.editHeadSize.text())
        numHeads = int(self.editNumHeads.text())
        ffDim = int(self.editFFDim.text())
        numTransBlocks = int(self.editNumTransBlocks.text())
        mlpUnits = int(self.editMLPUnits.text())
        dropout = float(self.editDropout.text())
        mlpDropout = float(self.editMLPDropout.text())
        if self.cbResume.isChecked() == False or self.modelLearner.modelLoaded == False:
            self.modelLearner.set(batchSize, epoch, learningRate, splitPercentage, windowSize,
                                  earlyStopping, verb, TfCallback(self.epochPbar),
                                  inputShape, headSize, numHeads, ffDim, numTransBlocks, [mlpUnits], dropout, mlpDropout)

        training_history = self.modelLearner.fit(x_data, y_data)

        y_predicted = self.modelLearner.predict(x_data)
        self.modelLearner.showResult(y_data, training_history, y_predicted, 'Sensors', 'Height')

    def doMachineLearningWithValidation(self, x_all_data, y_all_data, x_train_data, y_train_data, x_valid_data,
                                        y_valid_data, batchSize, epoch, learningRate, splitPercentage, windowSize,
                                        earlyStopping, verb):
        self.epochPbar.setMaximum(epoch)

        inputShape = x_train_data.shape[1:]
        headSize = int(self.editHeadSize.text())
        numHeads = int(self.editNumHeads.text())
        ffDim = int(self.editFFDim.text())
        numTransBlocks = int(self.editNumTransBlocks.text())
        mlpUnits = int(self.editMLPUnits.text())
        dropout = float(self.editDropout.text())
        mlpDropout = float(self.editMLPDropout.text())
        if self.cbResume.isChecked() == False or self.modelLearner.modelLoaded == False:
            self.modelLearner.set(batchSize, epoch, learningRate, splitPercentage, windowSize,
                                  earlyStopping, verb, TfCallback(self.epochPbar),
                                  inputShape, headSize, numHeads, ffDim, numTransBlocks, [mlpUnits], dropout, mlpDropout)

        training_history = self.modelLearner.fitWithValidation(x_train_data, y_train_data, x_valid_data, y_valid_data)

        y_predicted = self.modelLearner.predict(x_all_data)
        y_train_pred = self.modelLearner.predict(x_train_data)
        y_valid_pred = self.modelLearner.predict(x_valid_data)

        self.modelLearner.showResultValid(y_all_data, training_history, y_predicted, y_train_data, y_train_pred,
                                          y_valid_data, y_valid_pred, 'Sensors', 'Height')

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

class TfCallback(tf.keras.callbacks.Callback):
    def __init__(self, pbar):
        super().__init__()
        self.pbar = pbar
        self.curStep = 1

    def on_epoch_end(self, epoch, logs=None):
        self.pbar.setValue(self.curStep)
        self.curStep += 1

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MLWindow()
    sys.exit(app.exec_())