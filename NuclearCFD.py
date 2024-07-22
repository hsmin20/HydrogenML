import sys
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
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
                                              callbacks=[self.callback])
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

    def createModel(self, nnList):
        adamOpt = tf.keras.optimizers.Adam(learning_rate=self.learningRate)
        model = tf.keras.Sequential()

        firstLayer = True
        for nn in nnList:
            noOfNeuron = nn[0]
            activationFunc = nn[1]
            if firstLayer:
                model.add(tf.keras.layers.Dense(units=noOfNeuron, activation=activationFunc, input_shape=[noOfNeuron]))
                firstLayer = False
            else:
                model.add(tf.keras.layers.Dense(units=noOfNeuron, activation=activationFunc))

        model.compile(loss='mse', optimizer=adamOpt)

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

        fig, axs = plt.subplots(2, 1, figsize=(12, 12))
        title = sensor_name + height
        fig.suptitle(title)

        datasize = len(y_data)
        x_display = np.zeros((datasize, 1))
        for j in range(datasize):
            x_display[j][0] = j

        axs[0].scatter(x_display, y_data, color="red", s=1)
        axs[0].plot(x_display, y_predicted, color='blue')
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

class LayerDlg(QDialog):
    def __init__(self, unit='128', af='relu'):
        super().__init__()
        self.initUI(unit, af)

    def initUI(self, unit, af):
        self.setWindowTitle('Machine Learning Curve Fitting/Interpolation')

        label1 = QLabel('Units', self)
        self.tbUnits = QLineEdit(unit, self)
        self.tbUnits.resize(100, 40)

        label2 = QLabel('Activation f', self)
        self.cbActivation = QComboBox(self)
        self.cbActivation.addItem(af)
        self.cbActivation.addItem('swish')
        self.cbActivation.addItem('relu')
        self.cbActivation.addItem('selu')
        self.cbActivation.addItem('sigmoid')
        self.cbActivation.addItem('softmax')

        if af == 'linear':
            self.cbActivation.setEnabled(False)

        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        layout = QGridLayout()
        layout.addWidget(label1, 0, 0)
        layout.addWidget(self.tbUnits, 0, 1)

        layout.addWidget(label2, 1, 0)
        layout.addWidget(self.cbActivation, 1, 1)

        layout.addWidget(self.buttonBox, 2, 1)

        self.setLayout(layout)

class MLWindow(QMainWindow):
    N_FEATURE = 5
    COL_WIDTH_PER_ONE_SENSOR = 6
    NUM_DATA = 1000

    def __init__(self):
        super().__init__()

        self.DEFAULT_LAYER_FILE = 'defaultNuCFD.nn'

        self.distArrayName = ['S6m', 'S10m', 'S20m']
        self.distList = [6.0, 10.0, 20.0]

        self.barrierPosArrayName = ['B4m (H:2m, W:10m)', 'B5m (H:2m, W:10m)', 'B6m (H:2m, W:10m)',
                                    'B4m (H:2m, W:5m)', 'B5m (H:3m, W:5m)', 'B6m (H:4m, W:5m)']
        self.barrierPosList = [4.0, 5.0, 6.0, 4.0, 5.0, 6.0]

        self.initUI()

        self.indexijs = []
        self.time_data = None
        self.southSensors = []
        self.dataLoaded = False
        self.modelLearner = MachineLearner()

    def initMenu(self):
        # Menu
        openNN = QAction(QIcon('open.png'), 'Open NN', self)
        openNN.setStatusTip('Open Neural Network Structure from a File')
        openNN.triggered.connect(self.showNNFileDialog)

        saveNN = QAction(QIcon('save.png'), 'Save NN', self)
        saveNN.setStatusTip('Save Neural Network Structure in a File')
        saveNN.triggered.connect(self.saveNNFileDialog)

        exitMenu = QAction(QIcon('exit.png'), 'Exit', self)
        exitMenu.setStatusTip('Exit')
        exitMenu.triggered.connect(self.close)

        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(openNN)
        fileMenu.addAction(saveNN)
        fileMenu.addSeparator()
        fileMenu.addAction(exitMenu)

        self.statusBar().showMessage('Welcome to Machine Learning')
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

    def initNNTable(self):
        layout = QGridLayout()

        # NN Table
        self.tableNNWidget = QTableWidget()
        self.tableNNWidget.setColumnCount(2)
        self.tableNNWidget.setHorizontalHeaderLabels(['Units', 'Activation'])

        # read default layers
        self.updateNNList(self.DEFAULT_LAYER_FILE)

        self.tableNNWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tableNNWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tableNNWidget.setSelectionBehavior(QAbstractItemView.SelectRows)

        # Button of NN
        btnAdd = QPushButton('Add')
        btnAdd.setToolTip('Add a Hidden Layer')
        btnAdd.clicked.connect(self.addLayer)
        btnEdit = QPushButton('Edit')
        btnEdit.setToolTip('Edit a Hidden Layer')
        btnEdit.clicked.connect(self.editLayer)
        btnRemove = QPushButton('Remove')
        btnRemove.setToolTip('Remove a Hidden Layer')
        btnRemove.clicked.connect(self.removeLayer)
        btnLoad = QPushButton('Load')
        btnLoad.setToolTip('Load a NN File')
        btnLoad.clicked.connect(self.showNNFileDialog)
        btnSave = QPushButton('Save')
        btnSave.setToolTip('Save a NN File')
        btnSave.clicked.connect(self.saveNNFileDialog)
        btnMakeDefault = QPushButton('Make default')
        btnMakeDefault.setToolTip('Make this as a default NN layer')
        btnMakeDefault.clicked.connect(self.makeDefaultNN)

        layout.addWidget(self.tableNNWidget, 0, 0, 9, 6)
        layout.addWidget(btnAdd, 9, 0)
        layout.addWidget(btnEdit, 9, 1)
        layout.addWidget(btnRemove, 9, 2)
        layout.addWidget(btnLoad, 9, 3)
        layout.addWidget(btnSave, 9, 4)
        layout.addWidget(btnMakeDefault, 9, 5)

        return layout

    def initMLOption(self):
        layout = QGridLayout()

        batchLabel = QLabel('Batch Size')
        batchLabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.editBatch = QLineEdit('32')
        self.editBatch.setFixedWidth(100)
        epochLabel = QLabel('Epoch')
        epochLabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.editEpoch = QLineEdit('200')
        self.editEpoch.setFixedWidth(100)
        lrLabel = QLabel('Learning Rate')
        self.editLR = QLineEdit('0.0003')
        self.editLR.setFixedWidth(100)
        self.cbVerbose = QCheckBox('Verbose')
        self.cbVerbose.setChecked(True)

        splitLabel = QLabel('Split for Validation (0 means no split-data for validation)')
        self.editSplit = QLineEdit('0.2')
        self.editSplit.setFixedWidth(100)
        self.cbSKLearn = QCheckBox('use sklearn split')
        self.cbSKLearn.setChecked(True)
        self.cbEarlyStop = QCheckBox('Use Early Stopping (validation data)')

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

        layout.addWidget(self.cbMinMax, 2, 0, 1, 2)
        layout.addWidget(self.cbValidData, 2, 2, 1, 2)
        layout.addWidget(self.epochPbar, 2, 4, 1, 4)

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
        nnLayout = self.initNNTable()
        mlOptLayout = self.initMLOption()
        tableLayout = self.initGridTable()

        layout.addLayout(fileLayout)
        layout.addLayout(sensorLayout)
        layout.addLayout(readOptLayout)
        layout.addLayout(mlOptLayout)
        layout.addLayout(nnLayout)
        layout.addLayout(cmdLayout)
        layout.addLayout(tableLayout)

        self.centralWidget().setLayout(layout)

        self.resize(900, 800)
        self.center()
        self.show()

    def showNNFileDialog(self):
        fname = QFileDialog.getOpenFileName(self, 'Open NN file', './', filter="NN file (*.nn);;All files (*)")
        if fname[0] != '':
            self.updateNNList(fname[0])

    def updateNNList(self, filename):
        self.tableNNWidget.setRowCount(0)
        with open(filename, "r") as f:
            lines = f.readlines()
            count = 0
            for line in lines:
                strAr = line.split(',')
                self.tableNNWidget.insertRow(count)
                self.tableNNWidget.setItem(count, 0, QTableWidgetItem(strAr[0]))
                self.tableNNWidget.setItem(count, 1, QTableWidgetItem(strAr[1].rstrip()))
                count += 1

    def saveNNFileDialog(self):
        fname = QFileDialog.getSaveFileName(self, 'Save NN file', './', filter="NN file (*.nn)")
        if fname[0] != '':
            self.saveNNFile(fname[0])

    def makeDefaultNN(self):
        filename = self.defaultNuCFD
        self.saveNNFile(filename)
        QMessageBox.information(self, 'Saved', 'Neural Network Default Layers are set')

    def saveNNFile(self, filename):
        with open(filename, "w") as f:
            count = self.tableNNWidget.rowCount()
            for row in range(count):
                unit = self.tableNNWidget.item(row, 0).text()
                af = self.tableNNWidget.item(row, 1).text()
                f.write(unit + "," + af + "\n")

    def getNNLayer(self):
        nnList = []
        count = self.tableNNWidget.rowCount()
        for row in range(count):
            unit = int(self.tableNNWidget.item(row, 0).text())
            af = self.tableNNWidget.item(row, 1).text()
            nnList.append((unit, af))

        return nnList

    def addLayer(self):
        dlg = LayerDlg()
        rc = dlg.exec()
        if rc == 1: # ok
            unit = dlg.tbUnits.text()
            af = dlg.cbActivation.currentText()
            size = self.tableNNWidget.rowCount()
            self.tableNNWidget.insertRow(size-1)
            self.tableNNWidget.setItem(size-1, 0, QTableWidgetItem(unit))
            self.tableNNWidget.setItem(size-1, 1, QTableWidgetItem(af))

    def editLayer(self):
        row = self.tableNNWidget.currentRow()
        if row == -1 or row == (self.tableNNWidget.rowCount() - 1):
            return

        unit = self.tableNNWidget.item(row, 0).text()
        af = self.tableNNWidget.item(row, 1).text()
        dlg = LayerDlg(unit, af)
        rc = dlg.exec()
        if rc == 1: # ok
            unit = dlg.tbUnits.text()
            af = dlg.cbActivation.currentText()
            self.tableNNWidget.setItem(row, 0, QTableWidgetItem(unit))
            self.tableNNWidget.setItem(row, 1, QTableWidgetItem(af))

    def removeLayer(self):
        row = self.tableNNWidget.currentRow()
        if row > 0 and row < (self.tableNNWidget.rowCount() - 1):
            self.tableNNWidget.removeRow(row)

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

        col_no = (i * len(self.distArray) * self.COL_WIDTH_PER_ONE_SENSOR) + (j * self.COL_WIDTH_PER_ONE_SENSOR) + 5
        sensorName = self.distArrayName[j] + self.barrierPosArrayName[i]

        data = self.df.values[:, col_no:col_no + 1].flatten()

        return sensorName, data

    def _prepareForMachineLearning(self):
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

        x_train_data = np.zeros(shape=(self.NUM_DATA * len(trainIndex), self.N_FEATURE))
        y_train_data = np.zeros(shape=(self.NUM_DATA * len(trainIndex), 1))

        for i in range(len(trainIndex)):
            indexij = trainIndex[i]

            row_num = indexij[0]
            col_num = indexij[1]
            startNum = (row_num * len(self.distArray) * self.COL_WIDTH_PER_ONE_SENSOR) + (col_num * self.COL_WIDTH_PER_ONE_SENSOR)

            df_first_temp = df_normalized.iloc[:, startNum:startNum + self.COL_WIDTH_PER_ONE_SENSOR]

            for j in range(self.NUM_DATA):
                x_data_1 = df_first_temp.iloc[j:j+1, 0:5]
                x_data_2 = x_data_1.to_numpy()
                y_data_1 = df_first_temp.iloc[j:j+1, 5:6]
                y_data_2 = y_data_1.to_numpy()

                x_train_data[i*self.NUM_DATA + j] = x_data_2
                y_train_data[i*self.NUM_DATA + j] = y_data_2

        return x_train_data, y_train_data

    def _prepareForMachineLearningSKLearn(self, splitPecentage):
        numSensors = len(self.southSensors)

        df_normalized = self._normalize()

        # separate data to train & validation
        allIndex = []

        for i in range(numSensors):
            indexij = self.indexijs[i]
            allIndex.append(indexij)

        if len(allIndex) == 0:
            QMessageBox.warning(self, 'warning', 'No Training Data!')
            return

        x_all_data = np.zeros(shape=(self.NUM_DATA * len(allIndex), self.N_FEATURE))
        y_all_data = np.zeros(shape=(self.NUM_DATA * len(allIndex), 1))

        for i in range(len(allIndex)):
            indexij = allIndex[i]

            row_num = indexij[0]
            col_num = indexij[1]
            startNum = (row_num * len(self.distArray) * self.COL_WIDTH_PER_ONE_SENSOR) + (col_num * self.COL_WIDTH_PER_ONE_SENSOR)

            df_first_temp = df_normalized.iloc[:, startNum:startNum + self.COL_WIDTH_PER_ONE_SENSOR]

            for j in range(self.NUM_DATA):
                x_data_1 = df_first_temp.iloc[j:j+1, 0:5]
                x_data_2 = x_data_1.to_numpy()
                y_data_1 = df_first_temp.iloc[j:j+1, 5:6]
                y_data_2 = y_data_1.to_numpy()

                x_all_data[i*self.NUM_DATA + j] = x_data_2
                y_all_data[i*self.NUM_DATA + j] = y_data_2

        x_train_data, x_test_data, y_train_data, y_test_data = train_test_split(x_all_data, y_all_data,
                                                                                test_size = splitPecentage,
                                                                                random_state = 42)

        return x_all_data, y_all_data, x_train_data, y_train_data, x_test_data, y_test_data

    def _prepareForMachineLearningManually(self):
        numSensors = len(self.southSensors)

        df_normalized = self._normalize()

        # separate data to train & validation
        trainIndex = []
        validIndex = []

        allIndex = []

        for i in range(numSensors):
            indexij = self.indexijs[i]
            allIndex.append(indexij)

        x_all_data = np.zeros(shape=(self.NUM_DATA * len(allIndex), self.N_FEATURE))
        y_all_data = np.zeros(shape=(self.NUM_DATA * len(allIndex), 1))

        for i in range(len(allIndex)):
            indexij = allIndex[i]

            row_num = indexij[0]
            col_num = indexij[1]
            startNum = (row_num * len(self.distArray) * self.COL_WIDTH_PER_ONE_SENSOR) + (
                        col_num * self.COL_WIDTH_PER_ONE_SENSOR)

            df_first_temp = df_normalized.iloc[:, startNum:startNum + self.COL_WIDTH_PER_ONE_SENSOR]

            for j in range(self.NUM_DATA):
                x_data_1 = df_first_temp.iloc[j:j + 1, 0:5]
                x_data_2 = x_data_1.to_numpy()
                y_data_1 = df_first_temp.iloc[j:j + 1, 5:6]
                y_data_2 = y_data_1.to_numpy()

                x_all_data[i * self.NUM_DATA + j] = x_data_2
                y_all_data[i * self.NUM_DATA + j] = y_data_2

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

        x_train_data = np.zeros(shape=(self.NUM_DATA * len(trainIndex), self.N_FEATURE))
        y_train_data = np.zeros(shape=(self.NUM_DATA * len(trainIndex), 1))

        x_valid_data = np.zeros(shape=(self.NUM_DATA * len(validIndex), self.N_FEATURE))
        y_valid_data = np.zeros(shape=(self.NUM_DATA * len(validIndex), 1))

        for i in range(len(trainIndex)):
            indexij = trainIndex[i]

            row_num = indexij[0]
            col_num = indexij[1]
            startNum = (row_num * len(self.distArray) * self.COL_WIDTH_PER_ONE_SENSOR) + (col_num * self.COL_WIDTH_PER_ONE_SENSOR)

            df_first_temp = df_normalized.iloc[:, startNum:startNum + self.COL_WIDTH_PER_ONE_SENSOR]

            for j in range(self.NUM_DATA):
                x_data_1 = df_first_temp.iloc[j:j+1, 0:5]
                x_data_2 = x_data_1.to_numpy()
                y_data_1 = df_first_temp.iloc[j:j+1, 5:6]
                y_data_2 = y_data_1.to_numpy()

                x_train_data[i*self.NUM_DATA + j] = x_data_2
                y_train_data[i*self.NUM_DATA + j] = y_data_2

        for i in range(len(validIndex)):
            indexij = validIndex[i]

            row_num = indexij[0]
            col_num = indexij[1]
            startNum = (row_num * len(self.distArray) * self.COL_WIDTH_PER_ONE_SENSOR) + (col_num * self.COL_WIDTH_PER_ONE_SENSOR)

            df_first_temp = df_normalized.iloc[:, startNum:startNum + self.COL_WIDTH_PER_ONE_SENSOR]

            for j in range(self.NUM_DATA):
                x_data_1 = df_first_temp.iloc[j:j + 1, 0:5]
                x_data_2 = x_data_1.to_numpy()
                y_data_1 = df_first_temp.iloc[j:j + 1, 5:6]
                y_data_2 = y_data_1.to_numpy()

                x_valid_data[i*self.NUM_DATA + j] = x_data_2
                y_valid_data[i*self.NUM_DATA + j] = y_data_2

        return x_all_data, y_all_data, x_train_data, y_train_data, x_valid_data, y_valid_data

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
                firstcol = i * self.COL_WIDTH_PER_ONE_SENSOR
            else:
                indexij = self.indexijs[i]
                row_num = indexij[0]
                col_num = indexij[1]
                firstcol = (row_num * len(self.distArray) * self.COL_WIDTH_PER_ONE_SENSOR) + (col_num * self.COL_WIDTH_PER_ONE_SENSOR)

            df_temp = self.df.iloc[:, firstcol:firstcol + self.COL_WIDTH_PER_ONE_SENSOR]

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

        df_normalized = self.df.copy()

        if self.cbMinMax.isChecked() == True:
            numSensors = len(self.distArray)*len(self.barrierPosArray)
        else:
            numSensors = len(self.southSensors)

        for i in range(numSensors):
            if self.cbMinMax.isChecked() == True:
                firstcol = i * self.COL_WIDTH_PER_ONE_SENSOR
            else:
                indexij = self.indexijs[i]
                row_num = indexij[0]
                col_num = indexij[1]
                firstcol = (row_num * len(self.distArray) * self.COL_WIDTH_PER_ONE_SENSOR) + (col_num * self.COL_WIDTH_PER_ONE_SENSOR)

            noData = -1
            df_temp = df_normalized.iloc[:, firstcol:firstcol + self.COL_WIDTH_PER_ONE_SENSOR]
            for col in df_temp.columns:
                noData += 1
                if noData % self.COL_WIDTH_PER_ONE_SENSOR == 0:
                    df_temp[col] = (df_temp[col] - bposmin) / (bposmax - bposmin)
                if noData % self.COL_WIDTH_PER_ONE_SENSOR == 1:
                    df_temp[col] = (df_temp[col] - sdistmin) / (sdistmax - sdistmin)
                if noData % self.COL_WIDTH_PER_ONE_SENSOR == 2:
                    df_temp[col] = (df_temp[col] - bheightmin) / (bheightmax - bheightmin)
                if noData % self.COL_WIDTH_PER_ONE_SENSOR == 3:
                    df_temp[col] = (df_temp[col] - bwidthmin) / (bwidthmax - bwidthmin)
                if noData % self.COL_WIDTH_PER_ONE_SENSOR == 4:
                    df_temp[col] = (df_temp[col] - timemin) / (timemax - timemin)
                if noData % self.COL_WIDTH_PER_ONE_SENSOR == 5:
                    df_temp[col] = (df_temp[col] - pressuremin) / (pressuremax - pressuremin)

            df_normalized.update(df_temp)

        return df_normalized

    def doMachineLearningWithData(self):
        if not self.indexijs or not self.southSensors:
            QMessageBox.about(self, 'Warining', 'Load Data First')
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)

        batchSize = int(self.editBatch.text())
        epoch = int(self.editEpoch.text())
        if epoch < 1:
            QMessageBox.warning(self, 'warning', 'Epoch shall be greater than 0')
            return

        learningRate = float(self.editLR.text())
        verbose = self.cbVerbose.isChecked()

        splitPercentage = float(self.editSplit.text())
        useSKLearn = self.cbSKLearn.isChecked()
        earlyStopping = self.cbEarlyStop.isChecked()
        useValidation = self.cbValidData.isChecked()

        if useValidation:
            x_all_data, y_all_data, x_train_data, y_train_data, x_valid_data, y_valid_data = \
                self._prepareForMachineLearningManually()
        else:
            if splitPercentage > 0.0 and useSKLearn:
                    x_all_data, y_all_data, x_train_data, y_train_data, x_valid_data, y_valid_data = \
                        self._prepareForMachineLearningSKLearn(splitPercentage)
            else:
                x_train_data, y_train_data = self._prepareForMachineLearning()

        if useValidation or (splitPercentage > 0.0 and useSKLearn):
            self.doMachineLearningWithValidation(x_all_data, y_all_data, x_train_data, y_train_data, x_valid_data, y_valid_data, batchSize,
                                                 epoch, learningRate, splitPercentage, earlyStopping, verbose)
        else:
            self.doMachineLearning(x_train_data, y_train_data, batchSize, epoch, learningRate, splitPercentage,
                                   earlyStopping, verbose)

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

        x_data, y_data = self._prepareForMachineLearning()
        y_predicted = self.modelLearner.predict(x_data)

        QApplication.restoreOverrideCursor()

        print("--- %s seconds ---" % (time.time() - start_time))

        datasize = len(y_data)
        x_display = np.zeros((datasize, 1))
        for j in range(datasize):
            x_display[j][0] = j

        r2All = R_squared(y_data, y_predicted)
        r2AllValue = r2All.numpy()
        title = f'Machine Learning Validation (R2 = {r2AllValue})'

        plt.figure()
        plt.scatter(x_display, y_data, label='original data', color="red", s=1)
        plt.scatter(x_display, y_predicted, label='predicted', color="blue", s=1)
        plt.title(title)
        plt.xlabel('time (ms)')
        plt.ylabel('pressure (kPa)')
        plt.legend(loc='upper right')
        plt.grid()

        plt.show()

    def predict(self):
        if self.time_data is None:
            QMessageBox.warning(self, 'Warning', 'For timedata, load at least one data')
            return

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
            distOnly = dist[1:len(dist) - 1]

            for j in range(barrierPosCount):
                barrierPos = self.tableGridWidget.verticalHeaderItem(j).text()
                barrierPosOnly = barrierPos[1:len(barrierPos) - 1]

                distf = float(distOnly)
                barrierPosf = float(barrierPosOnly)

                sDistBarrierPosArray.append(dist + barrierPos)
                distBarrierPosArray.append((distf, barrierPosf))

        bpos = float(self.editBHeight.text())
        bwidth = float(self.editBWidth.text())
        y_array = []

        for distBarrierPos in distBarrierPosArray:
            x_data = self._prepareOneSensorForPredict(distBarrierPos, bpos, bwidth)
            y_predicted = self.modelLearner.predict(x_data)
            self.unnormalize(y_predicted, pressuremax, pressuremin)
            y_array.append(y_predicted)

        QApplication.restoreOverrideCursor()

        print("--- %s seconds ---" % (time.time() - start_time))

        resultArray = self.showPredictionGraphs(sDistBarrierPosArray, distBarrierPosArray, y_array)

        reply = QMessageBox.question(self, 'Message', 'Do you want to save pressureto a file?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            suggestion = '/srv/MLData/Prediction.csv'
            filename = QFileDialog.getSaveFileName(self, 'Save File', suggestion, "CSV Files (*.csv)")

            if filename[0] != '':
                file = open(filename[0], 'w')

                column1 = 'id, time,'
                for i in range(len(sDistBarrierPosArray)):
                    distBarrierPos = sDistBarrierPosArray[i]
                    # dist = distBarrierPos[0]
                    # bpos = distBarrierPos[1]
                    column1 = column1 + distBarrierPos

                    if i != (len(sDistBarrierPosArray) -1):
                        column1 = column1 + ','
                    else:
                        column1 = column1 + '\n'

                file.write(column1)

                s_data_raw = y_array[0]
                data_length = len(s_data_raw)

                t_data = self.time_data
                for j in range(data_length):
                    line = str(j + 1) + ',' + str(t_data[j])
                    for i in range(len(y_array)):
                        s_data_raw = y_array[i][j]
                        s_data = correctValue(s_data_raw[0])

                        line = line + ',' + str(s_data)

                    line = line + '\n'
                    file.write(line)

    def unnormalize(self, data, max, min):
        for i in range(len(data)):
            data[i] = data[i] * (max - min) + min

    def _prepareOneSensorForPredict(self, distBarrierPos, height, width):
        bposmax, bposmin, sdistmax, sdistmin, bheightmax, bheightmin, bwidthmax, bwidthmin, timemax, timemin, \
            pressuremax, pressuremin = self._getMaxMin()

        distance = distBarrierPos[0]
        barrierPos = distBarrierPos[1]

        distance_n = (distance - sdistmin) / (sdistmax - sdistmin)
        barrierPos_n = (barrierPos - bposmin) / (bposmax - bposmin)

        height_n = (height - bheightmin) / (bheightmax - bheightmin)
        width_n = (width - bwidthmin) / (bwidthmax - bwidthmin)

        x_data = np.zeros(shape=(self.NUM_DATA, self.N_FEATURE))

        for i in range(self.NUM_DATA):
            x_data[i][0] = barrierPos_n
            x_data[i][1] = distance_n
            x_data[i][2] = height_n
            x_data[i][3] = width_n
            x_data[i][4] = (self.time_data[i] - timemin) / (timemax - timemin)

        return x_data

    def showPredictionGraphs(self, sDistHeightArray, distHeightArray, y_array):
        # numSensors = len(y_array)
        resultArray = []

        plt.figure()
        for i in range(len(y_array)):
            t_data = self.time_data
            s_data_raw = y_array[i]

            s_data = correctValueList(s_data_raw)

            distHeight = distHeightArray[i]
            lab = sDistHeightArray[i]

            distance = distHeight[0]
            height = distHeight[1]

            index_at_max = max(range(len(s_data)), key=s_data.__getitem__)
            overpressure = max(s_data)
            impulse, index_at_zero = self.getImpulseAndIndexZero(s_data)

            dispLabel = lab # + '/op=' + format(overpressure[0], '.2f') + '/impulse=' + format(impulse, '.2f')

            resultArray.append(str(distance) + ',' + str(height) + ',' + str(index_at_max) + ',' +
                               format(overpressure[0], '.6f') + ',' + str(index_at_zero) + ',' + format(impulse, '.6f'))

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

    def doMachineLearning(self, x_data, y_data, batchSize, epoch, learningRate, splitPercentage, earlyStopping, verb):
        self.epochPbar.setMaximum(epoch)

        if self.cbResume.isChecked() == False or self.modelLearner.modelLoaded == False:
            nnList = self.getNNLayer()
            self.modelLearner.set(nnList, batchSize, epoch, learningRate, splitPercentage, earlyStopping, verb,
                                  TfCallback(self.epochPbar))

        training_history = self.modelLearner.fit(x_data, y_data)

        y_predicted = self.modelLearner.predict(x_data)

        r2 = R_squared(y_data, y_predicted)
        print('r2 = ', r2)
        self.modelLearner.showResult(y_data, training_history, y_predicted, 'Sensors', 'Height')

    def doMachineLearningWithValidation(self, x_all_data, y_all_data, x_train_data, y_train_data, x_valid_data,
                                        y_valid_data, batchSize, epoch, learningRate, splitPercentage, earlyStopping, verb):
        self.epochPbar.setMaximum(epoch)

        if self.cbResume.isChecked() == False or self.modelLearner.modelLoaded == False:
            nnList = self.getNNLayer()
            self.modelLearner.set(nnList, batchSize, epoch, learningRate, splitPercentage, earlyStopping, verb,
                                  TfCallback(self.epochPbar))

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