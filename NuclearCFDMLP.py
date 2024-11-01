import sys
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from PyQt5.QtWidgets import QApplication, QFileDialog, QGridLayout, QLabel, QLineEdit, QMessageBox, QCheckBox, QProgressBar
from PyQt5.QtCore import Qt
import time
from MachineLearner import MachineLearner, NUM_DATA
from MLWindow import MLWindow
from sklearn.metrics import r2_score

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

class MLPMachineLearner(MachineLearner):
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

class MLPWindow(MLWindow):
    N_FEATURE = 4 # no barrier width
    DEFAULT_LAYER_FILE = 'defaultNuCFD.nn'

    def __init__(self):
        super().__init__()

        self.modelLearner = MLPMachineLearner()

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

        layout.addWidget(self.cbMinMax, 2, 0, 1, 2)
        layout.addWidget(self.cbTestData, 2, 2, 1, 2)
        layout.addWidget(self.epochPbar, 2, 4, 1, 4)

        return layout

    def _prepareOneSensorData(self, barrierHeight, barrierPos, senserPos, data, maxp, minp):
        datasize = len(data)
        cur_data = list(data)

        # don't remember why I didn't normalize pressure....
        for j in range(datasize):
            cur_data[j] = (cur_data[j] - minp) / (maxp - minp)

        x_data = np.zeros((datasize, self.N_FEATURE))

        # fill the data
        for j in range(datasize):
            x_data[j][0] = self.time_data_n[j]
            x_data[j][1] = barrierHeight
            x_data[j][2] = barrierPos
            x_data[j][3] = senserPos

        y_data = np.zeros((datasize, 1))
        for j in range(datasize):
            y_data[j][0] = cur_data[j]

        return x_data, y_data

    def _prepareForMachineLearning(self):
        numSensors = len(self.southSensors)
        if numSensors == 0:
            QMessageBox.warning(self, 'warning', 'No Training Data!')
            return

        # get maxp/minp from all of the training data
        if self.cbMinMax.isChecked() == True:
            maxp, minp = self.getMinMaxPresssureOfAllData()
        else:
            maxp, minp = self.getMinMaxPressureOfLoadedData(self.southSensors)

        barrierHeightList_n, barrierPosList_n, distList_n = self._normalize()

        trainValidIndex = []
        testIndex = []

        for i in range(numSensors):
            indexij = self.indexijs[i]
            row_num = indexij[0]
            col_num = indexij[1]

            if self.cbArray[row_num][col_num].checkState() == Qt.Checked:
                trainValidIndex.append(indexij)
            elif self.cbArray[row_num][col_num].checkState() == Qt.PartiallyChecked:
                testIndex.append(indexij)

        x_trainValid_data = np.zeros(shape=(NUM_DATA * len(trainValidIndex), self.N_FEATURE))
        y_trainValid_data = np.zeros(shape=(NUM_DATA * len(trainValidIndex), 1))

        x_test_data = np.zeros(shape=(NUM_DATA * len(testIndex), self.N_FEATURE))
        y_test_data = np.zeros(shape=(NUM_DATA * len(testIndex), 1))

        tvindex = 0
        teindex = 0
        for i in range(numSensors):
            indexij = self.indexijs[i]
            row_num = indexij[0]
            col_num = indexij[1]

            barrierHeight = barrierHeightList_n[row_num]
            barrierPos = barrierPosList_n[row_num]
            senserPos = distList_n[col_num]

            s_data = self.southSensors[i]
            x_data_1, y_data_1 = self._prepareOneSensorData(barrierHeight, barrierPos, senserPos, s_data, maxp, minp)

            if self.cbArray[row_num][col_num].checkState() == Qt.Checked:
                for j in range(NUM_DATA):
                    x_trainValid_data[tvindex * NUM_DATA + j] = x_data_1[j]
                    y_trainValid_data[tvindex * NUM_DATA + j] = y_data_1[j]
                tvindex += 1
            elif self.cbArray[row_num][col_num].checkState() == Qt.PartiallyChecked:
                for j in range(NUM_DATA):
                    x_test_data[teindex * NUM_DATA + j] = x_data_1[j]
                    y_test_data[teindex * NUM_DATA + j] = y_data_1[j]
                teindex += 1

        splitPercentage = float(self.editSplit.text())
        x_train_data, x_valid_data, y_train_data, y_valid_data = train_test_split(x_trainValid_data, y_trainValid_data,
                                                                                  test_size=splitPercentage,
                                                                                  random_state=42)

        return x_train_data, y_train_data, x_valid_data, y_valid_data, x_test_data, y_test_data

    def checkVal(self):
        if not self.indexijs or not self.southSensors:
            QMessageBox.about(self, 'Warining', 'Load Data First')
            return
        if self.modelLearner.modelLoaded == False:
            QMessageBox.about(self, 'Warning', 'Model is not created/loaded')
            return

        start_time = time.time()

        QApplication.setOverrideCursor(Qt.WaitCursor)

        x_data, y_data = self._prepareForChecking()

        y_predicted = self.modelLearner.predict(x_data)

        QApplication.restoreOverrideCursor()

        print("--- %s seconds ---" % (time.time() - start_time))

        datasize = len(y_data)
        x_display = np.zeros((datasize, 1))
        for j in range(datasize):
            x_display[j][0] = j

        r2All = r2_score(y_data, y_predicted)
        title = f'Machine Learning Validation (R2 = {r2All})'

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

        if self.cbMinMax.isChecked() == True:
            maxp, minp = self.getMinMaxPresssureOfAllData()
        else:
            maxp, minp = self.getMinMaxPressureOfLoadedData(self.southSensors)

        distCount = self.tableGridWidget.columnCount()
        barrierPosCount = self.tableGridWidget.rowCount()
        if distCount < 1 or barrierPosCount < 1:
            QMessageBox.warning(self, 'Warning', 'You need to add distance or barrier pos to predict')
            return

        start_time = time.time()

        QApplication.setOverrideCursor(Qt.WaitCursor)

        sDistBarrierPosArray = []
        distBarrierPosArray = []

        if self.cbToBarrierPos.isChecked() == True:
            for j in range(barrierPosCount):
                barrierPos = self.tableGridWidget.verticalHeaderItem(j).text()
                barrierPosOnly = barrierPos[1:len(barrierPos) - 1]

                barrierPosf = float(barrierPosOnly)
                distf = float(barrierPosf)

                sDistBarrierPosArray.append(barrierPos + barrierPos)
                distBarrierPosArray.append((distf, barrierPosf))
        else:
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

        barrierHeight = float(self.editBHeight.text())
        y_array = []

        for distBarrierPos in distBarrierPosArray:
            x_data = self._prepareOneSensorForPredict(distBarrierPos[0], distBarrierPos[1], barrierHeight)
            y_predicted = self.modelLearner.predict(x_data)
            self.unnormalize(y_predicted, maxp, minp)
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
                        s_data = y_array[i][j]

                        line = line + ',' + str(s_data)

                    line = line + '\n'
                    file.write(line)

    def _prepareOneSensorForPredict(self, sensorPos, barrierPos, barrierHeight):
        barrierHeightList_n = np.copy(self.barrierHeightList)
        maxHeight = max(barrierHeightList_n)
        minHeight = min(barrierHeightList_n)
        barrierHeight_n = (barrierHeight - minHeight) / (maxHeight - minHeight)

        barrierPosList_n = np.copy(self.barrierPosList)
        maxPos = max(barrierPosList_n)
        minPos = min(barrierPosList_n)
        barrierPos_n = (barrierPos - minPos) / (maxPos - minPos)

        distList_n = np.copy(self.distList)
        maxDist = max(distList_n)
        minDist = min(distList_n)
        sensorPos_n = (sensorPos - minDist) / (maxDist - minDist)


        x_data = np.zeros(shape=(NUM_DATA, self.N_FEATURE))

        for i in range(NUM_DATA):
            x_data[i][0] = self.time_data_n[i]
            x_data[i][1] = barrierHeight_n
            x_data[i][2] = barrierPos_n
            x_data[i][3] = sensorPos_n

        return x_data

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MLPWindow()
    sys.exit(app.exec_())