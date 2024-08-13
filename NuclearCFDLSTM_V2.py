import sys
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras import layers
from tensorflow.keras.metrics import RootMeanSquaredError
from PyQt5.QtWidgets import QApplication, QFileDialog, QGridLayout, QLabel, QLineEdit, QMessageBox, QCheckBox, QProgressBar
from PyQt5.QtCore import Qt
import time
from MachineLearner import MachineLearner, NUM_DATA
from MLWindow import MLWindow
from sklearn.metrics import r2_score

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

class LSTMMachineLearner(MachineLearner):
    def setWindowSize(self, windowSize):
        self.windowSize = windowSize

    def createModel(self, nnList):
        adamOpt = tf.keras.optimizers.Adam(learning_rate=self.learningRate)
        model = tf.keras.Sequential()

        n_features = LSTMWindow.N_FEATURE
        model.add(layers.InputLayer((self.windowSize, n_features)))
        for n in range(len(nnList)):
            noOfNeuron = nnList[n][0]
            activationFunc = nnList[n][1]
            if activationFunc == 'LSTM':
                if n == len(nnList) - 3:
                    model.add(layers.LSTM(noOfNeuron))
                else:
                    model.add(layers.LSTM(noOfNeuron, return_sequences=True))
            elif activationFunc == 'GRU':
                if n == len(nnList) - 3:
                    model.add(layers.GRU(noOfNeuron))
                else:
                    model.add(layers.GRU(noOfNeuron, return_sequences=True))
            else:
                model.add(layers.Dense(noOfNeuron, activation=activationFunc))

        model.compile(loss='mse', optimizer=adamOpt, metrics=RootMeanSquaredError())

        if self.verbose:
            model.summary()

        return model

class LSTMWindow(MLWindow):
    N_FEATURE = 5
    DEFAULT_LAYER_FILE = 'defaultNuCFDLSTM.nn'

    def __init__(self):
        super().__init__()

        self.modelLearner = LSTMMachineLearner()

    def initMLOption(self):
        layout = QGridLayout()

        batchLabel = QLabel('Batch Size')
        batchLabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.editBatch = QLineEdit('32')
        self.editBatch.setFixedWidth(100)
        epochLabel = QLabel('Epoch')
        epochLabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.editEpoch = QLineEdit('20')
        self.editEpoch.setFixedWidth(100)
        lrLabel = QLabel('Learning Rate')
        self.editLR = QLineEdit('0.0004')
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
        self.editTmMulti = QLineEdit('63')
        self.editTmMulti.setFixedWidth(100)
        distMultiLabel = QLabel('Dist Multiplier')
        self.editDistMulti = QLineEdit('20')
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

    def onStartMachineLearning(self):
        windowSize = int(self.editWidSize.text())
        self.modelLearner.setWindowSize(windowSize)

    def _processForMachineLearning(self, xdata):
        return xdata

    def _prepareForMachineLearning(self):
        numSensors = len(self.southSensors)
        if numSensors == 0:
            QMessageBox.warning(self, 'warning', 'No Training Data!')
            return

        if self.cbMinMax.isChecked() == True:
            maxp, minp = self.getMinMaxPresssureOfAllData()
        else:
            maxp, minp = self.getMinMaxPressureOfLoadedData(self.southSensors)

        barrierHeightList_n, barrierPosList_n, distList_n = self._normalize()

        tmMultiplier = float(self.editTmMulti.text())
        distMultiplier = float(self.editDistMulti.text())

        # separate data to train & validation
        trainValidIndex = []
        testIndex = []

        for i in range(numSensors):
            index_ij = self.indexijs[i]
            row_num = index_ij[0]
            col_num = index_ij[1]

            if self.cbArray[row_num][col_num].checkState() == Qt.Checked:
                trainValidIndex.append(index_ij)
            elif self.cbArray[row_num][col_num].checkState() == Qt.PartiallyChecked:
                testIndex.append(index_ij)

        if len(trainValidIndex) == 0:
            QMessageBox.warning(self, 'warning', 'No Training Data!')
            return

        windowSize = int(self.editWidSize.text())

        x_trainValid_data = []
        y_trainValid_data = []

        x_test_data = []
        y_test_data = []

        for i in range(numSensors):
            index_ij = self.indexijs[i]
            row_num = index_ij[0]
            col_num = index_ij[1]

            barrierHeight = barrierHeightList_n[row_num]
            barrierPos = barrierPosList_n[row_num]
            senserPos = distList_n[col_num]
            s_data = self.southSensors[i]

            for j in range(NUM_DATA - windowSize):
                x_data = np.zeros((windowSize, self.N_FEATURE))
                for k in range(windowSize):
                    x_data[k][0] = self.time_data_n[j + k] * tmMultiplier
                    x_data[k][1] = barrierHeight
                    x_data[k][2] = barrierPos * distMultiplier
                    x_data[k][3] = senserPos
                    x_data[k][4] = (s_data[j + k] - minp) / (maxp - minp)

                x_data = self._processForMachineLearning(x_data)
                y_data = (s_data[j + windowSize] - minp) / (maxp - minp)

                if self.cbArray[row_num][col_num].checkState() == Qt.Checked:
                    x_trainValid_data.append(x_data)
                    y_trainValid_data.append(y_data)
                elif self.cbArray[row_num][col_num].checkState() == Qt.PartiallyChecked:
                    x_test_data.append(x_data)
                    y_test_data.append(y_data)

        splitPercentage = float(self.editSplit.text())
        x_train_data, x_valid_data, y_train_data, y_valid_data = train_test_split(x_trainValid_data, y_trainValid_data,
                                                                                  test_size=splitPercentage,
                                                                                  random_state=42)

        return np.array(x_train_data), np.array(y_train_data), np.array(x_valid_data), np.array(y_valid_data), \
            np.array(x_test_data), np.array(y_test_data)

    def _processForCheckVal(self, x_data, windowSize):
        x_input = np.array(x_data)
        x_input = x_input.reshape((1, windowSize, self.N_FEATURE))

        return x_input

    def checkVal(self):
        if self.modelLearner.modelLoaded == False:
            QMessageBox.about(self, 'Warning', 'Model is not created/loaded')
            return

        numSensors = len(self.southSensors)
        if numSensors == 0:
            QMessageBox.warning(self, 'warning', 'No Training Data!')
            return

        if self.cbMinMax.isChecked() == True:
            maxp, minp = self.getMinMaxPresssureOfAllData()
        else:
            maxp, minp = self.getMinMaxPressureOfLoadedData(self.southSensors)

        barrierHeightList_n, barrierPosList_n, distList_n = self._normalize()

        tmMultiplier = float(self.editTmMulti.text())
        distMultiplier = float(self.editDistMulti.text())
        windowSize = int(self.editWidSize.text())

        start_time = time.time()

        QApplication.setOverrideCursor(Qt.WaitCursor)

        totalSize = len(self.indexijs) * (NUM_DATA - windowSize)
        self.epochPbar.setMaximum(totalSize)

        y_data = []
        y_pred = []

        for i in range(numSensors):
            index_ij = self.indexijs[i]
            row_num = index_ij[0]
            col_num = index_ij[1]

            barrierHeight = barrierHeightList_n[row_num]
            barrierPos = barrierPosList_n[row_num]
            senserPos = distList_n[col_num]
            s_data_raw = self.southSensors[i]

            s_data = (s_data_raw - minp) / (maxp - minp)

            y_data.extend(s_data)

            x_data = [[0] * self.N_FEATURE for i in range(windowSize)]
            for k in range(windowSize):
                x_data[k][0] = self.time_data_n[k] * tmMultiplier
                x_data[k][1] = barrierHeight
                x_data[k][2] = barrierPos * distMultiplier
                x_data[k][3] = senserPos
                x_data[k][4] = s_data[k]

            time_diff = x_data[1][0] - x_data[0][0]

            for j in range(NUM_DATA - windowSize):
                x_input = self._processForCheckVal(x_data, windowSize)

                y_predicted = self.modelLearner.predict(x_input)

                p_predicted = y_predicted[0][0]
                y_pred.append(p_predicted)

                x_data.pop(0)
                x_data.append(x_data[-1].copy())
                x = x_data[windowSize - 1][0]
                x_data[windowSize - 1][0] = x + time_diff
                x_data[windowSize - 1][4] = p_predicted

                self.epochPbar.setValue(i * (NUM_DATA - windowSize) + j)

            y_data = y_data[:len(y_pred)]

        QApplication.restoreOverrideCursor()

        print("--- %s seconds ---" % (time.time() - start_time))

        self._drawCheckValGraph(y_data, y_pred)

    def _drawCheckValGraph(self, y_data, y_pred):
        datasize = len(y_data)
        x_display = np.zeros((datasize, 1))
        for j in range(datasize):
            x_display[j][0] = j

        r2All = r2_score(y_data, y_pred)
        title = f'LSTM Validation (R2 = {r2All})'

        plt.figure()
        plt.scatter(x_display, y_data, label='original data', color="red", s=1)
        plt.scatter(x_display, y_pred, label='predicted', color="blue", s=1)
        plt.title(title)
        plt.xlabel('time (ms)')
        plt.ylabel('pressure (kPa)')
        plt.legend(loc='upper right')
        plt.grid()

        plt.show()

    def _getNormalized(self, barrierHeight, barrierPos, sensorPos):
        maxHeight = max(self.barrierHeightList)
        minHeight = min(self.barrierHeightList)
        barrierHeight_n = (barrierHeight - minHeight) / (maxHeight - minHeight)

        maxPos = max(self.barrierPosList)
        minPos = min(self.barrierPosList)
        barrierPos_n = (barrierPos - minPos) / (maxPos - minPos)

        maxDist = max(self.distList)
        minDist = min(self.distList)
        sensorPos_n = (sensorPos - minDist) / (maxDist - minDist)

        return barrierHeight_n, barrierPos_n, sensorPos_n

    def predict(self):
        if self.modelLearner.modelLoaded == False:
            QMessageBox.about(self, 'Warning', 'Model is not created/loaded')
            return

        distCount = self.tableGridWidget.columnCount()
        barrierPosCount = self.tableGridWidget.rowCount()
        if distCount < 1 or barrierPosCount < 1:
            QMessageBox.warning(self, 'Warning', 'You need to add distance or barrier pos to predict')
            return

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

        y_array = []

        if self.cbMinMax.isChecked() == True:
            maxp, minp = self.getMinMaxPresssureOfAllData()
        else:
            maxp, minp = self.getMinMaxPressureOfLoadedData(self.southSensors)

        tmMultiplier = float(self.editTmMulti.text())
        distMultiplier = float(self.editDistMulti.text())

        windowSize = int(self.editWidSize.text())

        bHeight = float(self.editBHeight.text())
        # bWidth = float(self.editBWidth.text())

        s_data_raw = self.southSensors[0]
        s_data = (s_data_raw - minp) / (maxp - minp)

        totalSize = len(distBarrierPosArray) * (NUM_DATA - windowSize)
        self.epochPbar.setMaximum(totalSize)

        i = 0
        for distBarrierPos in distBarrierPosArray:
            sensorPos = distBarrierPos[0]
            barrierPos = distBarrierPos[1]

            barrierHeight_n, barrierPos_n, sensorPos_n = self._getNormalized(bHeight, barrierPos, sensorPos)

            x_data = [[0] * self.N_FEATURE for i in range(windowSize)]

            for k in range(windowSize):
                x_data[k][0] = self.time_data_n[k] * tmMultiplier
                x_data[k][1] = barrierHeight_n
                x_data[k][2] = barrierPos_n * distMultiplier
                x_data[k][3] = sensorPos_n
                x_data[k][4] = s_data[k]

            time_diff = x_data[1][0] - x_data[0][0]

            y_pred_arr = []
            for wi in range(windowSize):
                y_pred_arr.append(x_data[wi][4])
            for j in range(NUM_DATA - windowSize):
                x_input = self._processForCheckVal(x_data, windowSize)
                y_predicted = self.modelLearner.predict(x_input)

                p_predicted = y_predicted[0][0]
                y_pred_arr.append(p_predicted)

                x_data.pop(0)
                x_data.append(x_data[-1].copy())
                x = x_data[windowSize - 1][0]
                x_data[windowSize - 1][0] = x + time_diff
                x_data[windowSize - 1][4] = p_predicted

                self.epochPbar.setValue(i * (NUM_DATA - windowSize) + j)

            i += 1
            self.unnormalize(y_pred_arr, maxp, minp)
            y_array.append(y_pred_arr)

        QApplication.restoreOverrideCursor()

        print("--- %s seconds ---" % (time.time() - start_time))

        resultArray = self.showPredictionGraphs(sDistBarrierPosArray, distBarrierPosArray, y_array)

        reply = QMessageBox.question(self, 'Message', 'Do you want to save pressureto a file?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            self._savePressureToFile(sDistBarrierPosArray, y_array)

    def _savePressureToFile(self, sDistBarrierPosArray, y_array):
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

                if i != (len(sDistBarrierPosArray) - 1):
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

    def showPredictionGraphs(self, sDistBarrierPosArray, distBarrierPosArray, y_array):
        # numSensors = len(y_array)
        resultArray = []

        plt.figure()
        for i in range(len(y_array)):
            t_data = self.time_data
            s_data = y_array[i]

            distBarrierPos = distBarrierPosArray[i]
            lab = sDistBarrierPosArray[i]

            distance = distBarrierPos[0]
            barrierPos = distBarrierPos[1]

            overpressure = max(s_data)

            dispLabel = lab #+ '/op=' + format(overpressure[0], '.2f') + '/impulse=' + format(impulse, '.2f')

            resultArray.append(str(distance) + ',' + str(barrierPos))

            plt.scatter(t_data, s_data, label=dispLabel, s=1)

        plt.title('Pressure Graph')
        plt.xlabel('time (s)')
        plt.ylabel('pressure (kPa)')
        plt.legend(loc='upper right', markerscale=4.)
        plt.grid()

        plt.show()

        return resultArray

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = LSTMWindow()
    sys.exit(app.exec_())