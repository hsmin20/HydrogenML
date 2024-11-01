import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QDesktopWidget, QAction, QFileDialog, \
    QVBoxLayout, QWidget, QPushButton, QGridLayout, QLabel, QInputDialog, \
    QLineEdit, QMessageBox, QCheckBox, QProgressBar, QHBoxLayout, QTableWidget, QTableWidgetItem, \
    QAbstractItemView, QHeaderView
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
import LayerDlg
from MachineLearner import TfCallback, NUM_DATA

class MLWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.distArrayName = ['S2m', 'S6m', 'S10m', 'S20m']
        self.distList = [2.0, 6.0, 10.0, 20.0]

        self.barrierPosArrayName = ['case1', 'case2', 'case3', 'case4', 'case5', 'case10', 'case11', 'case12', 'case13']
        self.barrierPosList = [5.0, 5.0, 5.0, 4.0, 6.0, 6.0, 6.0, 4.5, 5.5]

        self.barrierHeightList = [2.0, 3.0, 4.0, 2.0, 2.0, 2.5, 3.5, 2, 2]

        self.initUI()

        self.indexijs = []
        self.time_data = None
        self.time_data_n = None
        self.southSensors = []
        self.dataLoaded = False

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
        self.editFile = QLineEdit('Please load /DataRepositoy/hydrocfd_KAERI.csv')
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

    def initCommand(self):
        layout = QGridLayout()

        mlWithDataBtn = QPushButton('ML with Data')
        mlWithDataBtn.clicked.connect(self.doMachineLearning)
        self.cbResume = QCheckBox('Resume Learning')
        # self.cbResume.setChecked(True)
        saveModelBtn = QPushButton('Save Model')
        saveModelBtn.clicked.connect(self.saveModel)
        loadModelBtn =  QPushButton('Load Model')
        loadModelBtn.clicked.connect(self.loadModel)
        checkValBtn = QPushButton('Check Trained')
        checkValBtn.clicked.connect(self.checkVal)
        saveModelJSBtn = QPushButton('Save Model for JS')
        saveModelJSBtn.clicked.connect(self.saveModelJS)

        layout.addWidget(mlWithDataBtn, 0, 0, 1, 1)
        layout.addWidget(self.cbResume, 0, 1, 1, 1)
        layout.addWidget(saveModelBtn, 0, 2, 1, 1)
        layout.addWidget(loadModelBtn, 0, 3, 1, 1)
        layout.addWidget(checkValBtn, 0, 4, 1, 1)
        layout.addWidget(saveModelJSBtn, 0, 5, 1, 1)

        return layout

    def initGridTable(self):
        layout = QGridLayout()

        self.tableGridWidget = QTableWidget()

        self.tableGridWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tableGridWidget.setColumnCount(1)
        item = QTableWidgetItem('')
        self.tableGridWidget.setHorizontalHeaderItem(0, item)

        # barrier distance and height
        bHeightLabel = QLabel('Barrier Height')
        bHeightLabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.editBHeight = QLineEdit('2')
        self.editBHeight.setFixedWidth(100)
        bWidthLabel = QLabel('Barrier Width')
        bWidthLabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.editBWidth = QLineEdit('10')
        self.editBWidth.setFixedWidth(100)
        self.cbToBarrierPos = QCheckBox('Set Distance to Barrier Position')

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
        layout.addWidget(self.cbToBarrierPos, 9, 4)
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

        self.time_data = self.df.values[0:NUM_DATA, 1:2].flatten()
        timeMax = max(self.time_data)
        timeMin = min(self.time_data)
        self.time_data_n = (self.time_data - timeMin) / (timeMax - timeMin)
        self.time_diff_n = self.time_data_n[1] - self.time_data_n[0]

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
            s_data = self.southSensors[i]

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

        col_no = (i * len(self.distArray)) + (j + 2)
        sensorName = self.distArrayName[j] + self.barrierPosArrayName[i]

        data = self.df.values[:, col_no:col_no + 1].flatten()

        return sensorName, data

    def getMinMaxPresssureOfAllData(self):
        maxPressure = -100000
        minPressure = 100000
        for i in range(len(self.distList) * len(self.barrierPosArrayName) + 2):
            if i <= 1:
                continue

            one_data = self.df.values[::, i:i+1].flatten()
            maxp_local = max(one_data)
            minp_local = min(one_data)
            if maxp_local > maxPressure:
                maxPressure = maxp_local
            if minp_local < minPressure:
                minPressure = minp_local

        return maxPressure, minPressure

    def getMinMaxPressureOfLoadedData(self, pressureArray):
        maxPressure = -100000
        minPressure = 100000

        numSensors = len(pressureArray)
        for i in range(numSensors):
            s_data = pressureArray[i]
            maxp_local = max(s_data)
            minp_local = min(s_data)
            if maxp_local > maxPressure:
                maxPressure = maxp_local
            if minp_local < minPressure:
                minPressure = minp_local

        return maxPressure, minPressure

    def _normalize(self):
        barrierHeightList_n = np.copy(self.barrierHeightList)
        maxHeight = max(barrierHeightList_n)
        minHeight = min(barrierHeightList_n)
        for j in range(len(barrierHeightList_n)):
            barrierHeightList_n[j] = (barrierHeightList_n[j] - minHeight) / (maxHeight - minHeight)

        barrierPosList_n = np.copy(self.barrierPosList)
        maxPos = max(barrierPosList_n)
        minPos = min(barrierPosList_n)
        for j in range(len(barrierPosList_n)):
            barrierPosList_n[j] = (barrierPosList_n[j] - minPos) / (maxPos - minPos)

        distList_n = np.copy(self.distList)
        maxDist = max(distList_n)
        minDist = min(distList_n)
        for j in range(len(distList_n)):
            distList_n[j] = (distList_n[j] - minDist) / (maxDist - minDist)

        return barrierHeightList_n, barrierPosList_n, distList_n

    def onStartMachineLearning(self):
        pass

    def doMachineLearning(self):
        if not self.indexijs or not self.southSensors:
            QMessageBox.about(self, 'Warining', 'Load Data First')
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)

        self.onStartMachineLearning()

        epoch = int(self.editEpoch.text())
        if epoch < 1:
            QMessageBox.warning(self, 'warning', 'Epoch shall be greater than 0')
            return

        x_train_data, y_train_data, x_valid_data, y_valid_data, x_test_data, y_test_data = \
            self._prepareForMachineLearning()

        self._doMachineLearning(x_train_data, y_train_data, x_valid_data, y_valid_data, x_test_data, y_test_data)

        QApplication.restoreOverrideCursor()

    def saveModel(self):
        suggestion = '/srv/MLData/tfModel.h5'
        filename = QFileDialog.getSaveFileName(self, 'Save Model File', suggestion, filter="h5 file (*.h5)")
        if filename[0] != '':
            self.modelLearner.saveModel(filename[0])

        QMessageBox.information(self, 'Saved', 'Model is saved.')

    def saveModelJS(self):
        suggestion = '/srv/MLData'
        filename = QFileDialog.getSaveFileName(self, 'Save File', suggestion, "Model")
        if filename[0] != '':
            self.modelLearner.saveModelJS(filename[0])

            QMessageBox.information(self, 'Saved', 'Model is saved for Javascript.')

    def loadModel(self):
        fname = QFileDialog.getOpenFileName(self, 'Open h5 model file', '/srv/MLData',
                                            filter="h5 file (*.h5);;All files (*)")
        if fname[0]:
            self.modelLearner.loadModel(fname[0])

        QMessageBox.information(self, 'Loaded', 'Model is loaded.')

    def _prepareForChecking(self):
        numSensors = len(self.southSensors)

        if self.cbMinMax.isChecked() == True:
            maxp, minp = self.getMinMaxPresssureOfAllData()
        else:
            maxp, minp = self.getMinMaxPressureOfLoadedData(self.southSensors)

        barrierHeightList_n, barrierPosList_n, distList_n = self._normalize()

        x_data = np.zeros(shape=(NUM_DATA * numSensors, self.N_FEATURE))
        y_data = np.zeros(shape=(NUM_DATA * numSensors, 1))

        for i in range(numSensors):
            indexij = self.indexijs[i]
            row_num = indexij[0]
            col_num = indexij[1]

            barrierHeight = barrierHeightList_n[row_num]
            barrierPos = barrierPosList_n[row_num]
            senserPos = distList_n[col_num]

            s_data = self.southSensors[i]
            x_data_1, y_data_1 = self._prepareOneSensorData(barrierHeight, barrierPos, senserPos, s_data, maxp, minp)

            for j in range(NUM_DATA):
                x_data[i * NUM_DATA + j] = x_data_1[j]
                y_data[i * NUM_DATA + j] = y_data_1[j]

        return x_data, y_data

    def unnormalize(self, data, max, min):
        for i in range(len(data)):
            data[i] = data[i] * (max - min) + min

    def showPredictionGraphs(self, sDistHeightArray, distHeightArray, y_array):
        # numSensors = len(y_array)
        resultArray = []

        plt.figure()
        for i in range(len(y_array)):
            t_data = self.time_data
            s_data = y_array[i]

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

    def _doMachineLearning(self, x_train_data, y_train_data, x_valid_data, y_valid_data, x_test_data, y_test_data):
        epoch = int(self.editEpoch.text())
        batchSize = int(self.editBatch.text())
        learningRate = float(self.editLR.text())
        splitPercentage = float(self.editSplit.text())
        earlyStopping = self.cbEarlyStop.isChecked()
        verb = self.cbVerbose.isChecked()

        self.epochPbar.setMaximum(epoch)

        if self.cbResume.isChecked() == False or self.modelLearner.modelLoaded == False:
            nnList = self.getNNLayer()
            self.modelLearner.set(nnList, batchSize, epoch, learningRate, splitPercentage, earlyStopping, verb,
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

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
