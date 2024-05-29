# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'TP_Metodologia.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtMultimediaWidgets import QVideoWidget


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(657, 434)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.label_arrastre = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setItalic(True)
        self.label_arrastre.setFont(font)
        self.label_arrastre.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_arrastre.setObjectName("label_arrastre")
        self.gridLayout.addWidget(self.label_arrastre, 1, 3, 1, 1)
        self.boxframe = QtWidgets.QFrame(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.boxframe.sizePolicy().hasHeightForWidth())
        self.boxframe.setSizePolicy(sizePolicy)
        self.boxframe.setFrameShape(QtWidgets.QFrame.Box)
        self.boxframe.setFrameShadow(QtWidgets.QFrame.Plain)
        self.boxframe.setObjectName("boxframe")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.boxframe)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.videoframe = QVideoWidget(self.boxframe)
        self.videoframe.setObjectName("videoframe")
        self.gridLayout_2.addWidget(self.videoframe, 0, 0, 1, 1)
        self.gridLayout.addWidget(self.boxframe, 0, 0, 1, 4)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_acpr = QtWidgets.QLabel(self.centralwidget)
        self.label_acpr.setObjectName("label_acpr")
        self.verticalLayout.addWidget(self.label_acpr)
        self.label_arpr = QtWidgets.QLabel(self.centralwidget)
        self.label_arpr.setObjectName("label_arpr")
        self.verticalLayout.addWidget(self.label_arpr)
        self.label_atpr = QtWidgets.QLabel(self.centralwidget)
        self.label_atpr.setObjectName("label_atpr")
        self.verticalLayout.addWidget(self.label_atpr)
        self.label_vcpr = QtWidgets.QLabel(self.centralwidget)
        self.label_vcpr.setObjectName("label_vcpr")
        self.verticalLayout.addWidget(self.label_vcpr)
        self.label_vrpr = QtWidgets.QLabel(self.centralwidget)
        self.label_vrpr.setObjectName("label_vrpr")
        self.verticalLayout.addWidget(self.label_vrpr)
        self.label_vtpr = QtWidgets.QLabel(self.centralwidget)
        self.label_vtpr.setObjectName("label_vtpr")
        self.verticalLayout.addWidget(self.label_vtpr)
        self.label_acpo = QtWidgets.QLabel(self.centralwidget)
        self.label_acpo.setObjectName("label_acpo")
        self.verticalLayout.addWidget(self.label_acpo)
        self.label_arpo = QtWidgets.QLabel(self.centralwidget)
        self.label_arpo.setObjectName("label_arpo")
        self.verticalLayout.addWidget(self.label_arpo)
        self.label_atpo = QtWidgets.QLabel(self.centralwidget)
        self.label_atpo.setObjectName("label_atpo")
        self.verticalLayout.addWidget(self.label_atpo)
        self.label_vcpo = QtWidgets.QLabel(self.centralwidget)
        self.label_vcpo.setObjectName("label_vcpo")
        self.verticalLayout.addWidget(self.label_vcpo)
        self.label_vrpo = QtWidgets.QLabel(self.centralwidget)
        self.label_vrpo.setObjectName("label_vrpo")
        self.verticalLayout.addWidget(self.label_vrpo)
        self.label_vtpo = QtWidgets.QLabel(self.centralwidget)
        self.label_vtpo.setObjectName("label_vtpo")
        self.verticalLayout.addWidget(self.label_vtpo)
        self.gridLayout.addLayout(self.verticalLayout, 0, 4, 1, 1)
        self.label_Jugador = QtWidgets.QLabel(self.centralwidget)
        self.label_Jugador.setObjectName("label_Jugador")
        self.gridLayout.addWidget(self.label_Jugador, 1, 0, 1, 1)
        self.label_FPS = QtWidgets.QLabel(self.centralwidget)
        self.label_FPS.setObjectName("label_FPS")
        self.gridLayout.addWidget(self.label_FPS, 1, 1, 1, 1)
        self.comboBox_Jugador = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_Jugador.setObjectName("comboBox_Jugador")
        self.comboBox_Jugador.addItem("")
        self.comboBox_Jugador.addItem("")
        self.gridLayout.addWidget(self.comboBox_Jugador, 2, 0, 1, 1)
        self.comboBox_FPS = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_FPS.setObjectName("comboBox_FPS")
        self.comboBox_FPS.addItem("")
        self.comboBox_FPS.addItem("")
        self.comboBox_FPS.addItem("")
        self.comboBox_FPS.addItem("")
        self.gridLayout.addWidget(self.comboBox_FPS, 2, 1, 1, 1)
        self.pushButton_play = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_play.setObjectName("pushButton_play")
        self.gridLayout.addWidget(self.pushButton_play, 2, 2, 1, 1)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_acprm = QtWidgets.QLabel(self.centralwidget)
        self.label_acprm.setAlignment(QtCore.Qt.AlignCenter)
        self.label_acprm.setObjectName("label_acprm")
        self.verticalLayout_2.addWidget(self.label_acprm)
        self.label_arprm = QtWidgets.QLabel(self.centralwidget)
        self.label_arprm.setAlignment(QtCore.Qt.AlignCenter)
        self.label_arprm.setObjectName("label_arprm")
        self.verticalLayout_2.addWidget(self.label_arprm)
        self.label_atprm = QtWidgets.QLabel(self.centralwidget)
        self.label_atprm.setAlignment(QtCore.Qt.AlignCenter)
        self.label_atprm.setObjectName("label_atprm")
        self.verticalLayout_2.addWidget(self.label_atprm)
        self.label_vcprm = QtWidgets.QLabel(self.centralwidget)
        self.label_vcprm.setAlignment(QtCore.Qt.AlignCenter)
        self.label_vcprm.setObjectName("label_vcprm")
        self.verticalLayout_2.addWidget(self.label_vcprm)
        self.label_vrprm = QtWidgets.QLabel(self.centralwidget)
        self.label_vrprm.setAlignment(QtCore.Qt.AlignCenter)
        self.label_vrprm.setObjectName("label_vrprm")
        self.verticalLayout_2.addWidget(self.label_vrprm)
        self.label_vtprm = QtWidgets.QLabel(self.centralwidget)
        self.label_vtprm.setAlignment(QtCore.Qt.AlignCenter)
        self.label_vtprm.setObjectName("label_vtprm")
        self.verticalLayout_2.addWidget(self.label_vtprm)
        self.label_acpom = QtWidgets.QLabel(self.centralwidget)
        self.label_acpom.setAlignment(QtCore.Qt.AlignCenter)
        self.label_acpom.setObjectName("label_acpom")
        self.verticalLayout_2.addWidget(self.label_acpom)
        self.label_arpom = QtWidgets.QLabel(self.centralwidget)
        self.label_arpom.setAlignment(QtCore.Qt.AlignCenter)
        self.label_arpom.setObjectName("label_arpom")
        self.verticalLayout_2.addWidget(self.label_arpom)
        self.label_atpom = QtWidgets.QLabel(self.centralwidget)
        self.label_atpom.setAlignment(QtCore.Qt.AlignCenter)
        self.label_atpom.setObjectName("label_atpom")
        self.verticalLayout_2.addWidget(self.label_atpom)
        self.label_vcpom = QtWidgets.QLabel(self.centralwidget)
        self.label_vcpom.setAlignment(QtCore.Qt.AlignCenter)
        self.label_vcpom.setObjectName("label_vcpom")
        self.verticalLayout_2.addWidget(self.label_vcpom)
        self.label_vrpom = QtWidgets.QLabel(self.centralwidget)
        self.label_vrpom.setAlignment(QtCore.Qt.AlignCenter)
        self.label_vrpom.setObjectName("label_vrpom")
        self.verticalLayout_2.addWidget(self.label_vrpom)
        self.label_vtpom = QtWidgets.QLabel(self.centralwidget)
        self.label_vtpom.setAlignment(QtCore.Qt.AlignCenter)
        self.label_vtpom.setObjectName("label_vtpom")
        self.verticalLayout_2.addWidget(self.label_vtpom)
        self.gridLayout.addLayout(self.verticalLayout_2, 0, 5, 1, 1)
        self.pushButton_procesar = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_procesar.setObjectName("pushButton_procesar")
        self.gridLayout.addWidget(self.pushButton_procesar, 2, 5, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem, 2, 3, 1, 2)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_arrastre.setText(_translate("MainWindow", "Arrastre un video para procesar  "))
        self.label_acpr.setText(_translate("MainWindow", "Ángulo cadera previo"))
        self.label_arpr.setText(_translate("MainWindow", "Ángulo rodilla previo"))
        self.label_atpr.setText(_translate("MainWindow", "Ángulo tobillo previo"))
        self.label_vcpr.setText(_translate("MainWindow", "Velocidad cadera previa"))
        self.label_vrpr.setText(_translate("MainWindow", "Velocidad rodilla previa"))
        self.label_vtpr.setText(_translate("MainWindow", "Velocidad tobillo previa"))
        self.label_acpo.setText(_translate("MainWindow", "Ángulo cadera posterior"))
        self.label_arpo.setText(_translate("MainWindow", "Ángulo rodilla posterior"))
        self.label_atpo.setText(_translate("MainWindow", "Ángulo tobillo posterior"))
        self.label_vcpo.setText(_translate("MainWindow", "Velocidad cadera posterior"))
        self.label_vrpo.setText(_translate("MainWindow", "Velocidad rodilla posterior"))
        self.label_vtpo.setText(_translate("MainWindow", "Velocidad tobillo posterior"))
        self.label_Jugador.setText(_translate("MainWindow", "Jugador"))
        self.label_FPS.setText(_translate("MainWindow", "FPS"))
        self.comboBox_Jugador.setCurrentText(_translate("MainWindow", "Diestro"))
        self.comboBox_Jugador.setItemText(0, _translate("MainWindow", "Diestro"))
        self.comboBox_Jugador.setItemText(1, _translate("MainWindow", "Zurdo"))
        self.comboBox_FPS.setCurrentText(_translate("MainWindow", "60"))
        self.comboBox_FPS.setItemText(0, _translate("MainWindow", "60"))
        self.comboBox_FPS.setItemText(1, _translate("MainWindow", "120"))
        self.comboBox_FPS.setItemText(2, _translate("MainWindow", "240"))
        self.comboBox_FPS.setItemText(3, _translate("MainWindow", "360"))
        self.pushButton_play.setText(_translate("MainWindow", "Play"))
        self.label_acprm.setText(_translate("MainWindow", "-"))
        self.label_arprm.setText(_translate("MainWindow", "-"))
        self.label_atprm.setText(_translate("MainWindow", "-"))
        self.label_vcprm.setText(_translate("MainWindow", "-"))
        self.label_vrprm.setText(_translate("MainWindow", "-"))
        self.label_vtprm.setText(_translate("MainWindow", "-"))
        self.label_acpom.setText(_translate("MainWindow", "-"))
        self.label_arpom.setText(_translate("MainWindow", "-"))
        self.label_atpom.setText(_translate("MainWindow", "-"))
        self.label_vcpom.setText(_translate("MainWindow", "-"))
        self.label_vrpom.setText(_translate("MainWindow", "-"))
        self.label_vtpom.setText(_translate("MainWindow", "-"))
        self.pushButton_procesar.setText(_translate("MainWindow", "Procesar"))

'''
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
'''