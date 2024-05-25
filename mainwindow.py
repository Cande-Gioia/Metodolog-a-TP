from TP_Metodologia_GUI import *
import sys, os
from PyQt5.QtWidgets import QApplication, QLabel, QBoxLayout, QMainWindow, QFileDialog, QApplication, QWidget, QPushButton, QAction, QLineEdit, QMessageBox, QRadioButton, QInputDialog
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, *args, **kwargs):
        QMainWindow.__init__(self, *args, **kwargs)
        self.setupUi(self)

        #Initialization
        self.setWindowTitle("TP Metodologia - Grupo 1")
        
        self.resize(1920, 1080)

        # VARIABLES
        self.jugador = 0 #0 es diestro, 1 es zurdo
        self.fps = 60
        self.file_path = 0

        self.acpr = '0'   #a: angulo c:cadera, r:rodilla, t:tobillo
        self.arpr = '0'   #pr: previo
        self.atpr = '0'   
        self.vcpr = '0'   #v: velocidad
        self.vrpr = '0'
        self.vtpr = '0'

        self.acpo = '0'   #po: posterior
        self.arpo = '0'
        self.atpo = '0'
        self.vcpo = '0'
        self.vrpo = '0'
        self.vtpo = '0'

        self.pushButton_procesar.clicked.connect(self.processvideo)
        self.pushButton_play.clicked.connect(self.playVideo)

        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self,event): 
        if event.mimeData().hasUrls():
            event.setDropAction(Qt.CopyAction)
            self.file_path = event.mimeData().urls()[0].toLocalFile()
            self.set_Video(self.file_path)
            event.accept()
        else:
            event.ignore()
    
    def set_Video(self, file_path):
        self.player = QMediaPlayer()
        self.player.setMedia(QMediaContent(QUrl.fromLocalFile(file_path)))
        self.player.setVideoOutput(self.videoframe)
        self.videoframe.show()
        self.player.play()
    
    def playVideo(self):
        self.player.play()


    def processvideo(self):
        self.fps = int(self.comboBox_FPS.currentText())
        self.jugador = self.comboBox_Jugador.currentIndex()
        path = self.file_path

        #TODO ACA VA EL CODIGO DE LA RED

        self.publishcorrections()

    def publishcorrections(self):
        #previos
        self.label_acprm.setText(self.acpr)
        self.label_arprm.setText(self.arpr)
        self.label_atprm.setText(self.atpr)
        self.label_vcprm.setText(self.vcpr)
        self.label_vrprm.setText(self.vrpr)
        self.label_vtprm.setText(self.vtpr)
        #posteriores
        self.label_acpom.setText(self.acpo)
        self.label_arpom.setText(self.arpo)
        self.label_atpom.setText(self.atpo)
        self.label_vcpom.setText(self.vcpo)
        self.label_vrpom.setText(self.vrpo)
        self.label_vtpom.setText(self.vtpo)
