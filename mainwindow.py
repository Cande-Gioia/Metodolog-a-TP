from TP_Metodologia_GUI2 import *
from process_video import *
import sys, os
from PyQt5.QtWidgets import QApplication, QLabel, QBoxLayout, QMainWindow, QFileDialog, QApplication, QWidget, QPushButton, QAction, QLineEdit, QMessageBox, QRadioButton, QInputDialog
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget

from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, *args, **kwargs):
        QMainWindow.__init__(self, *args, **kwargs)
        self.setupUi(self)

        #Initialization
        self.setWindowTitle("TP Metodologia - Grupo 1")
        
        self.resize(1280, 720)

        # self.comboBox_FPS.setCurrentIndex(2)

        self.canvas = PlotCanvas(self)

        layout = QVBoxLayout(self.plotWidget)
        layout.addWidget(self.canvas)

        # VARIABLES
        self.zurdo = False # False es diestro, True es zurdo
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

        if self.comboBox_Jugador.currentIndex() == 1:
            self.zurdo = True
        else:
            self.zurdo = False

        path = self.file_path

        #TODO ACA VA EL CODIGO DE LA RED
        diferencias_user_pro, time_data = analizar_video(self.file_path, self.fps, self.zurdo)

        self.acpr = str(diferencias_user_pro[0])   #a: angulo c:cadera, r:rodilla, t:tobillo
        self.arpr = str(diferencias_user_pro[1])   #pr: previo
        self.atpr = str(diferencias_user_pro[2])   
        self.vcpr = str(diferencias_user_pro[3])   #v: velocidad
        self.vrpr = str(diferencias_user_pro[4])
        self.vtpr = str(diferencias_user_pro[5])

        self.acpo = str(diferencias_user_pro[6])   #po: posterior
        self.arpo = str(diferencias_user_pro[7])
        self.atpo = str(diferencias_user_pro[8])
        self.vcpo = str(diferencias_user_pro[9])
        self.vrpo = str(diferencias_user_pro[10])
        self.vtpo = str(diferencias_user_pro[11])

        self.publishcorrections()
        self.canvas.plot(time_data[0], time_data[1], time_data[2], time_data[3], time_data[4], time_data[5], time_data[6])

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

#############################################
#   Para plotear
#############################################
class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure()
        super().__init__(self.fig)
        self.setParent(parent)

    def plot(self, angle_A_1, angle_B_1, angle_C_1, vel_angle_A_1, vel_angle_B_1, vel_angle_C_1, shoot_frame_1):
        ax1 = self.fig.add_subplot(211)
        ax2 = self.fig.add_subplot(212)

            # Gráfico de los ángulos video 1
        t = np.arange(1, len(angle_A_1) + 1, step = 1 )

        ax1.plot(t, angle_A_1, color = 'black', label='Ángulo de Cadera 1')
        ax1.plot(t, angle_B_1, color = 'red', label=' Ángulo de Rodilla 1')
        ax1.plot(t, angle_C_1, color = 'blue', label=' Ángulo de Tobillo 1')
        ax1.axvline(x=shoot_frame_1, color='g', linestyle='--')
        ax1.set_xlabel("Frame")
        ax1.set_ylabel("Ángulo [°]")
        ax1.legend()
        ax1.grid()

        # Gráfico de las velocidades angulares video 1
        t = np.arange(1, len(vel_angle_A_1) + 1, step = 1 )

        ax2.plot(t, vel_angle_A_1, color = 'black', label='Cadera 1')
        ax2.plot(t, vel_angle_B_1, color = 'red', label='Rodilla 1')
        ax2.plot(t, vel_angle_C_1, color = 'blue', label="Tobillo 1")
        ax2.axvline(x=shoot_frame_1-1, color='g', linestyle='--')

        ax2.set_xlabel("Frame")
        ax2.set_ylabel("Velocidad angular [°/s]")
        ax2.legend()

        ax2.grid()

        self.draw()