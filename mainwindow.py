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

        self.frames = []

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

        self.pushButton_2.clicked.connect(lambda: self.frame(2))
        self.pushButton_3.clicked.connect(lambda: self.frame(3))
        self.pushButton_4.clicked.connect(lambda: self.frame(4))
        self.pushButton_5.clicked.connect(lambda: self.frame(5))
        self.pushButton_6.clicked.connect(lambda: self.frame(6))
        self.pushButton_7.clicked.connect(lambda: self.frame(7))
        self.pushButton_8.clicked.connect(lambda: self.frame(8))
        self.pushButton_9.clicked.connect(lambda: self.frame(9))
        self.pushButton_10.clicked.connect(lambda: self.frame(10))
        self.pushButton_11.clicked.connect(lambda: self.frame(11))
        self.pushButton_12.clicked.connect(lambda: self.frame(12))
        self.pushButton_13.clicked.connect(lambda: self.frame(13))

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

    def frame(self, num):
        frame = self.frames[num-2]

        self.player.setPosition((frame * 1000) // 30)
        self.player.pause()

        pass


    def processvideo(self):
        self.fps = int(self.comboBox_FPS.currentText())

        if self.comboBox_Jugador.currentIndex() == 1:
            self.zurdo = True
        else:
            self.zurdo = False

        path = self.file_path

        #TODO ACA VA EL CODIGO DE LA RED
        diferencias_user_pro, time_data, self.frames = analizar_video(self.file_path, self.fps, self.zurdo)

        # a: angulo, c:cadera, r:rodilla, t:tobillo

        # Ángulos previos
        if diferencias_user_pro[0] < 0:
            self.acpr = "No llevar tan atrás la pierna {:.2f}º".format(diferencias_user_pro[0])
        else:
            self.acpr = "Llevar más atrás la pierna {:.2f}º".format(diferencias_user_pro[0])

        if diferencias_user_pro[1] < 0:
            self.arpr = "Flexionar menos la rodilla {:.2f}º".format(diferencias_user_pro[1])
        else: 
            self.arpr = "Flexionar más la rodilla {:.2f}º".format(diferencias_user_pro[1])

        if diferencias_user_pro[2] < 0:
            self.atpr = "Extender más el tobillo {:.2f}º".format(diferencias_user_pro[2])
        else: 
            self.atpr = "Extender menos el tobillo {:.2f}º".format(diferencias_user_pro[2])

        # Velocidades previas
        if diferencias_user_pro[3] < 0:
            self.vcpr = "Aumentar la velocidad de la pierna {:.2f}º/seg".format(diferencias_user_pro[3])
        else:
            self.vcpr = "Disminuir la velocidad de la pierna {:.2f}º/seg".format(diferencias_user_pro[3])

        if diferencias_user_pro[4] < 0:
            self.vrpr = "Aumentar la velocidad de rotación de la rodilla {:.2f}º/seg".format(diferencias_user_pro[4])
        else: 
            self.vrpr = "Disminuir la velocidad de rotación de la rodilla {:.2f}º/seg".format(diferencias_user_pro[4])

        if diferencias_user_pro[5] < 0:
            self.vtpr = "Aumentar la velocidad de rotación del tobillo {:.2f}º/seg".format(diferencias_user_pro[5])
        else: 
            self.vtpr = "Disminuir la velocidad de rotación del tobillo {:.2f}º/seg".format(diferencias_user_pro[5])

        # Ángulos posteriores
        if diferencias_user_pro[6] < 0:
            self.acpo = "Extender más la pierna {:.2f}º".format(diferencias_user_pro[6])
        else:
            self.acpo = "No extender tanto la pierna {:.2f}º".format(diferencias_user_pro[6])

        if diferencias_user_pro[7] < 0:
            self.arpo = "Extender más la rodilla {:.2f}º".format(diferencias_user_pro[7])
        else: 
            self.arpo = "No extender tanto la rodilla {:.2f}º".format(diferencias_user_pro[7])

        if diferencias_user_pro[8] < 0:
            self.atpo = "Extender más el tobillo {:.2f}º".format(diferencias_user_pro[8])
        else: 
            self.atpo = "No extender tanto el tobillo {:.2f}º".format(diferencias_user_pro[8])
        
        # Velocidades posteriores
        if diferencias_user_pro[9] < 0:
            self.vcpo = "Aumentar la velocidad de la pierna {:.2f}º/seg".format(diferencias_user_pro[9])
        else:
            self.vcpo = "Disminuir la velocidad de la pierna {:.2f}º/seg".format(diferencias_user_pro[9])

        if diferencias_user_pro[10] < 0:
            self.vrpo = "Aumentar la velocidad de rotación de la rodilla {:.2f}º/seg".format(diferencias_user_pro[10])
        else: 
            self.vrpo = "Disminuir la velocidad de rotación de la rodilla {:.2f}º/seg".format(diferencias_user_pro[10])

        if diferencias_user_pro[11] < 0:
            self.vtpo = "Aumentar la velocidad de rotación del tobillo {:.2f}º/seg".format(diferencias_user_pro[11])
        else: 
            self.vtpo = "Disminuir la velocidad de rotación del tobillo {:.2f}º/seg".format(diferencias_user_pro[11])

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
        self.ax1 = self.fig.add_subplot(211)
        self.ax2 = self.fig.add_subplot(212)
        super().__init__(self.fig)
        self.setParent(parent)

    def plot(self, angle_A_1, angle_B_1, angle_C_1, vel_angle_A_1, vel_angle_B_1, vel_angle_C_1, shoot_frame_1):
        # Clear the subplots
        self.ax1.clear()
        self.ax2.clear()

        # Gráfico de los ángulos video 1
        t = np.arange(1, len(angle_A_1) + 1, step = 1 )

        self.ax1.plot(t, angle_A_1, color = 'black', label='Ángulo de Cadera 1')
        self.ax1.plot(t, angle_B_1, color = 'red', label=' Ángulo de Rodilla 1')
        self.ax1.plot(t, angle_C_1, color = 'blue', label=' Ángulo de Tobillo 1')
        self.ax1.axvline(x=shoot_frame_1, color='g', linestyle='--', label=' Momento de disparo')
        self.ax1.set_xlabel("Frame")
        self.ax1.set_ylabel("Ángulo [°]")
        self.ax1.legend()
        self.ax1.grid()

        # Gráfico de las velocidades angulares video 1
        t = np.arange(1, len(vel_angle_A_1) + 1, step = 1 )

        self.ax2.plot(t, vel_angle_A_1, color = 'black', label='Cadera 1')
        self.ax2.plot(t, vel_angle_B_1, color = 'red', label='Rodilla 1')
        self.ax2.plot(t, vel_angle_C_1, color = 'blue', label="Tobillo 1")
        self.ax2.axvline(x=shoot_frame_1-1, color='g', linestyle='--', label=' Momento de disparo')

        self.ax2.set_xlabel("Frame")
        self.ax2.set_ylabel("Velocidad angular [°/s]")
        self.ax2.legend()

        self.ax2.grid()

        self.draw()