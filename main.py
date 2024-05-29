from mainwindow import *


if __name__ == "__main__":
    '''
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
    '''
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()