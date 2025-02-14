from PyQt6 import QtWidgets, uic, QtCore

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Hello World')
        uic.loadUi('file.ui', self)
        self.pushButton.clicked.connect(self.on_button_click)
        self.setWindowFlag(QtCore.Qt.WindowType.FramelessWindowHint)

        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        self.maximize_button.setCheckable(True)


        self.minimize_button.clicked.connect(self.showMinimized)
        self.maximize_button.clicked.connect(self.toggle_maximize_restore)
        self.close_button.clicked.connect(self.close)
        
        self.old_pos = self.pos()
        self.mousePressed = False


    def toggle_maximize_restore(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    # Define a method to be called when the button is clicked
    def on_button_click(self):
        print("Button from the UI was clicked!")
