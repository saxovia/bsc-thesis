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
        self.is_maximized = False


    def toggle_maximize_restore(self):
        if self.isFullScreen():
            self.showNormal()
            self.is_maximized = False
        else:
            self.showFullScreen()
            self.is_maximized = True

    # Define a method to be called when the button is clicked
    def on_button_click(self):
        print("Button from the UI was clicked!")

    # Dragging functions

    def mousePressEvent(self, event):
        if event.button()==QtCore.Qt.MouseButton.LeftButton:
            self.mousePressed=True
            self.old_pos=event.globalPosition().toPoint()
            

    def mouseMoveEvent(self, event):
        if self.mousePressed:
            delta=event.globalPosition().toPoint()-self.old_pos
            self.move(self.x()+delta.x(),self.y()+delta.y())
            self.old_pos=event.globalPosition().toPoint()

    def mouseReleaseEvent(self, event):
        if event.button()==QtCore.Qt.MouseButton.LeftButton:
            self.mousePressed=False
            screen = QtWidgets.QApplication.primaryScreen().geometry()
            window_pos = self.geometry()
            if window_pos.top() <= screen.top() + 10:
                    self.showMaximized()
                    self.is_maximized = True