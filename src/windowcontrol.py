from PyQt6 import QtWidgets, QtCore

class WindowControl:
    def __init__(self, window):
        self.window = window
        self.title_bar = window.title_frame
        self.old_pos = None
        self.mousePressed = False
        self.title_bar.setMouseTracking(True)

    def mousePressEvent(self, event):
        if event.button()==QtCore.Qt.MouseButton.LeftButton and self.title_bar.underMouse():
            self.mousePressed=True
            self.old_pos=event.globalPosition().toPoint()

    def mouseMoveEvent(self, event):
        if self.mousePressed:
            delta=event.globalPosition().toPoint()-self.old_pos
            self.window.move(self.window.x()+delta.x(),self.window.y()+delta.y())
            self.old_pos=event.globalPosition().toPoint()

    def mouseReleaseEvent(self, event):
        if event.button()==QtCore.Qt.MouseButton.LeftButton:
            self.mousePressed=False
            screen = QtWidgets.QApplication.primaryScreen().geometry()
            window_pos = self.window.geometry()
            if window_pos.top() <= screen.top() + 10:
                self.window.showMaximized()
                self.window.is_maximized = True
