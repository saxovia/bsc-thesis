from PyQt6 import QtWidgets, QtCore, QtGui

class CustomMessageBox(QtWidgets.QMessageBox):
    def __init__(self, title, message, buttons, parent=None):
        if parent is None:
            parent = QtWidgets.QWidget()
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setText(message)
        self.setWindowFlags(QtCore.Qt.WindowType.FramelessWindowHint | QtCore.Qt.WindowType.Dialog)

        self.closeButton=QtWidgets.QPushButton(self)
        icon=QtGui.QIcon("../resources/icons/close.png")
        self.closeButton.setIcon(icon)
        self.closeButton.setFixedSize(16, 16)
        self.closeButton.setStyleSheet("background:transparent;")
        self.closeButton.clicked.connect(self.close)

        closeLayout=QtWidgets.QHBoxLayout()
        closeLayout.addStretch()
        closeLayout.addWidget(self.closeButton)

        mainLayout=QtWidgets.QVBoxLayout(self)
        mainLayout.addLayout(closeLayout)
        mainLayout.addWidget(self)
        self.setLayout(mainLayout)

        self.button_map = {}  # Store buttons +  labels
        for button_text, role, callback in buttons:
            if isinstance(role, int):
                role = QtWidgets.QMessageBox.ButtonRole(role)
            btn = self.addButton(button_text, role)
            btn.setObjectName(button_text.lower().replace(" ", "_"))
            self.button_map[btn] = callback
        # Messagebox specific styling
        self.setStyleSheet("""
            QMessageBox QPushButton#discard:hover {
                background-color: #c42b1c !important;
            }
        """)


    def exec(self):
        result = super().exec()
        clicked_button = self.clickedButton()
        if clicked_button in self.button_map and self.button_map[clicked_button]:
            self.button_map[clicked_button]()
    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._startPos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event):
        if self._startPos is not None:
            self.move(event.globalPosition().toPoint() - self._startPos)

    def mouseReleaseEvent(self, event):
        self._startPos = None