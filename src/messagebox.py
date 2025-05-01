from PyQt6 import QtWidgets, QtCore, QtGui

class CustomMessageBox(QtWidgets.QMessageBox):
    def __init__(self, title, message, buttons, parent=None):
        if parent is None:
            parent = QtWidgets.QWidget()
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setText(message)
        self.setWindowFlags(QtCore.Qt.WindowType.FramelessWindowHint | QtCore.Qt.WindowType.Dialog)

        container = QtWidgets.QWidget()
        container_layout = QtWidgets.QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)

        self.closeButton = QtWidgets.QPushButton(container)
        icon = QtGui.QIcon("../resources/icons/close.png")
        self.closeButton.setIcon(icon)
        self.closeButton.setFixedSize(16, 16)
        self.closeButton.setStyleSheet("background:transparent;")
        self.closeButton.clicked.connect(self.close)

        close_layout = QtWidgets.QHBoxLayout()
        close_layout.addStretch()
        close_layout.addWidget(self.closeButton)
        close_layout.setContentsMargins(0, 0, 0, 0)

        container_layout.addLayout(close_layout)
        self.layout().addWidget(container, 0, 0, 1, 1)
        
        self.layout().setContentsMargins(40, 40, 40, 40)
        self.layout().setSpacing(10)

        self.button_map = {}  # Store buttons + labels
        for button_text, role, callback in buttons:
            if isinstance(role, int):
                role = QtWidgets.QMessageBox.ButtonRole(role)
            btn = self.addButton(button_text, role)
            btn.setObjectName(button_text.lower().replace(" ", "_"))
            self.button_map[btn] = callback

        # Messagebox specific styling
        self.setStyleSheet("""
            QMessageBox {
                background-color: #35314F;
                color: white;
                border-color: transparent;
                border-width: 0px;
                border-radius: 10px;
                font-size: 14px;
                text-align: center;
            }
            QMessageBox QLabel {
                margin: 0px;
                padding: 0px;
            }
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