from PyQt6 import QtWidgets, QtCore

class CustomMessageBox(QtWidgets.QMessageBox):
    def __init__(self, title, message, buttons, parent=None):
        if parent is None:
            parent = QtWidgets.QWidget()
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setText(message)
        self.setWindowFlags(QtCore.Qt.WindowType.FramelessWindowHint | QtCore.Qt.WindowType.Dialog)

        self.button_map = {}  # Store buttons with their labels
        for button_text, role, callback in buttons:
            # Ensure role is a valid QMessageBox.ButtonRole
            if isinstance(role, int):
                role = QtWidgets.QMessageBox.ButtonRole(role)  # Convert to proper ButtonRole
            btn = self.addButton(button_text, role)  # Use button text and role
            btn.setObjectName(button_text.lower().replace(" ", "_"))
            self.button_map[btn] = callback  # Store callback for later

        self.setStyleSheet("""
            QMessageBox QPushButton#discard_button:hover {
                background-color: #c42b1c;
            }
        """)

    def exec(self):
        result = super().exec()
        clicked_button = self.clickedButton()
        if clicked_button in self.button_map and self.button_map[clicked_button]:
            self.button_map[clicked_button]()
