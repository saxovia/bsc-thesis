from PyQt6 import QtWidgets, QtCore

class CustomMessageBox(QtWidgets.QMessageBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setWindowTitle("")  # Hide title bar
        self.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        self.setText("Are you sure you want to continue?")

        # Add custom buttons
        confirm_button = self.addButton("Discard", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
        cancel_button = self.addButton("Cancel", QtWidgets.QMessageBox.ButtonRole.RejectRole)

        # Set object names for custom styling
        confirm_button.setObjectName("confirmButton")
        cancel_button.setObjectName("cancelButton")

        # Custom Stylesheet with Different Hover for "Confirm" Button
        self.setStyleSheet("""
            QMessageBox {
                background-color: black;
                color: white;
                border-radius: 15px;
                font-size: 14px;
            }
            QMessageBox QLabel {
                color: white;
                font-size: 16px;
            }
            /* Default Button Style */
            QMessageBox QPushButton {
                background-color: #333;
                color: white;
                border-radius: 10px;
                padding: 8px;
                font-size: 14px;
            }
            /* Different Hover for Confirm Button */
            QMessageBox QPushButton#confirmButton:hover {
                background-color: #008000;  /* Green Hover */
            }
            /* Default Hover for Other Buttons */
            QMessageBox QPushButton:hover {
                background-color: #444;
            }
            QMessageBox QPushButton:pressed {
                background-color: #222;
            }
        """)

        self.setWindowFlags(QtCore.Qt.WindowType.FramelessWindowHint | QtCore.Qt.WindowType.Dialog)

        # Show the message box and handle result
        self.exec()
        if self.clickedButton() == confirm_button:
            print("User confirmed!")
        else:
            print("User canceled!")

def show_custom_warning():
    CustomMessageBox()
