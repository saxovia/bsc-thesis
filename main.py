try:
    from PyQt6 import QtWidgets, uic
except ImportError as e:
    print("PyQt6 is not installed. Please install it using in the command line:")
    print("pip install pyqt6")
    sys.exit(1)
    
from mainwindow import MainWindow

if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())