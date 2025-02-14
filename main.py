from PyQt6 import QtWidgets, uic
'''

def function_button():
    print('Button clicked!')



if __name__ == '__main__':
    import sys

    app = QtWidgets.QApplication(sys.argv)
    window = QtWidgets.QMainWindow()
    uic.loadUi('file.ui', window)
    window.pushButton.clicked.connect(function_button)
    window.setWindowTitle('Hello World')
    window.show()
    sys.exit(app.exec())'''


from mainwindow import MainWindow

if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())