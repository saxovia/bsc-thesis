try:
    from PyQt6 import QtWidgets, uic
    import psutil
    import GPUtil
except ImportError as e:
    # Install it for user
    import subprocess
    import sys
    import os

    def install(package):
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

    install("pyqt6")
    install("psutil")
    install("gputil")


    
from mainwindow import MainWindow

if __name__ == "__main__":
    import sys
    import os
    from PyQt6 import QtWidgets, uic, QtGui, QtCore

    font_path = "./fonts/DMSans-VariableFont_opsz.ttf"

    import os
    app = QtWidgets.QApplication(sys.argv)

    font_id = QtGui.QFontDatabase.addApplicationFont(font_path)
    if font_id == -1:
        print("Error: Font not loaded!")
        sys.exit(1)

    font_family = QtGui.QFontDatabase.applicationFontFamilies(font_id)[0]
    custom_font = QtGui.QFont(font_family, 12)

    app.setFont(custom_font)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())