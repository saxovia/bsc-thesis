
try:
    from PyQt6 import QtWidgets, uic
    import psutil
    import GPUtil
    import pandas
    import numpy

except ImportError as e:
    # Run libsinstaller.py
    import sys, os, subprocess
    subprocess.check_call([sys.executable, "libsinstaller.py"])
    #sys.exit(1)
    
from src.mainwindow import MainWindow

if __name__ == "__main__":
    import sys
    import os
    from PyQt6 import QtWidgets, uic, QtGui, QtCore



    font_path = "./resources/fonts/DMSans-VariableFont_opsz.ttf"

    import os
    app = QtWidgets.QApplication(sys.argv)
    

    font_id = QtGui.QFontDatabase.addApplicationFont(font_path)
    if font_id == -1:
        print("Error: Font not loaded!")
        sys.exit(1)

    font_family = QtGui.QFontDatabase.applicationFontFamilies(font_id)[0]
    custom_font = QtGui.QFont(font_family, 12)
    app.setWindowIcon(QtGui.QIcon("icon.ico"))

    app.setFont(custom_font)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())