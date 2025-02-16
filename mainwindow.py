from PyQt6 import QtWidgets, uic, QtCore
import psutil
import GPUtil
from messagebox import CustomMessageBox

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Hello World')
        uic.loadUi('file.ui', self)
        self.restart_button.clicked.connect(self.show_warning)
        self.setWindowFlag(QtCore.Qt.WindowType.FramelessWindowHint)

        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        self.maximize_button.setCheckable(True)
        if self.statusBar():
            self.statusBar().setSizeGripEnabled(False)


        self.minimize_button.clicked.connect(self.showMinimized)
        self.maximize_button.clicked.connect(self.toggle_maximize_restore)
        self.close_button.clicked.connect(self.close)
        self.home_dataset_button.clicked.connect(self.showDatasetPage)
        self.home_model_button.clicked.connect(self.showModelPage)
        self.home_results_button.clicked.connect(self.showChooseResultsPage)


        self.old_pos = self.pos()
        self.mousePressed = False
        self.is_maximized = False
        

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_ram_usage)
        self.timer.start(1000)
        self.showHomePage()

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

    def update_ram_usage(self):
        memory = psutil.virtual_memory()
        ram_usage = memory.used / (1024 ** 3)
        ram_total = memory.total / (1024 ** 3)

        gpus = GPUtil.getGPUs()
        gpu_usage = gpus[0].load * 100 if gpus else 0
        self.footer_label.setText(f"RAM: {ram_usage:.2f} GB / {ram_total:.2f} GB | CPU: {psutil.cpu_percent()}% | GPU Usage: {gpu_usage}%")

    def showHomePage(self):
        self.stackedWidget.setCurrentWidget(self.home_page)
    def showDatasetPage(self):
        self.stackedWidget.setCurrentWidget(self.dataset_page)

    def showModelPage(self):
        self.stackedWidget.setCurrentWidget(self.model_page)

    def showChooseResultsPage(self):
        self.stackedWidget.setCurrentWidget(self.choose_results_page)

    def show_warning(self):
        #CustomMessageBox.show_custom_warning()
        msg = QtWidgets.QMessageBox(self)
        #msg.setIcon(QtWidgets.QMessageBox.Icon.Question)
        msg.setWindowTitle("Restart Action")
        msg.setText("Warning!\nAre you sure you want to restart? Your progress will be lost.")
        #msg.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No)
        #msg.button(QtWidgets.QMessageBox.StandardButton.Yes).setText("discard")
        msg.setWindowFlags(QtCore.Qt.WindowType.FramelessWindowHint | QtCore.Qt.WindowType.Dialog)
        discard_button = msg.addButton("Discard", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
        cancel_button = msg.addButton("Cancel", QtWidgets.QMessageBox.ButtonRole.RejectRole)

        discard_button.setObjectName("discard_button")
        cancel_button.setObjectName("cancel_button")
        msg.setStyleSheet("""
            QMessageBox QPushButton#discard_button:hover {
                background-color: #c42b1c;
            }
        """)
        result = msg.exec()
        if msg.clickedButton() == discard_button:
            print("User discarded!")
            self.showHomePage()
