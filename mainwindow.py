from PyQt6 import QtWidgets, uic, QtCore
import psutil
import GPUtil
from messagebox import CustomMessageBox
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

        self.opacity_effect = QtWidgets.QGraphicsOpacityEffect(self.stackedWidget)
        self.stackedWidget.setGraphicsEffect(self.opacity_effect)
        self.fade_animation = QtCore.QPropertyAnimation(self.opacity_effect, b"opacity")
        self.fade_animation.setDuration(500)


        self.showHomePage() #this ensures to start at the home page



    def fadeToPage(self, new_page):
            """ Fade animation when switching pages """
            self.fade_animation.stop()
            self.fade_animation.setStartValue(0.0)
            self.fade_animation.setEndValue(1.0)
            self.fade_animation.start()
            self.stackedWidget.setCurrentWidget(new_page)

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

    # buttons functions

    def showHomePage(self):
        self.fadeToPage(self.home_page)

    def showDatasetPage(self):
        self.fadeToPage(self.dataset_page)

    def showModelPage(self):
        self.fadeToPage(self.model_page)

    def showChooseResultsPage(self):
        self.fadeToPage(self.choose_results_page)

        
    def show_warning(self):
        # Define the actions to be taken when the buttons are clicked
        def discard_action():
            print("User discarded!")
            self.showHomePage()

        def cancel_action():
            print("User canceled.")

        # Create the message box
        msg = CustomMessageBox(
            "Restart Action",
            "Warning!\nAre you sure you want to restart? Your progress will be lost.",
            [
                ("Discard", QtWidgets.QMessageBox.ButtonRole.AcceptRole, discard_action),
                ("Cancel", QtWidgets.QMessageBox.ButtonRole.RejectRole, cancel_action)
            ],
            self
        )
        # Run the message box
        msg.exec()
