from PyQt6 import QtWidgets, uic, QtCore, QtGui
import psutil
import GPUtil
from src.messagebox import CustomMessageBox
from src.uianimations import UIAnimations
from src.windowcontrol import WindowControl
from src.pagenavigationhandler import PageNavigationHandler
from src.reordertable import ReorderTableView, ReorderTableModel
from src.modeltraininghandler import ModelTrainingHandler
from src.resultshandler import ResultsHandler
from src.tablehandler import TableHandler



class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Sparse Neural Network Generator')
        uic.loadUi('mainwindowui.ui', self)
        self.restart_button.clicked.connect(lambda: self.show_warning(title="Warning", message="Warning!\nAre you sure you want to restart? Your progress will be lost.", actions=None, buttons=["discard", "cancel"]))
        self.setWindowFlag(QtCore.Qt.WindowType.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        self.maximize_button.setCheckable(True)


        self.page_navigation_handler = PageNavigationHandler(self)
        self.model_training_handler = ModelTrainingHandler(self)
        self.results_handler = ResultsHandler(self)
        self.table_handler = TableHandler(self)

        self.minimize_button.clicked.connect(self.showMinimized)
        self.maximize_button.clicked.connect(self.toggle_maximize_restore)
        self.close_button.clicked.connect(self.close)
        self.home_model_button.clicked.connect(self.page_navigation_handler.show_timeline_page)
        self.home_results_button.clicked.connect(self.page_navigation_handler.load_graphs_from_main_menu)
        self.save_process_button.hide()
        self.load_process_button.hide()
        self.ui_handler = UIAnimations()
        
        self.save_results_button.clicked.connect(self.page_navigation_handler.save_graphs)
        self.settings_button.clicked.connect(self.page_navigation_handler.show_settings_page)

        self.timeline_start_training_button.clicked.connect(self.model_training_handler.parse_through_processes_table)
        self.save_settings_button.clicked.connect(self.page_navigation_handler.save_settings)

        self.old_pos = self.pos()
        self.is_maximized = False
        

        #self.timer = QtCore.QTimer(self)
        #self.timer.timeout.connect(self.update_specs_usage)
        #self.timer.start(1000)

        self.opacity_effect = QtWidgets.QGraphicsOpacityEffect(self.stackedWidget)
        self.stackedWidget.setGraphicsEffect(self.opacity_effect)
        self.fade_animation = QtCore.QPropertyAnimation(self.opacity_effect, b"opacity")
        self.fade_animation.setDuration(500)

        self.df = None
        self.df_last_file_path = ""
        self.current_page = "Home"
        self.onSettingsPage = False
        self.window_control = WindowControl(self)
        #self.home_button.clicked.connect(self.page_navigation_handler.show_home_page)

        self.page_navigation_handler.show_home_page() #this ensures to start at the home page
        self.table_handler.show_table_widget()
        self.table_handler.show_table_widget2()

        self.model_train_button.setEnabled(True)

        self.neural_networks = []
        self.previous_results = []
        self.previous_results = []

        

    def fade_in_up(self, widget):
        self.ui_handler.fade_in_up(widget) #this redirects the pagenavigationhandler.py to the animations.py

    def fade_to_page(self, new_page):
        self.stackedWidget.setCurrentWidget(new_page)
        self.ui_handler.fade_in_up(new_page)


    def toggle_maximize_restore(self):
        if self.is_maximized:
            self.showNormal()
        else:
            self.showMaximized()
        self.is_maximized = not self.is_maximized

    def mousePressEvent(self, event):
        self.window_control.mousePressEvent(event)

    def mouseMoveEvent(self, event):
        self.window_control.mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self.window_control.mouseReleaseEvent(event)

    def type_text_effect(self, label, text, interval=50):
        self.typing_timer = QtCore.QTimer(self)
        self.typing_text = text
        self.typing_index = 0
        self.label_to_update = label

        self.typing_timer.timeout.connect(self.update_typing)
        self.typing_timer.start(interval)

    def update_typing(self):
        if self.typing_index < len(self.typing_text):
            self.label_to_update.setText(self.typing_text[:self.typing_index + 1])
            self.typing_index += 1
        else:
            self.typing_timer.stop()


    # for footer
    def update_specs_usage(self):
        memory = psutil.virtual_memory()
        ram_usage = memory.used / (1024 ** 3)
        ram_total = memory.total / (1024 ** 3)

        gpus = GPUtil.getGPUs()
        gpu_usage = gpus[0].load * 100 if gpus else "N/A"
        gpu_usage_string = f"{gpu_usage:.2f}%" if gpus else "N/A"
        statusbartext = f"RAM: {ram_usage:.2f} GB / {ram_total:.2f} GB | CPU: {psutil.cpu_percent()}% | GPU Usage: {gpu_usage_string}"
        self.statusbar.showMessage(statusbartext)


    def complete_reset(self):
        self.loading_label.hide()
        self.page_navigation_handler.show_home_page()
        self.model_train_button.setEnabled(True)
        self.undo_button.setEnabled(True)
        self.model_train_button.setText("Train Model")
        self.model_train_button.disconnect()
        self.model_train_button.setEnabled(True)
        self.save_results_button.show()
        if self.model_training_handler.trainer is not None:
            self.model_training_handler.trainer.running = False
            self.model_training_handler.trainer.terminate()
            self.model_training_handler.trainer.wait()
            self.model_training_handler.trainer = None
        self.neural_networks = []
        #self.listWidget.clearSelection()
        #self.listWidget_2.clearSelection()
        self.model_training_handler.update_training_process_label("")
        self.training_process_label.setText("")
        #reset selections of the tables
        self.reorder_table_view.clear_selection()
        self.reorder_table_view2.clear_selection()

        self.table_handler.reset_timeline_table()
        self.saved_label.setText("")


    def show_warning(self, title="Warning", message= "Warning!\nAre you sure you want to restart? Your progress will be lost.", actions=None, buttons=["discard, cancel"]):
        if not isinstance(title, str):
            title = str(title)

        def discard_action():
            self.complete_reset()


        if actions is None:
            actions = [
                ("OK", QtWidgets.QMessageBox.ButtonRole.AcceptRole, lambda: print("OK clicked")),
                ("Cancel", QtWidgets.QMessageBox.ButtonRole.RejectRole, lambda: print("Cancel clicked"))
            ]

        def cancel_action():
            pass

        def save_action():
            #self.page_navigation_handler.save_graphs()
            pass

        buttonaccept = None
        buttonreject = None

        for row in buttons:
            if row == 'discard':
                buttonaccept = ("Discard", QtWidgets.QMessageBox.ButtonRole.AcceptRole, discard_action)
            elif row == "cancel":
                buttonreject = ("Cancel", QtWidgets.QMessageBox.ButtonRole.RejectRole, cancel_action)
            elif row == 'save':
                buttonaccept = ("Save", QtWidgets.QMessageBox.ButtonRole.AcceptRole, save_action)

        if buttonaccept is None:
            buttonaccept = ("OK", QtWidgets.QMessageBox.ButtonRole.AcceptRole, lambda: None)
        if buttonreject is None:
            buttonreject = ("Cancel", QtWidgets.QMessageBox.ButtonRole.RejectRole, lambda: None)

        msg = CustomMessageBox(
            title,
            message,
            [
            buttonaccept,
            buttonreject
            ],
            self
        )
        msg.exec()
        self.setGraphicsEffect(None)
