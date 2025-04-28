from PyQt6 import QtWidgets, uic, QtCore, QtGui
import psutil
import GPUtil
from src.messagebox import CustomMessageBox
from src.uianimations import UIAnimations
from src.windowcontrol import WindowControl
from src.pagenavigationhandler import PageNavigationHandler
from src.reordertable import ReorderTableView, ReorderTableModel
from src.modeltraininghandler import ModelTrainingHandler



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


        self.minimize_button.clicked.connect(self.showMinimized)
        self.maximize_button.clicked.connect(self.toggle_maximize_restore)
        self.close_button.clicked.connect(self.close)
        self.home_model_button.clicked.connect(self.page_navigation_handler.showTimelinePage)
        self.home_results_button.clicked.connect(self.page_navigation_handler.load_graphs_from_main_menu)
        self.save_process_button.clicked.connect(self.model_training_handler.saveModel)
        self.save_process_button.hide()
        self.load_process_button.clicked.connect(self.model_training_handler.loadModel)
        self.load_process_button.hide()
        self.ui_handler = UIAnimations()
        
        self.save_results_button.clicked.connect(self.page_navigation_handler.save_graphs)
        self.settings_button.clicked.connect(self.page_navigation_handler.showSettingsPage)

        self.timeline_start_training_button.clicked.connect(self.model_training_handler.parseThroughProcessesTable)
        self.save_settings_button.clicked.connect(self.page_navigation_handler.saveSettings)

        self.old_pos = self.pos()
        self.is_maximized = False
        

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_specs_usage)
        self.timer.start(1000)

        self.opacity_effect = QtWidgets.QGraphicsOpacityEffect(self.stackedWidget)
        self.stackedWidget.setGraphicsEffect(self.opacity_effect)
        self.fade_animation = QtCore.QPropertyAnimation(self.opacity_effect, b"opacity")
        self.fade_animation.setDuration(500)

        self.df = None
        self.df_last_file_path = ""
        self.current_page = "Home"
        self.onSettingsPage = False
        self.window_control = WindowControl(self)
        #self.home_button.clicked.connect(self.page_navigation_handler.showHomePage)



        self.data = [
            ["", "2", "MLP", "Prune", "MNIST", "[89, 44, 22, 11, 4, 80]", "CrossEntropy", "Adam", "1", "", "", 32, 0.001, "Full"],
            ["", "3", "MLP", "Prune", "MNIST", "[15,9,6,4,2,12]", "CrossEntropy", "Adam", "1", "", "", 32, 0.01, "Full"],
            ["", "3", "MLP", "Prune", "MNIST", "[11,3,6,4,2,8]", "CrossEntropy", "Adam", "1", "", "", 32, 0.01, "Full"],
            ["", "1", "MLP", "Prior", "MNIST", "48", "CrossEntropy", "Adam", "1", 2, 1.0, 64,0.001, "WS"],
            ["", "4", "MLP", "Prior", "MNIST", "70", "CrossEntropy", "Adam", "1", 2, 0.8, 64,0.01, "WS"],
            ["", "4", "MLP", "Prior", "MNIST", "100", "CrossEntropy", "Adam", "1", 2, 0.7, 64,0.01, "WS"],
            ["", "4", "MLP", "Prior", "MNIST", "250", "CrossEntropy", "Adam", "1", 2, 0.5, 64,0.01, "WS"],
        ]
        self.hiddendata = [
            [ ["","", "2", "Retrain", "-", "-", "-", "1", "0.001"], ["", "", "3", "Prune", "FULL", "10", "Magnitude", "-", "-"], ["", "", "4", "Retrain", "-", "-", "-", "1", "0.001"] ],
        ]
        self.page_navigation_handler.showHomePage() #this ensures to start at the home page
        self.showTableWidget()
        self.showTableWidget2()

        self.model_train_button.setEnabled(True)

        self.neural_networks = []
        self.previous_results = []
        self.previous_results = []

    def fadeInUp(self, widget):
        self.ui_handler.fadeInUp(widget) #this redirects the pagenavigationhandler.py to the animations.py

    def fadeToPage(self, new_page):
        self.stackedWidget.setCurrentWidget(new_page)
        self.ui_handler.fadeInUp(new_page)


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

    def showTableWidget(self):
        # sample data
        processed_data = [[False] + row for row in self.hiddendata]

        self.pruningTableModel = ReorderTableModel(processed_data, headers=["", "Step", "Action", "Scope", "Pruning %", "Method", "Epochs", "Learning Rate"], show_edit_column=False)


        self.reorder_table_view = ReorderTableView(self)
        self.reorder_table_view.setModel(self.pruningTableModel)
        self.reorder_table_view.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.DoubleClicked)


        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.reorder_table_view)

        header = self.reorder_table_view.horizontalHeader()
        header.setDefaultAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Interactive)
        header.setStyleSheet("QHeaderView::section {"
                            "   qproperty-alignment: AlignCenter;"
                            "   padding: 4px;"
                            "   font-size: 7pt;"
                            "   white-space: normal;"
                            "}")
        # Adjust font size
        font = header.font()
        font.setPointSize(8)  # Set to a smaller font size
        header.setFont(font)
        self.reorder_table_view.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.reorder_table_view.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.MultiSelection)

        self.reorder_table_view.verticalHeader().hide()
        self.reorder_table_view.resizeColumnsToContents()
        self.reorder_table_view.setColumnWidth(0, 20)
        if self.tableWidgetPruning.layout():
            QtWidgets.QWidget().setLayout(self.tableWidgetPruning.layout()) 


        header = self.reorder_table_view.horizontalHeader()
        for col in range(2, self.pruningTableModel.columnCount() - 2):
            header.setSectionResizeMode(col, QtWidgets.QHeaderView.ResizeMode.Stretch)

        self.reorder_table_view.verticalHeader().hide()
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.reorder_table_view)

        if self.tableWidgetPruning.layout():
            QtWidgets.QWidget().setLayout(self.tableWidgetPruning.layout()) 
        self.tableWidgetPruning.setLayout(layout)
        self.tableWidgetPruning.resizeColumnsToContents()

    def showTableWidget2(self):
        # sample data

        self.timelineTableModel = ReorderTableModel(self.data, headers=["", "", "Model\nType", "Start", "Dataset", "Hidden\nsizes", "Loss", "Optimizer", "Epochs", "k", "p", "Batch\nSize", "Learning\nRate", "Graph\nType"])

        self.reorder_table_view2 = ReorderTableView(self)
        self.reorder_table_view2.setModel(self.timelineTableModel)
        self.reorder_table_view2.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.DoubleClicked)

        for i in range(len(self.data)):
            for j in range(len(self.hiddendata)):
                self.timelineTableModel.set_hidden_data(i, self.hiddendata[j])

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.reorder_table_view2)


        header = self.reorder_table_view2.horizontalHeader()
        header.setDefaultAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Interactive)
        header.setStyleSheet("QHeaderView::section {"
                            "   qproperty-alignment: AlignCenter;"
                            "   padding: 4px;"
                            "   font-size: 7pt;"
                            "   white-space: normal;"
                            "}")

        font = header.font()
        font.setPointSize(8)
        header.setFont(font)
        self.reorder_table_view2.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.reorder_table_view2.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.MultiSelection)

        if self.tableWidgetPruning_2.layout():
            QtWidgets.QWidget().setLayout(self.tableWidgetPruning_2.layout()) 

        self.reorder_table_view2.resizeColumnsToContents()

        self.reorder_table_view2.setColumnWidth(0,20)
        self.reorder_table_view2.setColumnWidth(1,1)
        self.reorder_table_view2.setColumnWidth(6,80)
        self.reorder_table_view2.setColumnWidth(7,80)
        self.reorder_table_view2.setColumnWidth(8,60)
        self.reorder_table_view2.setColumnWidth(11,30)
        self.reorder_table_view2.setColumnWidth(12,30)

        header = self.reorder_table_view2.horizontalHeader()
        for col in range(2, self.timelineTableModel.columnCount() - 2):
            header.setSectionResizeMode(col, QtWidgets.QHeaderView.ResizeMode.Stretch)

        self.reorder_table_view2.verticalHeader().hide()

        self.tableWidgetPruning_2.setLayout(layout)
        self.tableWidgetPruning_2.resizeColumnsToContents()
        self.multiply_rows_timeline_button.clicked.connect(lambda: self.multiply_rows_timeline(self.timelineTableModel))
        self.multiply_rows_pruning_button.clicked.connect(lambda: self.multiply_rows_timeline(self.pruningTableModel))

        self.reorder_table_view2.rowEdited.connect(lambda row: self.handle_row_edit(row))

    def multiply_rows_timeline(self, model):
        popup = QtWidgets.QDialog(self)
        popup.setWindowFlags(QtCore.Qt.WindowType.FramelessWindowHint)
        popup.setMinimumSize(300, 150)
        popup.setStyleSheet("""
            QDialog {
                background-color: #121212;
                border: 1px solid #333333;
                border-radius: 10px;
            }
            QLabel {
                color: #ffffff;
                font-size: 14px;
            }
            QSpinBox {
                background-color: #1e1e1e;
                color: #ffffff;           
                border: 1px solid #333333;
                border-radius: 5px;
                padding: 5px;
            }
            QPushButton {
                background-color: #1e1e1e;
                color: #ffffff;           
                border: 1px solid #333333;
                border-radius: 5px;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #333333;
            }
        """)

        layout = QtWidgets.QVBoxLayout(popup)
        label = QtWidgets.QLabel("How many times would you like to multiply the selected rows?")
        layout.addWidget(label)

        spin_box = QtWidgets.QSpinBox()
        spin_box.setRange(1, 100)
        spin_box.setValue(1)
        layout.addWidget(spin_box)

        button_layout = QtWidgets.QHBoxLayout()
        ok_button = QtWidgets.QPushButton("OK")
        cancel_button = QtWidgets.QPushButton("Cancel")
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

        ok_button.clicked.connect(popup.accept)
        cancel_button.clicked.connect(popup.reject)

        window_rect = self.geometry()
        dialog_rect = popup.geometry()
        x = window_rect.center().x() - dialog_rect.center().x()
        y = window_rect.center().y() - dialog_rect.center().y()
        popup.move(x, y)

        # Show the popup and handle the result
        if popup.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            count = spin_box.value()
            model.multiply_selected_items(count + 1)
                
    def handle_row_edit(self, row):
        data = self.timelineTableModel.get_hidden_data(row)

        if len(data) <= 1:
            data = [
                [ ["","", "2", "Retrain", "-", "-", "-", "1", "0.001"], ["", "", "3", "Prune", "FULL", "10", "Magnitude", "-", "-"], ["", "", "4", "Retrain", "-", "-", "-", "1", "0.001"] ],
                ]
        self.pruningTableModel.beginResetModel()
        self.pruningTableModel._data = []

        for hidden_row in data:
            new_row = [""] * self.pruningTableModel.columnCount()

            for j in range(min(len(hidden_row), self.pruningTableModel.columnCount())):
                new_row[j] = hidden_row[j]
            self.pruningTableModel._data.append(new_row)

        self.pruningTableModel.endResetModel()
        self.page_navigation_handler.showModelPage()
        self.page_navigation_handler.showModelPruningTablePage()


    def overwrite_table_data(self, table, data):
        table.beginResetModel()
        table._data = []

        for row in data:
            new_row = [""] * table.columnCount()
            for j in range(min(len(row), table.columnCount())):
                new_row[j] = row[j]
            table._data.append(row)
        table.endResetModel()


    def complete_reset(self):
        self.loading_label.hide()
        self.page_navigation_handler.showHomePage()
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
        self.model_training_handler.updateTrainingProcessLabel("")
        self.training_process_label.setText("")
        #reset selections of the tables
        self.reorder_table_view.clear_selection()
        self.reorder_table_view2.clear_selection()

        self.reset_timeline_table()
        self.saved_label.setText("")


    def show_warning(self, title="Warning", message= "Warning!\nAre you sure you want to restart? Your progress will be lost.", actions=None, buttons=["discard, cancel"]):
        if not isinstance(title, str):
            title = str(title)

        #TODO generalize this function to be used in other places as well
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
