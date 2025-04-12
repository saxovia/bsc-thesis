from PyQt6 import QtWidgets, uic, QtCore, QtGui
import psutil
import GPUtil
from src.messagebox import CustomMessageBox
import sys
import pandas as pd
import numpy as np
from src.uianimations import UIAnimations
from src.windowcontrol import WindowControl
from src.pagenavigationhandler import PageNavigationHandler
from src.reordertable import ReorderTableView, ReorderTableModel
from src.modeltraininghandler import ModelTrainingHandler

#from temp.csvhandler import CSVHandler


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Sparse Neural Network Generator')
        uic.loadUi('mainwindowui.ui', self)
        self.restart_button.clicked.connect(self.show_warning)
        self.setWindowFlag(QtCore.Qt.WindowType.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)

        self.GLOBAL_CHOSEN_MODEL = None
        self.GLOBAL_CHOSEN_START = None
        self.GLOBAL_STAGE = 1
        #for widget in self.findChildren(QtWidgets.QPushButton):
        #    widget.installEventFilter(self)

        self.maximize_button.setCheckable(True)


        self.page_navigation_handler = PageNavigationHandler(self)
        self.model_training_handler = ModelTrainingHandler(self)


        self.minimize_button.clicked.connect(self.showMinimized)
        self.maximize_button.clicked.connect(self.toggle_maximize_restore)
        self.close_button.clicked.connect(self.close)
        #self.home_dataset_button.clicked.connect(self.page_navigation_handler.showTimelinePage)
        self.home_model_button.clicked.connect(self.page_navigation_handler.showTimelinePage)
        self.home_results_button.clicked.connect(self.page_navigation_handler.showChooseResultsPage)
        #self.dataset_choose_model_button.clicked.connect(self.page_navigation_handler.showModelPage)
        self.model_train_button.clicked.connect(self.page_navigation_handler.showModelStartingPage)
        self.save_process_button.clicked.connect(self.model_training_handler.saveModel)
        self.load_process_button.clicked.connect(self.model_training_handler.loadModel)
        # Other buttons
        self.ui_handler = UIAnimations()
        self.settings_button.clicked.connect(self.page_navigation_handler.showSettingsPage)

        self.timeline_start_training_button.clicked.connect(self.model_training_handler.parseThroughProcessesTable)

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

        self.page_navigation_handler.showHomePage() #this ensures to start at the home page
        self.page_navigation_handler.showModelStartingPage()
        self.showTableWidget()
        self.showTableWidget2()

        self.model_train_button.setEnabled(False)
        self.listWidget.itemSelectionChanged.connect(lambda: self.on_item_selected(self.listWidget, "GLOBAL_CHOSEN_MODEL"))
        self.listWidget_2.itemSelectionChanged.connect(lambda: self.on_item_selected(self.listWidget_2, "GLOBAL_CHOSEN_START")) #unused

        #use array instead for all this input
        self.neural_networks = []

    def update_variable(self, input_widget, var_value): #TODO rewrite this to be more generic and not throw exceptions when debugging
        if isinstance(input_widget, QtWidgets.QLineEdit):
            text_value = input_widget.text()
        elif isinstance(input_widget, QtWidgets.QComboBox):
            text_value = input_widget.currentText()
        else:
            print("Unsupported widget")
            return

        current_value = getattr(self, var_value, None)

        if ',' in text_value:
            node_values = [int(value.strip()) for value in text_value.split(',') if value.strip()]
            setattr(self, var_value, node_values)
            return

        if isinstance(current_value, int):
            try:
                setattr(self, var_value, int(text_value))
            except ValueError:
                setattr(self, var_value, 0)
        elif isinstance(current_value, float):
            try:
                setattr(self, var_value, float(text_value))
            except ValueError:
                setattr(self, var_value, 0.0)
        else:
            setattr(self, var_value, text_value)




    def eventFilter(self, obj, event):
        if isinstance(obj, QtWidgets.QPushButton):
            if event.type() == QtCore.QEvent.Type.Enter:
                #self.highlight_button(obj, True)  # Hover In
                pass
            elif event.type() == QtCore.QEvent.Type.Leave:
                ##self.highlight_button(obj, False)  # Hover Out
                pass
            elif event.type() == QtCore.QEvent.Type.MouseButtonPress:
                pass
                #self.flash_color(obj)  # Click effect
        return super().eventFilter(obj, event)


    def toggle_stackedWidget2_page(self):
        """.
        current_index = self.stackedWidget_2.currentIndex()
        next_index = 1 if current_index == 0 else 0 
        self.stackedWidget2.setCurrentIndex(next_index)"""

    def on_item_selected(self, list_widget, global_var_name):
        selected_items = list_widget.selectedItems()
        if selected_items:
            selected_value = selected_items[0].text()
            setattr(self, global_var_name, selected_value)
        
            print(f"Updated {global_var_name}: {selected_value}")

        self.page_navigation_handler.update_button_state()
        print(self.GLOBAL_CHOSEN_MODEL, self.GLOBAL_CHOSEN_START, self.GLOBAL_STAGE)


    def applyChangesToDataset(self):
        if self.encoding_input.toPlainText() != "":
            text = self.encoding_input.toPlainText()
            try:
                exec_env = {"df": self.df}
                exec(text, exec_env)

                # if there are modifications:
                if "df" in exec_env:
                    self.df = exec_env["df"]
                self.dataset_error_message_label.setText("Code executed successfully.")
            except Exception as e:
                self.dataset_error_message_label.setText(f"Error: {str(e)}")

    def viewDataset(self):
        data = self.df.head(5)
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Dataset Header")

        table = QtWidgets.QTableWidget(dialog)
        table.setRowCount(data.shape[0])
        table.setColumnCount(data.shape[1])
        table.setHorizontalHeaderLabels(data.columns)

        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                item = QtWidgets.QTableWidgetItem(str(data.iloc[row, col]))
                table.setItem(row, col, item)


        table.resizeColumnsToContents()
        table.resizeRowsToContents()
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(table)
        dialog.setLayout(layout)

        dialog.exec()

    def fadeInUp(self, widget):
        self.ui_handler.fadeInUp(widget) #this redirects the pagenavigationhandler.py to the animations.py

    def fadeToPage(self, new_page):
        self.stackedWidget.setCurrentWidget(new_page)
        self.ui_handler.fadeInUp(new_page)

    def toggle_maximize_restore(self):
        if self.isFullScreen():
            self.showNormal()
            self.is_maximized = False
        else:
            self.showFullScreen()
            self.is_maximized = True

    # Dragging functions
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


    def showTableWidget2(self):
        # sample data
        data = [
            ["", "2", "MLP", "Prune", "MNIST", "[89, 44, 22, 11, 4, 80]", "CrossEntropy", "Adam", "1", "", "", 32, 0.001, "Full"],
            ["", "3", "LSTM", "Prune", "MNIST", "[15,9,6,4,2,12]", "CrossEntropy", "Adam", "1", "", "", 32, 0.01, "Full"],
            ["", "1", "LSTM", "Prior", "MNIST", "48", "CrossEntropy", "Adam", "30", 2, 1.0, 64,0.001, "WS"],
            ["", "4", "MLP", "Prior", "MNIST", "250", "CrossEntropy", "Adam", "30", 2, 0.7, 64,0.01, "WS"],
        ]

        self.timelineTableModel = ReorderTableModel(data, headers=["", "", "Model\nType", "Start", "Dataset", "Hidden\nsizes", "Loss", "Optimizer", "Epochs", "k", "p", "Batch\nSize", "Learning\nRate", "Graph\nType"])

        self.reorder_table_view2 = ReorderTableView(self)
        self.reorder_table_view2.setModel(self.timelineTableModel)
        self.reorder_table_view2.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.DoubleClicked)
        #default values for hidden data:
        # For each pruned data, set hidden data to be the same as the proceeding:
        for i in range(0, len(data)):
            self.timelineTableModel.set_hidden_data(i, [["1", "Prune", "Global", "FULL", "50", "Magnitude", "-", "-"], ["2", "Retrain", "-", "-", "-", "-", "1", "0.001"], ["3", "Prune", "No", "IH", "10", "Magnitude", "-", "-"]])
        #self.timelineTableModel.set_hidden_data(0, [["1", "Prune", "Global", "FULL", "50", "Magnitude", "-", "-"], ["2", "Retrain", "-", "-", "-", "-", "1", "0.001"], ["3", "Prune", "No", "IH", "10", "Magnitude", "-", "-"]])
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

        if self.tableWidgetPruning_2.layout(): #TODO rename this. rename dataset pages
            QtWidgets.QWidget().setLayout(self.tableWidgetPruning_2.layout()) 

        self.reorder_table_view2.resizeColumnsToContents()
        self.reorder_table_view2.setColumnWidth(0, 20)

        self.reorder_table_view2.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Fixed)
        self.reorder_table_view2.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Fixed)
        self.reorder_table_view2.horizontalHeader().setSectionResizeMode(self.timelineTableModel.columnCount() - 2, QtWidgets.QHeaderView.ResizeMode.Fixed)
        self.reorder_table_view2.horizontalHeader().setSectionResizeMode(self.timelineTableModel.columnCount() - 1, QtWidgets.QHeaderView.ResizeMode.Fixed)
        self.reorder_table_view2.setColumnWidth(self.timelineTableModel.columnCount() - 2, 10)
        self.reorder_table_view2.setColumnWidth(self.timelineTableModel.columnCount() - 1, 10)  # Delete col
        self.reorder_table_view2.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Fixed)
        self.reorder_table_view2.horizontalHeader().setSectionResizeMode(self.timelineTableModel.columnCount() - 2, QtWidgets.QHeaderView.ResizeMode.Fixed)
        self.reorder_table_view2.horizontalHeader().setSectionResizeMode(self.timelineTableModel.columnCount() - 1, QtWidgets.QHeaderView.ResizeMode.Fixed)

        for col in range(2, self.timelineTableModel.columnCount() - 2):
            self.reorder_table_view2.resizeColumnToContents(col)

        self.reorder_table_view2.verticalHeader().hide()
        self.reorder_table_view2.setColumnWidth(1, 0)
        self.reorder_table_view2.setColumnWidth(6, 80)
        self.reorder_table_view2.setColumnWidth(7, 80)
        
        self.reorder_table_view2.setColumnWidth(9, 40)
        self.reorder_table_view2.setColumnWidth(8, 60)
        self.reorder_table_view2.setColumnWidth(10, 30)
        self.reorder_table_view2.setColumnWidth(11, 30)
        self.reorder_table_view2.setColumnWidth(12, 30)
        self.tableWidgetPruning_2.setLayout(layout)
        self.tableWidgetPruning_2.resizeColumnsToContents()
        self.multiply_rows_timeline_button.clicked.connect(lambda: self.multiply_rows_timeline(self.timelineTableModel))

        

        self.delete_rows_timeline_button.clicked.connect(self.timelineTableModel.remove_selected_items)
        self.reorder_table_view2.rowEdited.connect(lambda row: self.handle_row_edit(row))

    def multiply_rows_timeline(self, model):
        count, response = QtWidgets.QInputDialog.getInt(
            self, "Multiply Items", "How many copies?", 2, 1, 100, 1
        )
        if response:
            model.multiply_selected_items(count)
        
            
    def handle_row_edit(self, row):
        print("Row edited:", row)
        # Transition pages
        # Get row data and fill in the table fields (this should be hidden data for each row)

        data = self.timelineTableModel.get_hidden_data(row)
        # Example data = ["1", "Prune", "Global", "FULL", "50", "Magnitude", "-", "-"]

        # Ensure pruningTableModel has enough rows
        required_rows = len(data)
        current_rows = self.pruningTableModel.rowCount()
        if current_rows < required_rows:
            for _ in range(required_rows - current_rows):
                self.pruningTableModel.beginInsertRows(QtCore.QModelIndex(), current_rows, current_rows)
                self.pruningTableModel._data.append([""] * self.pruningTableModel.columnCount())
                self.pruningTableModel.endInsertRows()
                current_rows += 1

        for i in range(len(data)):
            for j in range(len(data[i])):
                index = self.pruningTableModel.index(i, j, QtCore.QModelIndex())
                self.pruningTableModel.setData(index, data[i][j], QtCore.Qt.ItemDataRole.EditRole)

        self.page_navigation_handler.showModelPage()
        self.page_navigation_handler.showModelPruningTablePage()
        # Iterate and edit them to be the same

    def showTableWidget(self):
        # sample data
        data = [
            ["1", "Prune", "Global", "FULL", "50", "Magnitude", "-", "-"],
            ["2", "Retrain", "-", "-", "-", "-", "10", "0.001"],
        ]

        self.pruningTableModel = ReorderTableModel(data, headers=["", "Step", "Action", "Scope", "Layer(s) Affected", "Pruning %", "Method", "Epochs", "Learning Rate"], show_edit_column=False)

        self.reorder_table_view = ReorderTableView(self)
        self.reorder_table_view.setModel(self.pruningTableModel)
        self.reorder_table_view.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.DoubleClicked)


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
        self.reorder_table_view.verticalHeader().hide()
        self.reorder_table_view.resizeColumnsToContents()
        self.reorder_table_view.setColumnWidth(0, 20)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.reorder_table_view)

        if self.tableWidgetPruning.layout():
            QtWidgets.QWidget().setLayout(self.tableWidgetPruning.layout()) 
        self.tableWidgetPruning.setLayout(layout)
        self.tableWidgetPruning.resizeColumnsToContents()

    def show_warning(self, title="Warning", message="Are you sure you want to proceed?", actions=None):
        def discard_action():
            
            self.loading_label.hide()
            self.page_navigation_handler.showHomePage()
            self.model_train_button.setEnabled(True)
            self.undo_button.setEnabled(True)
            self.GLOBAL_CHOSEN_MODEL = None
            self.GLOBAL_CHOSEN_START = None
            self.GLOBAL_STAGE = 1
            self.model_train_button.setText("Train Model")
            self.model_train_button.disconnect()
            self.model_train_button.clicked.connect(self.page_navigation_handler.showModelStartingPage)
            self.model_train_button.setEnabled(True)
            if self.model_training_handler.trainer is not None:
                self.model_training_handler.trainer.running = False
                self.model_training_handler.trainer.quit()
                self.model_training_handler.trainer.wait()
                self.model_training_handler.trainer = None
            self.neural_networks = []
            self.page_navigation_handler.showModelStartingPage()
            self.listWidget.clearSelection()
            self.listWidget_2.clearSelection()
            self.model_training_handler.updateTrainingProcessLabel("")
            self.training_process_label.setText("")


        if actions is None:
            # Default actions if none are provided
            actions = [
                ("OK", QtWidgets.QMessageBox.ButtonRole.AcceptRole, lambda: print("OK clicked")),
                ("Cancel", QtWidgets.QMessageBox.ButtonRole.RejectRole, lambda: print("Cancel clicked"))
            ]
        def cancel_action():
            print("User canceled.")

        msg = CustomMessageBox(
            "Restart Action",
            "Warning!\nAre you sure you want to restart? Your progress will be lost.",
            [
            ("Discard", QtWidgets.QMessageBox.ButtonRole.AcceptRole, discard_action),
            ("Cancel", QtWidgets.QMessageBox.ButtonRole.RejectRole, cancel_action)
            ],
            self
        )
        msg.exec()
        self.setGraphicsEffect(None)
