from PyQt6 import QtWidgets, uic, QtCore, QtGui
import psutil
import GPUtil
from src.messagebox import CustomMessageBox
import sys
import pandas as pd
import numpy as np
from src.uianimations import UIAnimations
from src.windowcontrol import WindowControl
from src.pagenavigator import PageNavigator
from src.templatetable import ReorderTableView, ReorderTableModel

#from temp.csvhandler import CSVHandler


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Sparse Neural Network Generator')
        uic.loadUi('mainwindowui.ui', self)
        self.restart_button.clicked.connect(self.show_warning)
        self.setWindowFlag(QtCore.Qt.WindowType.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)

        self.GLOBAL_CHOSEN_MODEL = None #TODO remove these and make them more generic to self.trainer = []. its not global but specific to a trainer model
        self.GLOBAL_CHOSEN_START = None
        self.GLOBAL_STAGE = 1
        for widget in self.findChildren(QtWidgets.QPushButton):
            widget.installEventFilter(self)

        """
        self.setCentralWidget(self.centralwidget)

        self.shadow = QtWidgets.QGraphicsDropShadowEffect(self)
        self.shadow.setBlurRadius(20)
        self.shadow.setXOffset(0)
        self.shadow.setYOffset(0)
        self.shadow.setColor(QtGui.QColor(0, 0, 0, 150))

        self.main_frame = QtWidgets.QFrame(self.centralwidget)
        self.main_frame.setStyleSheet("background-color: white; border-radius: 10px;")
        self.main_frame.setGraphicsEffect(self.shadow)
        self.layout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.layout.setContentsMargins(10, 10, 10, 10)  # Ensure space for the shadow
        self.layout.addWidget(self.main_frame)
        """
        #self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        self.maximize_button.setCheckable(True)
        #if self.statusBar():
        #    self.statusBar().setSizeGripEnabled(False)


        self.navigator = PageNavigator(self)
        self.minimize_button.clicked.connect(self.showMinimized)
        self.maximize_button.clicked.connect(self.toggle_maximize_restore)
        self.close_button.clicked.connect(self.close)
        #self.home_dataset_button.clicked.connect(self.navigator.showDatasetPage)
        self.home_model_button.clicked.connect(self.navigator.showModelPage)
        self.home_results_button.clicked.connect(self.navigator.showChooseResultsPage)
        self.modify_dataset_button.clicked.connect(self.navigator.showDatasetPage)
        self.dataset_choose_model_button.clicked.connect(self.navigator.showModelPage)
        self.model_train_button.clicked.connect(self.navigator.showModelStartingPage)

        # Other buttons
        self.ui_handler = UIAnimations()
        #self.csv_handler = CSVHandler(self)
        #self.dataset_csv_button.clicked.connect(self.csv_handler.load_csv)
        #self.dataset_save_button.clicked.connect(self.csv_handler.save_csv)
        #self.dataset_apply_button.clicked.connect(self.applyChangesToDataset)
        #self.view_header_button.clicked.connect(self.viewDataset)
        self.settings_button.clicked.connect(self.navigator.showSettingsPage)

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
        #self.home_button.clicked.connect(self.navigator.showHomePage)

        self.navigator.showHomePage() #this ensures to start at the home page
        self.navigator.showModelStartingPage()
        self.showTableWidget()
        self.showTableWidget2()

        self.model_train_button.setEnabled(False)
        self.listWidget.itemSelectionChanged.connect(lambda: self.on_item_selected(self.listWidget, "GLOBAL_CHOSEN_MODEL"))
        self.listWidget_2.itemSelectionChanged.connect(lambda: self.on_item_selected(self.listWidget_2, "GLOBAL_CHOSEN_START"))

        #use array instead for all this input
        self.neural_networks = []
        self.neural_networks.append({ # example
            "number_of_nodes": [],
            "number_of_layers": 0,
            "activation_function": "",
            "optimizer": "",
            "loss_function": "",
            "epochs": 0,
            "k": 0,
            "p": 0.0,
            "learning_rate": 0.0,
            "batch_size": 0,
            "validation_split": 0
        })

        self.number_of_nodes = []
        self.number_of_layers = 0
        self.activation_function = "" #..?
        self.optimizer = ""
        self.loss_function = ""
        self.epochs = 0
        self.k = 0
        self.p = 0.0
        self.learning_rate = 0.0
        self.batch_size = 0 #TODO clean up default values assignment because ghjrgh It is all over the place

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
                self.highlight_button(obj, True)  # Hover In
            elif event.type() == QtCore.QEvent.Type.Leave:
                self.highlight_button(obj, False)  # Hover Out
            elif event.type() == QtCore.QEvent.Type.MouseButtonPress:
                self.flash_color(obj)  # Click effect
        return super().eventFilter(obj, event)

    def highlight_button(self, button, hover):
        #if hover:
        #    button.setStyleSheet("background-color: rgb(230, 230, 230); border: none;")
        #else:
        #    button.setStyleSheet("")  # Reset to default
        pass



    def flash_color(self, button):
        #previous_style = button.styleSheet()
        #button.setStyleSheet("background-color: rgb(200, 200, 200); border: none;")
        #QtCore.QTimer.singleShot(100, lambda: button.setStyleSheet("background-color: rgb(230, 230, 230); border: none;"))  

        #button.setStyleSheet(previous_style)
        pass

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

        self.navigator.update_button_state()
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
        self.ui_handler.fadeInUp(widget) #this redirects the pagenavigator.py to the animations.py

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
            ["", "1", "LSTM", "Prior", "MNIST", "250", "CrossEntropy", "Adam", "30", 2, 0.05, 64,0.001],
            ["", "2", "MLP", "Prune", "CIFAR-10", "6", "CrossEntropy", "Adam", "10", "", "", 32, 0.01],
        ]

        model = ReorderTableModel(data, headers=["", "Edit", "Model Type", "Start", "Dataset", "Hidden sizes", "Loss", "Optimizer", "Epochs", "k", "p", "Batch Size", "Learning Rate"])

        self.reorder_table_view = ReorderTableView(self)
        self.reorder_table_view.setModel(model)
        self.reorder_table_view.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.DoubleClicked)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.reorder_table_view)

        if self.tableWidgetPruning_2.layout(): #TODO rename this. rename dataset pages
            QtWidgets.QWidget().setLayout(self.tableWidgetPruning_2.layout()) 
        self.tableWidgetPruning_2.setLayout(layout)


    def showTableWidget(self):
        # sample data
        data = [
            ["1", "Prune", "Global", "FULL", "50", "Magnitude", "-", "-"],
            ["2", "Retrain", "-", "-", "-", "-", "10", "0.001"],
        ]

        model = ReorderTableModel(data, headers=["", "Step", "Action", "Scope", "Layer(s) Affected", "Pruning %", "Method", "Epochs", "Learning Rate"])

        self.reorder_table_view = ReorderTableView(self)
        self.reorder_table_view.setModel(model)
        self.reorder_table_view.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.DoubleClicked)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.reorder_table_view)

        if self.tableWidgetPruning.layout():
            QtWidgets.QWidget().setLayout(self.tableWidgetPruning.layout()) 
        self.tableWidgetPruning.setLayout(layout)

    #unused but will be used later for project loading, et.c
    def load_csv(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select CSV File", "", "CSV Files (*.csv)")
        
        if file_path:
            try:
                self.df = pd.read_csv(file_path)
                self.dataset_filepath_label.setText(file_path)
                print("CSV loaded successfully!")
                print(self.df.head())
            except Exception as e:
                print(f"Error while loading CSV: {e}")
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load CSV: {e}")
                def cancel_action():
                    print("User canceled.")

                msg = CustomMessageBox(
                    "Error Warning",
                    "Warning!\n Failed to load CSV.",
                    [
                        ("Cancel", QtWidgets.QMessageBox.ButtonRole.RejectRole, cancel_action)
                    ],
                    self
                )
                msg.exec()
        else:
            return
    #unused
    def save_csv(self, df):
        if self.df is None or self.df.empty:
            #QtWidgets.QMessageBox.warning(self, "Warning", "No data to save.")
            def cancel_action():
                print("User canceled.")

            msg = CustomMessageBox(
                "No Data Warning",
                "Warning!\nThere is no data to save.",
                [
                    ("Cancel", QtWidgets.QMessageBox.ButtonRole.RejectRole, cancel_action)
                ],
                self
            )
            msg.exec()
            return
        
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save CSV File", self.df_last_file_path, "CSV Files (*.csv);;All Files (*)")
        
        print("runs until here")
        if not file_path:
            return

        if not file_path.endswith(".csv"):
            file_path += ".csv"

        try:
            self.df.to_csv(file_path, index=False)
            print(f"CSV saved successfully at {file_path}!")
            #QtWidgets.QMessageBox.information(self, "Success", f"CSV saved successfully at:\n{file_path}")
            text = f"Success!\nCSV saved successfully at:\n{file_path}"
            def cancel_action():
                print("User canceled.")

            msg = CustomMessageBox(
                "Success",
                text,
                [
                    ("Ok", QtWidgets.QMessageBox.ButtonRole.RejectRole, cancel_action)
                ],
                self
            )
            msg.exec()

        except Exception as e:
            print(f"Error while saving CSV: {e}")
            #QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save CSV: {e}")
            text = f"Error!\nFailed to save CSV: {e}"
            def cancel_action():
                print("User canceled.")

            msg = CustomMessageBox(
                "Error",
                text,
                [
                    ("Discard", QtWidgets.QMessageBox.ButtonRole.RejectRole, cancel_action)
                ],
                self
            )
            msg.exec()

    def show_warning(self):
        #opacity_effect = QtWidgets.QGraphicsOpacityEffect(self)
        #opacity_effect.setOpacity(0.5)  # Set the dimming level (0.0 to 1.0)
        #self.setGraphicsEffect(opacity_effect)
        # actions to be taken when the buttons are clicked
        
        def discard_action():
            #print("User reset!")
            
            self.loading_label.hide()
            self.navigator.showHomePage()
            self.model_train_button.setEnabled(True)
            self.modify_dataset_button.setEnabled(True)
            self.model_undo_button.setEnabled(True)
            self.GLOBAL_CHOSEN_MODEL = None
            self.GLOBAL_CHOSEN_START = None
            self.GLOBAL_STAGE = 1
            self.model_train_button.setText("Train Model")
            self.model_train_button.disconnect()
            self.model_train_button.clicked.connect(self.navigator.showModelStartingPage)
            self.model_train_button.setEnabled(True)
            if self.navigator.trainer is not None:
                self.navigator.trainer.running = False
                self.navigator.trainer.quit()
                self.navigator.trainer.wait()
                #self.navigator.trainer.join() # Wait for the thread to finish! doesnt work
            self.neural_networks = []
            self.navigator.showModelStartingPage()
            self.listWidget.clearSelection()
            self.listWidget_2.clearSelection()
            self.navigator.updateTrainingProcessLabel("")
            self.training_process_label.setText("")



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
