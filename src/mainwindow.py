from PyQt6 import QtWidgets, uic, QtCore
import psutil
import GPUtil
from src.messagebox import CustomMessageBox
import sys
import pandas as pd
import numpy as np

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


        # Other buttons
        self.dataset_csv_button.clicked.connect(self.load_csv)
        self.dataset_save_button.clicked.connect(self.save_csv)
        self.dataset_apply_button.clicked.connect(self.applyChangesToDataset)
        self.view_header_button.clicked.connect(self.viewDataset)
        self.settings_button.clicked.connect(self.showSettingsPage)

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

        self.df = None
        self.df_last_file_path = ""
        self.current_page = "Home"
        self.onSettingsPage = False
        

        self.showHomePage() #this ensures to start at the home page

    def applyChangesToDataset(self):
        if self.encoding_input.toPlainText() != "":
            #print("runnign")
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


    def fadeToPage(self, new_page):
            #print(self.current_page, "when fading to page")
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

    def changeSettingsButton(self):
        self.settings_button.clicked.connect(self.showSettingsPage)

    def showHomePage(self):
        self.fadeToPage(self.home_page)
        self.previous_page = self.current_page
        self.current_page = "Home"
        self.changeSettingsButton()

    def showDatasetPage(self):
        self.fadeToPage(self.dataset_page)
        self.previous_page = self.current_page
        self.current_page = "Dataset"
        self.changeSettingsButton()

    def showModelPage(self):
        self.fadeToPage(self.model_page)
        self.previous_page = self.current_page
        self.current_page = "Model"
        self.changeSettingsButton()

    def showChooseResultsPage(self):
        self.fadeToPage(self.choose_results_page)
        self.previous_page = self.current_page
        self.current_page = "Results"
        self.changeSettingsButton()

    def showSettingsPage(self):
        self.fadeToPage(self.settings_page)
        self.previous_page = self.current_page
        self.current_page = "Settings"

        if self.previous_page == "Home":
            self.settings_button.clicked.connect(self.showHomePage)
        elif self.previous_page == "Dataset":
            self.settings_button.clicked.connect(self.showDatasetPage)
        elif self.previous_page == "Model":
            self.settings_button.clicked.connect(self.showModelPage)
        elif self.previous_page == "Results":
            self.settings_button.clicked.connect(self.showChooseResultsPage)

    def load_csv(self):
        # Open file dialog to choose CSV file
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
        # If the user cancels
        if not file_path:
            return

        # Ensure it has a .csv extension
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
        # Actions to be taken when the buttons are clicked
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
