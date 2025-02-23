import pandas as pd
from PyQt6 import QtWidgets

class CSVHandler:
    def __init__(self, main_window):
        self.main_window = main_window

    def load_csv(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self.main_window, "Select CSV File", "", "CSV Files (*.csv)")
        if file_path:
            try:
                self.main_window.df = pd.read_csv(file_path)
                self.main_window.dataset_filepath_label.setText(file_path)
            except Exception as e:
                QtWidgets.QMessageBox.critical(self.main_window, "Error", f"Failed to load CSV: {e}")

    def save_csv(self):
        if self.main_window.df is None or self.main_window.df.empty:
            QtWidgets.QMessageBox.warning(self.main_window, "Warning", "No data to save.")
            return

        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(self.main_window, "Save CSV File", "", "CSV Files (*.csv)")
        if file_path:
            if not file_path.endswith(".csv"):
                file_path += ".csv"

            try:
                self.main_window.df.to_csv(file_path, index=False)
            except Exception as e:
                QtWidgets.QMessageBox.critical(self.main_window, "Error", f"Failed to save CSV: {e}")
