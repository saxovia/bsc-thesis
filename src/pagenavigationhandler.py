import PyQt6 as Qt
from PyQt6.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt6.QtWidgets import QVBoxLayout
import matplotlib.pyplot as plt
import os
from datetime import datetime
import csv
from io import BytesIO
import networkx as nx
import pickle


class PageNavigationHandler:
    def __init__(self, main_window):
        self.main_window = main_window
        self.trainer = None
        self.trainers = []
        self.metrics = {}
        
    def fadeToPage(self, new_page):
        #force it to wait at first - for the padding to apply
        if new_page == None:
            return
        QTimer.singleShot(100, lambda: self.performFadeIn(new_page))
        #set the page to the new page
        #self.main_window.stackedWidget.setCurrentWidget(new_page)
        #self.main_window.fadeInUp(new_page)

    def performFadeIn(self, new_page):
        # Set the page to the new page
        self.main_window.stackedWidget.setCurrentWidget(new_page)
        self.main_window.fadeInUp(new_page)

    def resetSettingsButton(self):
        self.main_window.settings_button.disconnect()
        self.main_window.settings_button.clicked.connect(self.showSettingsPage)
    def showHomePage(self):
        self.fadeToPage(self.main_window.home_page)
        self.main_window.previous_page = self.main_window.current_page
        self.main_window.current_page = "Home"
        self.main_window.ui_handler.type_text_effect(self.main_window.home_text, self.main_window.home_text.text(), self.main_window.home_page)
        self.main_window.restart_button.hide()
    
        self.resetSettingsButton()
        self.main_window.undo_button.show()

    def showModelPage(self):
        self.fadeToPage(self.main_window.model_page)
        self.main_window.previous_page = self.main_window.current_page
        self.main_window.current_page = "Model"
        self.main_window.restart_button.show()
        self.resetSettingsButton()

    def showTimelinePage(self):
        QTimer.singleShot(100, lambda: self.fadeToPage(self.main_window.stackedWidget.setCurrentWidget(self.main_window.timeline_page)))
        for child in self.main_window.findChildren(Qt.QtWidgets.QAbstractItemView):
            child.clearSelection()
        self.fadeToPage(self.main_window.timeline_page)
        self.main_window.previous_page = self.main_window.current_page
        self.main_window.current_page = "Timeline"
        self.main_window.restart_button.show()
        self.resetSettingsButton()
        self.main_window.model_train_button.setText("Start Training")
        self.main_window.model_train_button.clicked.connect(self.startTrainingButton)
        self.main_window.undo_button.hide()

    def startTrainingButton(self):
        if self.main_window.current_page == "Model":
            self.savePruningChangesAndGoBack()
        self.main_window.model_training_handler.parseThroughProcessesTable()

    def showChooseResultsPage(self):
        self.fadeToPage(self.main_window.choose_results_page)
        self.main_window.previous_page = self.main_window.current_page
        self.main_window.current_page = "Results"
        self.main_window.restart_button.show()
        self.resetSettingsButton()
        self.main_window.undo_button.hide()
        self.main_window.model_training_handler.resetUI()

    def showSettingsPage(self):
        self.fadeToPage(self.main_window.settings_page)
        self.main_window.previous_page = self.main_window.current_page
        self.main_window.current_page = "Settings"
        print(self.main_window.previous_page)
        self.main_window.settings_button.disconnect()

        default_dir = self.load_default_graph_directory()
        self.main_window.input_prior_graph_save_dir.setPlaceholderText(default_dir)

        if self.main_window.previous_page == "Home":
            self.main_window.settings_button.clicked.connect(self.showHomePage)
        elif self.main_window.previous_page == "Results":
            self.main_window.settings_button.clicked.connect(self.showChooseResultsPage)
        elif self.main_window.previous_page == "Model":
            self.main_window.settings_button.clicked.connect(self.showModelPage)
        elif self.main_window.previous_page == "Timeline":
            self.main_window.settings_button.clicked.connect(self.showTimelinePage)
        else:
            print("Error: No previous page found")

    def showModelPriorPage(self):
        self.main_window.stackedWidget_2.setCurrentWidget(self.main_window.model_prior_start_page)
        self.main_window.model_train_button.setText("Start Training")
        if self.main_window.model_train_button.signalsBlocked():
            self.main_window.model_train_button.disconnect()

    def showPruningStartPage(self):
        self.main_window.stackedWidget_2.setCurrentWidget(self.main_window.model_pruning_page)
        self.main_window.model_train_button.setText("Start Training")
        if self.main_window.model_train_button.signalsBlocked():
            self.main_window.model_train_button.disconnect()
        self.main_window.model_train_button.clicked.connect(self.showModelPruningTablePage)

    def showModelPruningTablePage(self):
        self.main_window.stackedWidget_2.setCurrentWidget(self.main_window.model_pruning_table_page)
        self.main_window.model_train_button.setText("Start Training")
        self.main_window.undo_button.show()
        try:
            self.main_window.undo_button.clicked.disconnect()
        except:
            pass
            
        self.main_window.undo_button.show()
        self.main_window.undo_button.setEnabled(True)
        self.main_window.undo_button.clicked.connect(self.savePruningChangesAndGoBack)

    def savePruningChangesAndGoBack(self):
        current_row = self.main_window.reorder_table_view2.currentIndex().row()
        self.main_window.reorder_table_view2.clearSelection()
        selection_model = self.main_window.reorder_table_view2.selectionModel()
        selected_rows = set(index.row() for index in selection_model.selectedRows())
        if not selected_rows:
            current_row = self.main_window.reorder_table_view2.currentIndex().row()
            if current_row >= 0:
                selected_rows = {current_row}

        for row in range(self.main_window.pruningTableModel.rowCount()):
            index = self.main_window.pruningTableModel.index(row, 0)
            self.main_window.pruningTableModel.setData(index, False, Qt.QtCore.Qt.ItemDataRole.EditRole)

        if selected_rows:
            pruning_data = []
            for row in range(self.main_window.pruningTableModel.rowCount()):
                row_data = []
                for col in range(self.main_window.pruningTableModel.columnCount()):
                    index = self.main_window.pruningTableModel.index(row, col)
                    row_data.append(self.main_window.pruningTableModel.data(index, Qt.QtCore.Qt.ItemDataRole.DisplayRole))

        self.showTimelinePage()
    def visualize_results(self):
        if not self.main_window.previous_results:
            print("No results available for visualization.")
            return
        self.main_window.results_handler.visualize_results(self.main_window.previous_results)
    
    def load_graphs_from_main_menu(self):
        self.main_window.results_handler.load_graphs_from_main_menu()
    
    def load_default_graph_directory(self):
        return self.main_window.results_handler.load_default_graph_directory()
    
    def save_graphs(self):
        self.main_window.results_handler.save_graphs()

    def saveSettings(self):
        settings_file = os.path.join(os.path.dirname(__file__), "..", "settings.txt")
        
        prior_graph_save_dir = self.main_window.input_prior_graph_save_dir.text()
        if not prior_graph_save_dir:
            return
        messages = []

        if not os.path.isdir(prior_graph_save_dir):
            prior_graph_save_dir = os.path.join(os.path.dirname(__file__), "..", "savedgraphs")
            messages.append("Default graph save directory used!")
        if messages:
            self.main_window.saved_settings_label.setText(" ".join(messages))

        try:
            with open(settings_file, 'w') as f:
                f.write("# Saved graphs file location\n")
                f.write(f"{prior_graph_save_dir}\n")
            self.main_window.saved_settings_label.setText(f"Settings successfully saved, with {prior_graph_save_dir} as graph save directory!")
        except (FileNotFoundError, IOError) as e:
            self.main_window.saved_settings_label.setText(f"Error saving settings: {str(e)}")