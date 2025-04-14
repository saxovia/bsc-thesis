import PyQt6 as Qt
from src.trainer import Trainer
from PyQt6.QtGui import QMovie
from PyQt6.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt6.QtWidgets import QVBoxLayout
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from matplotlib.gridspec import GridSpec


class PageNavigationHandler:
    def __init__(self, main_window):
        self.main_window = main_window
        self.trainer = None
        self.trainers = []
        
    def fadeToPage(self, new_page):
        #force it to wait at first - for the padding to apply
        
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
        self.fadeToPage(self.main_window.timeline_page)
        self.main_window.previous_page = self.main_window.current_page
        self.main_window.current_page = "Timeline"
        self.main_window.restart_button.show()
        self.resetSettingsButton()
        self.main_window.model_train_button.setText("Start Training")
        self.main_window.model_train_button.clicked.connect(self.main_window.model_training_handler.parseThroughProcessesTable)

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
        self.main_window.model_train_button.clicked.connect(self.showModelTrainingPage)

    def showPruningStartPage(self):
        self.main_window.stackedWidget_2.setCurrentWidget(self.main_window.model_pruning_page)
        self.main_window.model_train_button.setText("Start Training")
        if self.main_window.model_train_button.signalsBlocked():
            self.main_window.model_train_button.disconnect()
        self.main_window.model_train_button.clicked.connect(self.showModelPruningTablePage)

    def showModelPruningTablePage(self):
        self.main_window.stackedWidget_2.setCurrentWidget(self.main_window.model_pruning_table_page)
        self.main_window.model_train_button.setText("Start Training")
        if self.main_window.model_train_button.signalsBlocked():
            self.main_window.model_train_button.disconnect()
        self.main_window.model_train_button.clicked.connect(self.showModelTrainingPage)
        self.main_window.undo_button.setEnabled(True)
        self.main_window.undo_button.clicked.connect(self.showTimelinePage)

    def showModelTrainingPage(self): # TODO delete later safely
        pass

    def visualize_results(self): #TODO migrate this elsewhere ?
        if not self.main_window.previous_results:
            print("No results available for visualization.")
            return

        metrics = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "global_sparsity": [],
            "total_parameters": [],
            "model_type": [],
            "graph_type": [],
            "prune_type": [],
            "degree": [],
            "eccentricity": [],
            "closeness": [],
            "betweenness": [],
            "edge_betweenness": []
        }
        for result in self.main_window.previous_results:
            metrics["model_type"].append(result["model_type"])
            metrics["graph_type"].append(result["graph_type"])

            training_metrics = result["training_metrics"]
            metrics["train_loss"].append(training_metrics["final_train_loss"])
            metrics["train_accuracy"].append(training_metrics["final_train_accuracy"])
            metrics["val_loss"].append(training_metrics["final_val_loss"])
            metrics["val_accuracy"].append(training_metrics["final_val_accuracy"])
            graph_metrics = result["graph_metrics"]
            metrics["global_sparsity"].append(graph_metrics["global_sparsity"])
            metrics["total_parameters"].append(graph_metrics["total_parameters"])
            metrics["prune_type"].append(result["prune_type"])

            metrics["degree"].append(graph_metrics.get("degree", {}))
            metrics["eccentricity"].append(graph_metrics.get("eccentricity", {}))
            metrics["closeness"].append(graph_metrics.get("closeness", {}))
            metrics["betweenness"].append(graph_metrics.get("betweenness", {}))
            metrics["edge_betweenness"].append(graph_metrics.get("edge_betweenness", {}))

        self.display_graphs(metrics)


    def display_graphs(self, metrics):
        layout = self.main_window.widget_11.layout()
        if layout is None:
            layout = QVBoxLayout(self.main_window.widget_11)
            self.main_window.widget_11.setLayout(layout)
        else:
            while layout.count():
                child = layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()

        # Create a figure with 2 rows and 3 columns of subplots
        fig, axs = plt.subplots(2, 3, figsize=(18, 12))
        
        # First subplot: Number of parameters vs accuracy (colored by graph_type)
        graph_types = ['BA', 'WS', 'Full']
        colors = {'BA': 'r', 'WS': 'g', 'Full': 'b'}
        
        for graph_type in graph_types:
            # Filter data for this graph_type
            indices = [i for i, gt in enumerate(metrics['graph_type']) if gt == graph_type]
            params = [metrics['total_parameters'][i] for i in indices]
            accuracies = [metrics['val_accuracy'][i] for i in indices]
            
            axs[0, 0].scatter(params, accuracies, color=colors[graph_type], label=graph_type)
        
        axs[0, 0].set_xlabel('Number of Parameters')
        axs[0, 0].set_ylabel('Validation Accuracy')
        axs[0, 0].set_title('Parameters vs Accuracy by Graph Type')
        axs[0, 0].legend()
        axs[0, 0].grid(True)
        
        # Prepare data for the other plots
        prune_percents = []
        eccentricities = []
        degrees = []
        closeness = []
        betweenness = []
        edge_betweenness = []
        prune_labels = []
        
        for i in range(len(metrics['model_type'])):
            if (metrics['model_type'][i] == 'MLP' and metrics['graph_type'][i] == 'Full'):
                current_prune_type = metrics['prune_type'][i][-1] if isinstance(metrics['prune_type'][i], list) else metrics['prune_type'][i]
                
                if current_prune_type in ['IH', 'HH', 'HO', 'FULL']:
                    # Get global sparsity (prune %)
                    prune_percent = metrics['global_sparsity'][i] * 100  # Convert to percentage
                    
                    # Get all metrics (assuming they're dictionaries)
                    eccentricity_dict = metrics['eccentricity'][i]
                    degree_dict = metrics['degree'][i]
                    closeness_dict = metrics['closeness'][i]
                    betweenness_dict = metrics['betweenness'][i]
                    edge_betweenness_dict = metrics['edge_betweenness'][i]

                    if eccentricity_dict:  # Check if not empty
                        # Calculate mean values
                        prune_percents.append(prune_percent)
                        prune_labels.append(current_prune_type)
                        eccentricities.append(sum(eccentricity_dict.values()) / len(eccentricity_dict))
                        degrees.append(sum(degree_dict.values()) / len(degree_dict) if degree_dict else 0)
                        closeness.append(sum(closeness_dict.values()) / len(closeness_dict) if closeness_dict else 0)
                        betweenness.append(sum(betweenness_dict.values()) / len(betweenness_dict) if betweenness_dict else 0)
                        edge_betweenness.append(sum(edge_betweenness_dict.values()) / len(edge_betweenness_dict) if edge_betweenness_dict else 0)

        # Create color mapping for prune types
        prune_colors = {'IH': 'red', 'HH': 'blue', 'HO': 'green', 'FULL': 'purple'}
        
        # Function to create a scatter plot for a given metric
        def plot_metric(ax, y_values, y_label, title):
            for prune_type in ['IH', 'HH', 'HO', 'FULL']:
                indices = [i for i, pt in enumerate(prune_labels) if pt == prune_type]
                ax.scatter(
                    [prune_percents[i] for i in indices],
                    [y_values[i] for i in indices],
                    color=prune_colors[prune_type],
                    label=prune_type
                )
            ax.set_xlabel('Prune Percentage (%)')
            ax.set_ylabel(y_label)
            ax.set_title(title)
            ax.legend()
            ax.grid(True)
        
        # Second subplot (top middle): Eccentricity vs Prune %
        plot_metric(axs[0, 1], eccentricities, 'Mean Eccentricity', 'Eccentricity vs Prune % (MLP, Full Graph)')
        
        # Third subplot (top right): Degree vs Prune %
        plot_metric(axs[0, 2], degrees, 'Mean Degree', 'Degree vs Prune % (MLP, Full Graph)')
        
        # Fourth subplot (bottom left): Closeness vs Prune %
        plot_metric(axs[1, 0], closeness, 'Mean Closeness', 'Closeness vs Prune % (MLP, Full Graph)')
        
        # Fifth subplot (bottom middle): Betweenness vs Prune %
        plot_metric(axs[1, 1], betweenness, 'Mean Betweenness', 'Betweenness vs Prune % (MLP, Full Graph)')
        
        # Sixth subplot (bottom right): Edge Betweenness vs Prune %
        plot_metric(axs[1, 2], edge_betweenness, 'Mean Edge Betweenness', 'Edge Betweenness vs Prune % (MLP, Full Graph)')
        
        # Adjust layout and display
        plt.tight_layout()
        
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)