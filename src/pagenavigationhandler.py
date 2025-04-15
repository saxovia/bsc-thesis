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

            model_metrics = training_metrics['model_metrics']
            metrics["global_sparsity"].append(model_metrics["global_sparsity"])
            metrics["total_parameters"].append(model_metrics["total_parameters"])

            if "prune_type" in model_metrics:
                metrics["prune_type"].append(model_metrics["prune_type"])
            else:
                metrics["prune_type"].append("None")

            metrics["degree"].append(graph_metrics.get("degree", {}))
            metrics["eccentricity"].append(graph_metrics.get("eccentricity", {}))
            metrics["closeness"].append(graph_metrics.get("closeness", {}))
            metrics["betweenness"].append(graph_metrics.get("betweenness", {}))
            metrics["edge_betweenness"].append(graph_metrics.get("edge_betweenness", {}))

        metrics['graph_type'].extend(['WS', 'Full'])
        metrics['total_parameters'].extend([30000, 1500])
        metrics['val_accuracy'].extend([85.0, 25.0])
        self.display_graphs(metrics)

    def display_graphs(self, metrics):
        container = self.main_window.widget_11
        layout = container.layout()
        if layout is None:
            layout = QVBoxLayout(container)
            container.setLayout(layout)
        else:
            while layout.count():
                child = layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()

        scroll = Qt.QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = Qt.QtWidgets.QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        
        self.add_graph_widget(scroll_layout, self.create_parameters_vs_accuracy_graph(metrics))
        self.add_graph_widget(scroll_layout, self.create_prune_metric_graph(metrics, "eccentricity", "Mean Eccentricity"))
        self.add_graph_widget(scroll_layout, self.create_prune_metric_graph(metrics, "degree", "Mean Degree"))
        self.add_graph_widget(scroll_layout, self.create_prune_metric_graph(metrics, "closeness", "Mean Closeness"))
        self.add_graph_widget(scroll_layout, self.create_prune_metric_graph(metrics, "betweenness", "Mean Betweenness"))
        self.add_graph_widget(scroll_layout, self.create_prune_metric_graph(metrics, "edge_betweenness", "Mean Edge Betweenness"))
        
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)

    def add_graph_widget(self, layout, figure):
        widget = Qt.QtWidgets.QWidget()
        widget_layout = QVBoxLayout(widget)
        widget_layout.setContentsMargins(0, 0, 0, 0)
        
        canvas = FigureCanvas(figure)
        widget_layout.addWidget(canvas)
        
        widget.setStyleSheet("border: 1px solid #ddd; margin-bottom: 10px;")
        layout.addWidget(widget)
        layout.addSpacing(10)

    def create_parameters_vs_accuracy_graph(self, metrics):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        graph_types = ['BA', 'WS', 'Full']
        colors = {'BA': 'r', 'WS': 'g', 'Full': 'b'}
        markers = {'BA': 'o', 'WS': 's', 'Full': 'D'}
        
        for graph_type in graph_types:
            indices = [i for i, gt in enumerate(metrics['graph_type']) if gt == graph_type]
            params = [metrics['total_parameters'][i] for i in indices]
            accuracies = [metrics['val_accuracy'][i] for i in indices]
            
            # Make single points larger and add labels
            if len(params) == 1:
                ax.scatter(params, accuracies, color=colors[graph_type], 
                        marker=markers[graph_type], s=200, label=f'{graph_type}',
                        edgecolors='black', linewidths=1)
            else:
                ax.scatter(params, accuracies, color=colors[graph_type],
                        marker=markers[graph_type], label=graph_type, s=100)
        
        ax.set_xlabel('Number of Parameters')
        ax.set_ylabel('Validation Accuracy')
        ax.set_title('Parameters vs Accuracy by Graph Type')
        ax.legend()
        ax.grid(True)
        
        # Add annotations for single points
        for i, (gt, param, acc) in enumerate(zip(metrics['graph_type'], 
                                            metrics['total_parameters'], 
                                            metrics['val_accuracy'])):
            ax.annotate(f"{gt}\n{acc:.2f}%", 
                    (param, acc), 
                    textcoords="offset points",
                    xytext=(10,10), 
                    ha='center')
        
        plt.tight_layout()
        return fig

    def create_prune_metric_graph(self, metrics, metric_name, y_label):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        prune_percents = []
        metric_values = []
        prune_labels = []
        
        for i in range(len(metrics['model_type'])):
            if (metrics['model_type'][i] == 'MLP' and metrics['graph_type'][i] == 'Full'):
                current_prune_type = metrics['prune_type'][i][-1] if isinstance(metrics['prune_type'][i], list) else metrics['prune_type'][i]
                
                if current_prune_type in ['IH', 'HH', 'HO', 'FULL']:
                    prune_percent = metrics['global_sparsity'][i] * 100
                    metric_dict = metrics[metric_name][i]
                    
                    if metric_dict:
                        prune_percents.append(prune_percent)
                        prune_labels.append(current_prune_type)
                        metric_values.append(sum(metric_dict.values()) / len(metric_dict))
        
        prune_colors = {'IH': 'red', 'HH': 'blue', 'HO': 'green', 'FULL': 'purple'}
        
        for prune_type in ['IH', 'HH', 'HO', 'FULL']:
            indices = [i for i, pt in enumerate(prune_labels) if pt == prune_type]
            ax.scatter(
                [prune_percents[i] for i in indices],
                [metric_values[i] for i in indices],
                color=prune_colors[prune_type],
                label=prune_type
            )
        
        ax.set_xlabel('Prune Percentage (%)')
        ax.set_ylabel(y_label)
        ax.set_title(f'{y_label} vs Prune % (MLP, Full Graph)')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        return fig