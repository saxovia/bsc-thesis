import PyQt6 as Qt
from PyQt6.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt6.QtWidgets import QVBoxLayout
import matplotlib.pyplot as plt
import os


class PageNavigationHandler:
    def __init__(self, main_window):
        self.main_window = main_window
        self.trainer = None
        self.trainers = []
        self.metrics = {}
        
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
        self.metrics = {
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
            self.metrics["model_type"].append(result["model_type"])
            self.metrics["graph_type"].append(result["graph_type"])

            training_metrics = result["training_metrics"]
            self.metrics["train_loss"].append(training_metrics["final_train_loss"])
            self.metrics["train_accuracy"].append(training_metrics["final_train_accuracy"])
            self.metrics["val_loss"].append(training_metrics["final_val_loss"])
            self.metrics["val_accuracy"].append(training_metrics["final_val_accuracy"])
            graph_metrics = result["graph_metrics"]

            model_metrics = training_metrics['model_metrics']
            self.metrics["global_sparsity"].append(model_metrics["global_sparsity"])
            self.metrics["total_parameters"].append(model_metrics["total_parameters"])

            if "prune_type" in model_metrics:
                self.metrics["prune_type"].append(model_metrics["prune_type"])
            else:
                self.metrics["prune_type"].append("None")

            self.metrics["degree"].append(graph_metrics.get("degree", {}))
            self.metrics["eccentricity"].append(graph_metrics.get("eccentricity", {}))
            self.metrics["closeness"].append(graph_metrics.get("closeness", {}))
            self.metrics["betweenness"].append(graph_metrics.get("betweenness", {}))
            self.metrics["edge_betweenness"].append(graph_metrics.get("edge_betweenness", {}))

        #TODO REMOVE!! just tests
        self.metrics['graph_type'].extend(['Full', 'Full'])
        self.metrics['total_parameters'].extend([30000, 1500])
        self.metrics['val_accuracy'].extend([85.0, 25.0])
        self.metrics['model_type'].extend(['MLP', 'MLP'])
        self.metrics['prune_type'].extend(['IH', 'HH'])
        self.metrics['global_sparsity'].extend([0.5, 0.2])
        self.metrics['degree'].extend([{'node1': 0.5, 'node2': 0.7}, {'node1': 0.3, 'node2': 0.4}])
        self.metrics['eccentricity'].extend([{'node1': 0.1, 'node2': 0.2}, {'node1': 0.3, 'node2': 0.4}])
        self.metrics['closeness'].extend([{'node1': 0.6, 'node2': 0.8}, {'node1': 0.4, 'node2': 0.5}])
        self.metrics['betweenness'].extend([{'node1': 0.2, 'node2': 0.3}, {'node1': 0.4, 'node2': 0.5}])
        self.metrics['edge_betweenness'].extend([{'node1': 0.3, 'node2': 0.4}, {'node1': 0.5, 'node2': 0.6}])

        self.display_graphs(self.metrics)

    def display_graphs(self, metrics):
        scroll_content = self.main_window.scrollAreaWidgetContents_2
        layout = scroll_content.layout()
        if layout is None:
            layout = QVBoxLayout(scroll_content)
            scroll_content.setLayout(layout)
        else:
            self.clear_layout(layout)
        
        graphs = [
            self.create_parameters_vs_accuracy_graph(metrics),
            self.create_prune_metric_graph(metrics, "eccentricity", "Mean Eccentricity"),
            self.create_prune_metric_graph(metrics, "degree", "Mean Degree"),
            self.create_prune_metric_graph(metrics, "closeness", "Mean Closeness"),
            self.create_prune_metric_graph(metrics, "betweenness", "Mean Betweenness"),
            self.create_prune_metric_graph(metrics, "edge_betweenness", "Mean Edge Betweenness")
        ]
        
        plt.style.use('dark_background')
        for fig in graphs:
            container = Qt.QtWidgets.QWidget()
            container.setMinimumSize(800, 500)
            
            container_layout = QVBoxLayout(container)
            container_layout.setContentsMargins(10, 10, 10, 10)
            
            canvas = FigureCanvas(fig)
            container_layout.addWidget(canvas)
            
            layout.addWidget(container)
            layout.addSpacing(15)
        
        # Ensure proper updating
        scroll_content.adjustSize()
    def clear_layout(self, layout):
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def add_graph_widget(self, layout, figure):
        widget = Qt.QtWidgets.QWidget()
        widget_layout = QVBoxLayout(widget)
        widget_layout.setContentsMargins(0,0,0,0)
        
        canvas = FigureCanvas(figure)
        widget_layout.addWidget(canvas)
        
        widget.setStyleSheet("border: 1px solid #ddd; margin-bottom: 10px;")
        layout.addWidget(widget)
        layout.addSpacing(10)

    def create_parameters_vs_accuracy_graph(self, metrics):
        fig = plt.figure(figsize=(8, 3))
        ax = fig.add_subplot(111)
        
        graph_types = ['BA','WS','Full']
        colors = {'BA':'r','WS':'g','Full':'b'}
        
        for graph_type in graph_types:
            indices = [i for i, gt in enumerate(metrics['graph_type']) if gt == graph_type]
            params = [metrics['total_parameters'][i] for i in indices]
            accuracies = [metrics['val_accuracy'][i] for i in indices]
            
            if params:
                ax.scatter(params, accuracies, color=colors[graph_type], 
                        label=f"{graph_type} (n={len(params)})", s=100)
        
        ax.set_xlabel('Number of Parameters')
        ax.set_ylabel('Validation Accuracy')
        ax.set_title('Parameters vs Accuracy by Graph Type')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        return fig

    def create_prune_metric_graph(self, metrics, metric_name, y_label):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        prune_percents = []
        metric_values = []
        prune_labels = []
        
        for i in range(len(metrics['graph_type'])):
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
    

    def save_graphs(self, save_dir="savedgraphs"):
        save_dir = "savedgraphs" #TODO Temporary fix, remove later
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        graphs = [
            ("parameters_vs_accuracy.png", self.create_parameters_vs_accuracy_graph(self.metrics)),
            ("mean_eccentricity.png", self.create_prune_metric_graph(self.metrics, "eccentricity", "Mean Eccentricity")),
            ("mean_degree.png", self.create_prune_metric_graph(self.metrics, "degree", "Mean Degree")),
            ("mean_closeness.png", self.create_prune_metric_graph(self.metrics, "closeness", "Mean Closeness")),
            ("mean_betweenness.png", self.create_prune_metric_graph(self.metrics, "betweenness", "Mean Betweenness")),
            ("mean_edge_betweenness.png", self.create_prune_metric_graph(self.metrics, "edge_betweenness", "Mean Edge Betweenness"))
        ]
        
        for filename, fig in graphs:
            fig.savefig(os.path.join(save_dir, filename))
            plt.close(fig)
        self.main_window.saved_label.setText("Graphs saved successfully!")
                
    def load_and_display_graphs(self, save_dir="savedgraphs"):
        self.showChooseResultsPage()
        self.main_window.previous_page = self.main_window.current_page
        save_dir = "savedgraphs"
        scroll_content = self.main_window.scrollAreaWidgetContents_2
        layout = scroll_content.layout()
        if layout is None:
            layout = QVBoxLayout(scroll_content)
            scroll_content.setLayout(layout)
        else:
            self.clear_layout(layout)
        scroll_area_width = self.main_window.scrollArea_2.width()

        for filename in os.listdir(save_dir):
            if filename.endswith(".png"):
                graph_path = os.path.join(save_dir, filename)
                label = Qt.QtWidgets.QLabel()
                pixmap = Qt.QtGui.QPixmap(graph_path)
                scaled_pixmap = pixmap.scaled(scroll_area_width - 20, pixmap.height(),Qt.QtCore.Qt.AspectRatioMode.KeepAspectRatio,Qt.QtCore.Qt.TransformationMode.SmoothTransformation)
                label.setPixmap(scaled_pixmap)
                label.setPixmap(pixmap)
                label.setScaledContents(False)
                label.setAlignment(Qt.QtCore.Qt.AlignmentFlag.AlignCenter)
                layout.addWidget(label)
                layout.addSpacing(15)
        
        scroll_content.adjustSize()