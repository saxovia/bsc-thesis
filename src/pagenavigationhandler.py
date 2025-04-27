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
            
        selection_model = self.main_window.reorder_table_view2.selectionModel()
        selected_rows = set(index.row() for index in selection_model.selectedRows())
        if not selected_rows:
            current_row = self.main_window.reorder_table_view2.currentIndex().row()
            if current_row >= 0:
                selected_rows = {current_row}
        if selected_rows:
            pruning_data = []
            for row in range(self.main_window.pruningTableModel.rowCount()):
                row_data = []
                for col in range(self.main_window.pruningTableModel.columnCount()):
                    index = self.main_window.pruningTableModel.index(row, col)
                    row_data.append(self.main_window.pruningTableModel.data(index, Qt.QtCore.Qt.ItemDataRole.DisplayRole))
                pruning_data.append(row_data)
            
            for row in selected_rows:
                if 0 <= row < self.main_window.timelineTableModel.rowCount():
                    self.main_window.timelineTableModel.set_hidden_data(row, pruning_data)
        self.showTimelinePage()


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
            "trainable_parameters": [],
            "layer_sparsity": [],
            "feature_size": [],
            "sequence_length": [],
            "input_size": [],
            "num_classes": [],
            "dataset_type": [],
            "model_type": [],
            "graph_type": [],
            "prune_type": [],
            "prune_mode": [],
            "degree": [],
            "eccentricity": [],
            "closeness": [],
            "betweenness": [],
            "edge_betweenness": [],
            "graph_type": [],
            "hidden_sizes": [],
            "lr": [],
            "batch_size": [],
            "k": [],
            "p": [],
            "optimizer_type": [],
            "loss": [],
            "serialized_graph": [],

            "avg_degree": [],
            "avg_eccentricity": [],
            "avg_closeness": [],
            "avg_betweenness": [],
            "avg_edge_betweenness": [],
            "avg_global_sparsity": [],

            "min_degree": [],
            "min_eccentricity": [],
            "min_closeness": [],
            "min_betweenness": [],
            "min_edge_betweenness": [],
            "min_global_sparsity": [],

            "max_degree": [],
            "max_eccentricity": [],
            "max_closeness": [],
            "max_betweenness": [],
            "max_edge_betweenness": [],
            "max_global_sparsity": [],

            "std_degree": [],
            "std_eccentricity": [],
            "std_closeness": [],
            "std_betweenness": [],
            "std_edge_betweenness": [],
            "std_global_sparsity": [],
            
            "variance_degree": [],
            "variance_eccentricity": [],
            "variance_closeness": [],
            "variance_betweenness": [],
            "variance_edge_betweenness": [],
            "variance_global_sparsity": [],

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
            if "prune_mode" in model_metrics:
                self.metrics["prune_mode"].append(model_metrics["prune_mode"])
            else:
                self.metrics["prune_mode"].append("None")

            self.metrics["degree"].append(graph_metrics.get("degree", {}))
            self.metrics["eccentricity"].append(graph_metrics.get("eccentricity", {}))
            self.metrics["closeness"].append(graph_metrics.get("closeness", {}))
            self.metrics["betweenness"].append(graph_metrics.get("betweenness", {}))
            self.metrics["edge_betweenness"].append(graph_metrics.get("edge_betweenness", {}))
            self.metrics["hidden_sizes"].append(result.get("hidden_sizes", []))
            self.metrics["lr"].append(result.get("lr", 0))
            self.metrics["batch_size"].append(result.get("batch_size", 0))
            self.metrics["k"].append(result.get("k", 0))
            self.metrics["p"].append(result.get("p", 0))
            self.metrics["optimizer_type"].append(result.get("optimizer_type", ""))
            self.metrics["dataset_type"].append(result.get("dataset_type", ""))
            self.metrics["feature_size"].append(result.get("feature_size", 0))
            self.metrics["sequence_length"].append(result.get("sequence_length", 0))
            self.metrics["input_size"].append(result.get("input_size", 0))
            self.metrics["num_classes"].append(result.get("num_classes", 0))
            self.metrics["layer_sparsity"].append(model_metrics.get("layer_sparsity", []))
            self.metrics["trainable_parameters"].append(model_metrics.get("trainable_parameters", 0))
            self.metrics["loss"].append(result.get("criterion", 0))
            self.metrics["serialized_graph"].append(result.get("serialized_graph", ""))

            # Calculated metrics
            self.metrics["avg_degree"].append(sum(graph_metrics["degree"].values()) / len(graph_metrics["degree"]))
            self.metrics["avg_eccentricity"].append(sum(graph_metrics["eccentricity"].values()) / len(graph_metrics["eccentricity"]))
            self.metrics["avg_closeness"].append(sum(graph_metrics["closeness"].values()) / len(graph_metrics["closeness"]))
            self.metrics["avg_betweenness"].append(sum(graph_metrics["betweenness"].values()) / len(graph_metrics["betweenness"]))
            self.metrics["avg_edge_betweenness"].append(sum(graph_metrics["edge_betweenness"].values()) / len(graph_metrics["edge_betweenness"]))

            self.metrics["min_degree"].append(min(graph_metrics["degree"].values()))
            self.metrics["min_eccentricity"].append(min(graph_metrics["eccentricity"].values()))
            self.metrics["min_closeness"].append(min(graph_metrics["closeness"].values()))
            self.metrics["min_betweenness"].append(min(graph_metrics["betweenness"].values()))
            self.metrics["min_edge_betweenness"].append(min(graph_metrics["edge_betweenness"].values()))

            self.metrics["max_degree"].append(max(graph_metrics["degree"].values()))
            self.metrics["max_eccentricity"].append(max(graph_metrics["eccentricity"].values()))
            self.metrics["max_closeness"].append(max(graph_metrics["closeness"].values()))
            self.metrics["max_betweenness"].append(max(graph_metrics["betweenness"].values()))
            self.metrics["max_edge_betweenness"].append(max(graph_metrics["edge_betweenness"].values()))

            self.metrics["std_degree"].append(sum((x - self.metrics["avg_degree"][-1]) ** 2 for x in graph_metrics["degree"].values()) / len(graph_metrics["degree"]))
            self.metrics["std_eccentricity"].append(sum((x - self.metrics["avg_eccentricity"][-1]) ** 2 for x in graph_metrics["eccentricity"].values()) / len(graph_metrics["eccentricity"]))
            self.metrics["std_closeness"].append(sum((x - self.metrics["avg_closeness"][-1]) ** 2 for x in graph_metrics["closeness"].values()) / len(graph_metrics["closeness"]))
            self.metrics["std_betweenness"].append(sum((x - self.metrics["avg_betweenness"][-1]) ** 2 for x in graph_metrics["betweenness"].values()) / len(graph_metrics["betweenness"]))
            self.metrics["std_edge_betweenness"].append(sum((x - self.metrics["avg_edge_betweenness"][-1]) ** 2 for x in graph_metrics["edge_betweenness"].values()) / len(graph_metrics["edge_betweenness"]))

            self.metrics["variance_degree"].append(
                sum((x - self.metrics["avg_degree"][-1]) ** 2 for x in graph_metrics["degree"].values()) / len(graph_metrics["degree"])
                if len(graph_metrics["degree"]) > 0 else 0
            )
            self.metrics["variance_eccentricity"].append(
                sum((x - self.metrics["avg_eccentricity"][-1]) ** 2 for x in graph_metrics["eccentricity"].values()) / len(graph_metrics["eccentricity"])
                if len(graph_metrics["eccentricity"]) > 0 else 0
            )
            self.metrics["variance_closeness"].append(
                sum((x - self.metrics["avg_closeness"][-1]) ** 2 for x in graph_metrics["closeness"].values()) / len(graph_metrics["closeness"])
                if len(graph_metrics["closeness"]) > 0 else 0
            )
            self.metrics["variance_betweenness"].append(
                sum((x - self.metrics["avg_betweenness"][-1]) ** 2 for x in graph_metrics["betweenness"].values()) / len(graph_metrics["betweenness"])
                if len(graph_metrics["betweenness"]) > 0 else 0
            )
            self.metrics["variance_edge_betweenness"].append(
                sum((x - self.metrics["avg_edge_betweenness"][-1]) ** 2 for x in graph_metrics["edge_betweenness"].values()) / len(graph_metrics["edge_betweenness"])
                if len(graph_metrics["edge_betweenness"]) > 0 else 0
            )

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

        custom_theme = {
            'axes.facecolor': '#303338',
            'axes.edgecolor': '#FFF7ED',
            'axes.labelcolor': '#FFF7ED',
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.xmargin': 0.02,
            'axes.ymargin': 0.02,
            'xtick.color': '#F2E6D4',
            'ytick.color': '#F2E6D4',
            'xtick.direction': 'inout',
            'ytick.direction': 'inout',
            'xtick.major.size': 10,
            'ytick.major.size': 10,
            'xtick.minor.size': 3,
            'ytick.minor.size': 3,
            'xtick.major.width': 2,
            'ytick.major.width': 2,
            'xtick.minor.width': 0.75,
            'ytick.minor.width': 0.75,
            'grid.color': '#F2E6D4',
            'grid.linewidth': 0.5,
            'grid.alpha': 0.7,
            'figure.facecolor': '#24272B',
            'text.color': '#FFFDFB',
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial'],
            'lines.color': '#FFF7ED',
            'legend.facecolor': '#303338',
            'legend.edgecolor': '#FFF7ED',
            'axes.titleweight': 'bold',
            'axes.titlepad': 10, 
            'axes.titlelocation': 'left', 
        }
        
        plt.rcParams.update(custom_theme)
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
        fig = plt.figure(figsize=(7, 3))
        ax = fig.add_subplot(111)
        
        graph_types = ['BA','WS','Full']
        colors = {'BA':'#a386fc','WS':'#86a9fc','Full':'#86dffc'}
        
        
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
        
        prune_colors = {'IH': '#a386fc', 'HH': '#86a9fc', 'HO': '#86dffc', 'FULL': '#86fcdf'}
        
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
    

    def load_default_graph_directory(self):
        settings_file = os.path.join(os.path.dirname(__file__), "..", "settings.txt")
        fallback_path = os.path.join(os.path.dirname(__file__), "..", ".")
        
        default_dir = fallback_path

        try:
            with open(settings_file, 'r') as f:
                lines = [line.strip() for line in f.readlines()]
                if "# Saved graphs file location" in lines:
                    index = lines.index("# Saved graphs file location")
                    if index + 1 < len(lines):
                        default_dir = lines[index + 1]
        except (FileNotFoundError, IOError) as e:
            print(f"Note: Using fallback path ({fallback_path}) because: {str(e)}")

        return default_dir
        

    def save_graphs(self):
        default_dir = self.load_default_graph_directory()
        timestamp = datetime.now().strftime("model-%Y-%m-%d-%H-%M-%S")
        save_dir = os.path.join(default_dir, timestamp)
        os.makedirs(save_dir, exist_ok=True)

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

        csv_file_path = os.path.join(save_dir, "model_results.csv")
        with open(csv_file_path, "w", newline="") as csvfile:
            fieldnames = [
                "model_type", "graph_type", "prune_type", "global_sparsity", "total_parameters", 
                "train_loss", "train_accuracy", "val_loss", "val_accuracy",
                "degree", "eccentricity", "closeness", "betweenness", "edge_betweenness",
                "hidden_sizes", "lr", "batch_size", "k", "p", "optimizer_type",
                "dataset_type", "feature_size", "sequence_length", "input_size", "num_classes",
                "layer_sparsity", "trainable_parameters",
                "avg_degree", "avg_eccentricity", "avg_closeness", "avg_betweenness", "avg_edge_betweenness",
                "min_degree", "min_eccentricity", "min_closeness", "min_betweenness", "min_edge_betweenness",
                "max_degree", "max_eccentricity", "max_closeness", "max_betweenness", "max_edge_betweenness",
                "std_degree", "std_eccentricity", "std_closeness", "std_betweenness", "std_edge_betweenness",
                "variance_degree", "variance_eccentricity", "variance_closeness", "variance_betweenness", "variance_edge_betweenness"
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for i in range(len(self.main_window.neural_networks)):
                writer.writerow({
                    "model_type": self.metrics["model_type"][i],
                    "graph_type": self.metrics["graph_type"][i],
                    "prune_type": self.metrics["prune_type"][i],
                    "global_sparsity": self.metrics["global_sparsity"][i],
                    "total_parameters": self.metrics["total_parameters"][i],
                    "train_loss": self.metrics["train_loss"][i],
                    "train_accuracy": self.metrics["train_accuracy"][i],
                    "val_loss": self.metrics["val_loss"][i],
                    "val_accuracy": self.metrics["val_accuracy"][i],
                    "degree": self.metrics["degree"][i],
                    "eccentricity": self.metrics["eccentricity"][i],
                    "closeness": self.metrics["closeness"][i],
                    "betweenness": self.metrics["betweenness"][i],
                    "edge_betweenness": self.metrics["edge_betweenness"][i],
                    "hidden_sizes": self.metrics["hidden_sizes"][i],
                    "lr": self.metrics["lr"][i],
                    "batch_size": self.metrics["batch_size"][i],
                    "k": self.metrics["k"][i],
                    "p": self.metrics["p"][i],
                    "optimizer_type": self.metrics["optimizer_type"][i],
                    "dataset_type": self.metrics["dataset_type"][i],
                    "feature_size": self.metrics["feature_size"][i],
                    "sequence_length": self.metrics["sequence_length"][i],
                    "input_size": self.metrics["input_size"][i],
                    "num_classes": self.metrics["num_classes"][i],
                    "layer_sparsity": self.metrics["layer_sparsity"][i],
                    "trainable_parameters": self.metrics["trainable_parameters"][i],
                    "avg_degree": self.metrics["avg_degree"][i],
                    "avg_eccentricity": self.metrics["avg_eccentricity"][i],
                    "avg_closeness": self.metrics["avg_closeness"][i],
                    "avg_betweenness": self.metrics["avg_betweenness"][i],
                    "avg_edge_betweenness": self.metrics["avg_edge_betweenness"][i],
                    "min_degree": self.metrics["min_degree"][i],
                    "min_eccentricity": self.metrics["min_eccentricity"][i],
                    "min_closeness": self.metrics["min_closeness"][i],
                    "min_betweenness": self.metrics["min_betweenness"][i],
                    "min_edge_betweenness": self.metrics["min_edge_betweenness"][i],
                    "max_degree": self.metrics["max_degree"][i],
                    "max_eccentricity": self.metrics["max_eccentricity"][i],
                    "max_closeness": self.metrics["max_closeness"][i],
                    "max_betweenness": self.metrics["max_betweenness"][i],
                    "max_edge_betweenness": self.metrics["max_edge_betweenness"][i],
                    "std_degree": self.metrics["std_degree"][i],
                    "std_eccentricity": self.metrics["std_eccentricity"][i],
                    "std_closeness": self.metrics["std_closeness"][i],
                    "std_betweenness": self.metrics["std_betweenness"][i],
                    "std_edge_betweenness": self.metrics["std_edge_betweenness"][i],
                    "variance_degree": self.metrics["variance_degree"][i],
                    "variance_eccentricity": self.metrics["variance_eccentricity"][i],
                    "variance_closeness": self.metrics["variance_closeness"][i],
                    "variance_betweenness": self.metrics["variance_betweenness"][i],
                    "variance_edge_betweenness": self.metrics["variance_edge_betweenness"][i],
                })

        graph_dir = os.path.join(save_dir, "graphs")
        os.makedirs(graph_dir, exist_ok=True)

        for i, serialized_graph in enumerate(self.metrics["serialized_graph"]):
            graph_file_path = os.path.join(graph_dir, f"graph_{i}.graphml")
            try:
                deserialized_graph = self.deserialize_graph(serialized_graph)
                with open(graph_file_path, "wb") as f:
                    nx.write_graphml(deserialized_graph, f)
            except Exception as e:
                print(f"Error saving graph {i}: {e}")
        self.main_window.saved_label.setText(f"Graphs and results saved successfully in {graph_dir}!")


    def deserialize_graph(self, byte_data):
        buffer = BytesIO(byte_data)
        buffer.seek(0)
        return pickle.load(buffer)
    
    def load_graphs_from_main_menu(self):
        self.main_window.save_results_button.hide()
        self.main_window.saved_label.setText("")
        self.load_and_display_graphs()

    def load_and_display_graphs(self, folder_path=None):
        if folder_path is None:
            folder_path = Qt.QtWidgets.QFileDialog.getExistingDirectory(
                self.main_window, "Select Folder", self.load_default_graph_directory()
            )
            if not folder_path:
                return

        png_files = [f for f in os.listdir(folder_path) if f.endswith(".png")]
        if not png_files:
            self.main_window.show_warning("No Graphs Found", "The selected folder does not contain any PNG files.")
            return

        self.showChooseResultsPage()
        self.main_window.previous_page = self.main_window.current_page
        scroll_content = self.main_window.scrollAreaWidgetContents_2
        layout = scroll_content.layout()
        if layout is None:
            layout = QVBoxLayout(scroll_content)
            scroll_content.setLayout(layout)
        else:
            self.clear_layout(layout)
        scroll_area_width = self.main_window.scrollArea_2.width()

        for filename in png_files:
            graph_path = os.path.join(folder_path, filename)
            label = Qt.QtWidgets.QLabel()
            pixmap = Qt.QtGui.QPixmap(graph_path)
            scaled_pixmap = pixmap.scaled(
                scroll_area_width - 20, pixmap.height(),
                Qt.QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                Qt.QtCore.Qt.TransformationMode.SmoothTransformation
            )
            label.setPixmap(scaled_pixmap)
            label.setScaledContents(False)
            label.setAlignment(Qt.QtCore.Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(label)
            layout.addSpacing(15)

        scroll_content.adjustSize()

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