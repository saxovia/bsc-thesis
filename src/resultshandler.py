import os
from datetime import datetime
import csv
from io import BytesIO
import pickle
import matplotlib.pyplot as plt
from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtWidgets import QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import networkx as nx


class ResultsHandler:
    def __init__(self, main_window):
        self.main_window = main_window
        self.metrics = self._initialize_metrics_dict()
        
    def _initialize_metrics_dict(self):
        return {
            "train_loss": [], "train_accuracy": [], "val_loss": [], "val_accuracy": [],
            "global_sparsity": [], "total_parameters": [], "trainable_parameters": [],
            "layer_sparsity": [], "feature_size": [], "sequence_length": [],
            "input_size": [], "num_classes": [], "dataset_type": [], "model_type": [],
            "graph_type": [], "prune_type": [], "prune_mode": [], "degree": [],
            "eccentricity": [], "closeness": [], "betweenness": [], "edge_betweenness": [],
            "hidden_sizes": [], "lr": [], "batch_size": [], "k": [], "p": [],
            "optimizer_type": [], "loss": [], "serialized_graph": [],
            
            # Calculated metrics
            "avg_degree": [], "avg_eccentricity": [], "avg_closeness": [],
            "avg_betweenness": [], "avg_edge_betweenness": [], "avg_global_sparsity": [],
            
            "min_degree": [], "min_eccentricity": [], "min_closeness": [],
            "min_betweenness": [], "min_edge_betweenness": [], "min_global_sparsity": [],
            
            "max_degree": [], "max_eccentricity": [], "max_closeness": [],
            "max_betweenness": [], "max_edge_betweenness": [], "max_global_sparsity": [],
            
            "std_degree": [], "std_eccentricity": [], "std_closeness": [],
            "std_betweenness": [], "std_edge_betweenness": [], "std_global_sparsity": [],
            
            "variance_degree": [], "variance_eccentricity": [], "variance_closeness": [],
            "variance_betweenness": [], "variance_edge_betweenness": [], "variance_global_sparsity": [],
        }

    def visualize_results(self, previous_results):
        if not previous_results:
            print("No results available for visualization.")
            return
            
        self.metrics = self._initialize_metrics_dict()  # Reset metrics
        
        for result in previous_results:
            self._process_result(result)
            
        # TODO: Remove test data after development
        self._add_test_data()
        
        self.display_graphs(self.metrics)

    def _process_result(self, result):
        self.metrics["model_type"].append(result["model_type"])
        self.metrics["graph_type"].append(result["graph_type"])

        # Training metrics
        training_metrics = result["training_metrics"]
        self.metrics["train_loss"].append(training_metrics["final_train_loss"])
        self.metrics["train_accuracy"].append(training_metrics["final_train_accuracy"])
        self.metrics["val_loss"].append(training_metrics["final_val_loss"])
        self.metrics["val_accuracy"].append(training_metrics["final_val_accuracy"])
        
        # Graph metrics
        graph_metrics = result["graph_metrics"]
        model_metrics = training_metrics['model_metrics']
        
        # Model metrics
        self.metrics["global_sparsity"].append(model_metrics["global_sparsity"])
        self.metrics["total_parameters"].append(model_metrics["total_parameters"])
        self.metrics["prune_type"].append(model_metrics.get("prune_type", "None"))
        self.metrics["prune_mode"].append(model_metrics.get("prune_mode", "None"))
        
        # Graph properties
        self.metrics["degree"].append(graph_metrics.get("degree", {}))
        self.metrics["eccentricity"].append(graph_metrics.get("eccentricity", {}))
        self.metrics["closeness"].append(graph_metrics.get("closeness", {}))
        self.metrics["betweenness"].append(graph_metrics.get("betweenness", {}))
        self.metrics["edge_betweenness"].append(graph_metrics.get("edge_betweenness", {}))
        
        # Training parameters
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
        
        # Calculate derived metrics
        self._calculate_derived_metrics(graph_metrics)

    def _calculate_derived_metrics(self, graph_metrics):
        # Average metrics
        self.metrics["avg_degree"].append(sum(graph_metrics["degree"].values()) / len(graph_metrics["degree"]))
        self.metrics["avg_eccentricity"].append(sum(graph_metrics["eccentricity"].values()) / len(graph_metrics["eccentricity"]))
        self.metrics["avg_closeness"].append(sum(graph_metrics["closeness"].values()) / len(graph_metrics["closeness"]))
        self.metrics["avg_betweenness"].append(sum(graph_metrics["betweenness"].values()) / len(graph_metrics["betweenness"]))
        self.metrics["avg_edge_betweenness"].append(sum(graph_metrics["edge_betweenness"].values()) / len(graph_metrics["edge_betweenness"]))

        # Min metrics
        self.metrics["min_degree"].append(min(graph_metrics["degree"].values()))
        self.metrics["min_eccentricity"].append(min(graph_metrics["eccentricity"].values()))
        self.metrics["min_closeness"].append(min(graph_metrics["closeness"].values()))
        self.metrics["min_betweenness"].append(min(graph_metrics["betweenness"].values()))
        self.metrics["min_edge_betweenness"].append(min(graph_metrics["edge_betweenness"].values()))

        # Max metrics
        self.metrics["max_degree"].append(max(graph_metrics["degree"].values()))
        self.metrics["max_eccentricity"].append(max(graph_metrics["eccentricity"].values()))
        self.metrics["max_closeness"].append(max(graph_metrics["closeness"].values()))
        self.metrics["max_betweenness"].append(max(graph_metrics["betweenness"].values()))
        self.metrics["max_edge_betweenness"].append(max(graph_metrics["edge_betweenness"].values()))

        # Standard deviation and variance
        self._calculate_variance_metrics(graph_metrics)

    def _calculate_variance_metrics(self, graph_metrics):
        def calculate_variance(values, avg):
            return sum((x - avg) ** 2 for x in values) / len(values) if len(values) > 0 else 0

        # Degree
        avg_degree = self.metrics["avg_degree"][-1]
        self.metrics["std_degree"].append(calculate_variance(graph_metrics["degree"].values(), avg_degree))
        self.metrics["variance_degree"].append(calculate_variance(graph_metrics["degree"].values(), avg_degree))

        # Eccentricity
        avg_eccentricity = self.metrics["avg_eccentricity"][-1]
        self.metrics["std_eccentricity"].append(calculate_variance(graph_metrics["eccentricity"].values(), avg_eccentricity))
        self.metrics["variance_eccentricity"].append(calculate_variance(graph_metrics["eccentricity"].values(), avg_eccentricity))

        # Closeness
        avg_closeness = self.metrics["avg_closeness"][-1]
        self.metrics["std_closeness"].append(calculate_variance(graph_metrics["closeness"].values(), avg_closeness))
        self.metrics["variance_closeness"].append(calculate_variance(graph_metrics["closeness"].values(), avg_closeness))

        # Betweenness
        avg_betweenness = self.metrics["avg_betweenness"][-1]
        self.metrics["std_betweenness"].append(calculate_variance(graph_metrics["betweenness"].values(), avg_betweenness))
        self.metrics["variance_betweenness"].append(calculate_variance(graph_metrics["betweenness"].values(), avg_betweenness))

        # Edge Betweenness
        avg_edge_betweenness = self.metrics["avg_edge_betweenness"][-1]
        self.metrics["std_edge_betweenness"].append(calculate_variance(graph_metrics["edge_betweenness"].values(), avg_edge_betweenness))
        self.metrics["variance_edge_betweenness"].append(calculate_variance(graph_metrics["edge_betweenness"].values(), avg_edge_betweenness))

    def _add_test_data(self):
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

    def display_graphs(self, metrics):
        self._setup_matplotlib_theme()
        
        scroll_content = self.main_window.scrollAreaWidgetContents_2
        layout = scroll_content.layout()
        
        if layout is None:
            layout = QVBoxLayout(scroll_content)
            scroll_content.setLayout(layout)
        else:
            self.clear_layout(layout)
        
        # Create and add all graphs
        graphs = [
            self.create_parameters_vs_accuracy_graph(metrics),
            self.create_prune_metric_graph(metrics, "eccentricity", "Mean Eccentricity"),
            self.create_prune_metric_graph(metrics, "degree", "Mean Degree"),
            self.create_prune_metric_graph(metrics, "closeness", "Mean Closeness"),
            self.create_prune_metric_graph(metrics, "betweenness", "Mean Betweenness"),
            self.create_prune_metric_graph(metrics, "edge_betweenness", "Mean Edge Betweenness")
        ]
        
        for fig in graphs:
            self._add_figure_to_layout(layout, fig)
        
        scroll_content.adjustSize()

    def _setup_matplotlib_theme(self):
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

    def _add_figure_to_layout(self, layout, figure):
        container = QtWidgets.QWidget()
        container.setMinimumSize(800, 500)
        
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(10, 10, 10, 10)
        
        canvas = FigureCanvas(figure)
        container_layout.addWidget(canvas)
        
        layout.addWidget(container)
        layout.addSpacing(15)

    def clear_layout(self, layout):
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def create_parameters_vs_accuracy_graph(self, metrics):
        fig = plt.figure(figsize=(7, 3))
        ax = fig.add_subplot(111)
        
        graph_types = ['BA', 'WS', 'Full']
        colors = {'BA': '#a386fc', 'WS': '#86a9fc', 'Full': '#86dffc'}
        
        for graph_type in graph_types:
            indices = [i for i, gt in enumerate(metrics['graph_type']) if gt == graph_type]
            params = [metrics['total_parameters'][i] for i in indices]
            accuracies = [metrics['val_accuracy'][i] for i in indices]
            
            if params:
                ax.scatter(
                    params, accuracies, 
                    color=colors[graph_type], 
                    label=f"{graph_type} (n={len(params)})", 
                    s=100
                )
        
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
                current_prune_type = (
                    metrics['prune_type'][i][-1] 
                    if isinstance(metrics['prune_type'][i], list) 
                    else metrics['prune_type'][i]
                )
                
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

        # Save metrics as CSV
        self._save_metrics_to_csv(save_dir)
        
        self._save_serialized_graphs(save_dir)
        
        self.main_window.saved_label.setText(f"Graphs and results saved successfully in {save_dir}!")

    def _save_metrics_to_csv(self, save_dir):
        csv_file_path = os.path.join(save_dir, "model_results.csv")
        with open(csv_file_path, "w", newline="") as csvfile:
            fieldnames = [
                "model_type", "graph_type", "prune_type", "global_sparsity", "total_parameters", 
                "train_loss", "train_accuracy", "val_loss", "val_accuracy",
                "degree", "eccentricity", "closeness", "betweenness", "edge_betweenness",
                "hidden_sizes", "lr", "batch_size", "k", "p", "optimizer_type",
                "dataset_type", "feature_size", "sequence_length", "input_size", "num_classes",
                "layer_sparsity", "trainable_parameters", "loss",
                "avg_degree", "avg_eccentricity", "avg_closeness", "avg_betweenness", "avg_edge_betweenness",
                "min_degree", "min_eccentricity", "min_closeness", "min_betweenness", "min_edge_betweenness",
                "max_degree", "max_eccentricity", "max_closeness", "max_betweenness", "max_edge_betweenness",
                "std_degree", "std_eccentricity", "std_closeness", "std_betweenness", "std_edge_betweenness",
                "variance_degree", "variance_eccentricity", "variance_closeness", "variance_betweenness", "variance_edge_betweenness"
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for i in range(len(self.metrics["model_type"])):
                row_data = {field: self.metrics[field][i] for field in fieldnames if field in self.metrics}
                writer.writerow(row_data)

    def _save_serialized_graphs(self, save_dir):
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

    def deserialize_graph(self, byte_data):
        buffer = BytesIO(byte_data)
        buffer.seek(0)
        return pickle.load(buffer)

    def load_graphs_from_main_menu(self):
        self.main_window.save_results_button.hide()
        self.main_window.saved_label.setText("")
        self.load_and_display_graphs()

    def load_and_display_graphs(self, folder_path=None):
        allowed_filenames = {
            "parameters_vs_accuracy.png", "mean_eccentricity.png", 
            "mean_degree.png", "mean_closeness.png", 
            "mean_betweenness.png", "mean_edge_betweenness.png"
        }

        if folder_path is None:
            folder_path = QtWidgets.QFileDialog.getExistingDirectory(
                self.main_window, "Select Folder", self.load_default_graph_directory()
            )
            if not folder_path:
                return

        png_files = [f for f in os.listdir(folder_path) if f.endswith(".png") and f in allowed_filenames]
        if not png_files:
            self.main_window.show_warning(
                title="No Allowed Graphs Found",
                message="The selected folder does not contain any allowed PNG files."
            )
            return

        self.main_window.page_navigation_handler.showChooseResultsPage()
        self.main_window.page_navigation_handler.previous_page = self.main_window.current_page
        
        scroll_content = self.main_window.scrollAreaWidgetContents_2
        layout = scroll_content.layout()
        
        if layout is None:
            layout = QVBoxLayout(scroll_content)
            scroll_content.setLayout(layout)
        else:
            self.clear_layout(layout)
            
        scroll_area_width = self.main_window.scrollArea_2.width()

        for filename in png_files:
            self._display_saved_graph(layout, folder_path, filename, scroll_area_width)

        scroll_content.adjustSize()

    def _display_saved_graph(self, layout, folder_path, filename, scroll_area_width):
        graph_path = os.path.join(folder_path, filename)
        label = QtWidgets.QLabel()
        pixmap = QtGui.QPixmap(graph_path)
        
        scaled_pixmap = pixmap.scaled(
            scroll_area_width - 20, pixmap.height(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation
        )
        
        label.setPixmap(scaled_pixmap)
        label.setScaledContents(False)
        label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        
        layout.addWidget(label)
        layout.addSpacing(15)

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