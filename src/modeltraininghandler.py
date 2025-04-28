from PyQt6.QtGui import QMovie
from src.trainer import Trainer
from PyQt6 import QtWidgets
from PyQt6 import QtCore
import os

class ModelTrainingHandler:
    def __init__(self, main_window):
        self.main_window = main_window
        self.trainer = None
        self.trainers = []
        self.current_model_index = 0
        self.action_queue = []
        
    def setup_training_UI(self):
        self.main_window.model_train_button.setEnabled(False)
        self.main_window.undo_button.setEnabled(False)
        self.main_window.stackedWidget_2.setCurrentWidget(self.main_window.model_training_page)
        self.main_window.model_train_button.setText("...")
        self.main_window.model_train_button.disconnect()
        self.main_window.model_train_button.clicked.connect(self.main_window.page_navigation_handler.show_choose_results_page)
        self.show_loading_animation()

    def reset_UI(self):
        self.main_window.model_train_button.setEnabled(True)
        self.main_window.undo_button.setEnabled(True)   
        self.main_window.model_train_button.setText("Start Training")
        try:
            self.main_window.model_train_button.clicked.disconnect()
        except TypeError:
            pass
        self.main_window.model_train_button.clicked.connect(self.parse_through_processes_table)
    
    def show_loading_animation(self):
        self.main_window.loading_label.setFixedSize(50, 50)
        movie = QMovie("./resources/icons/loading.gif")
        self.main_window.loading_label.setVisible(True)
        self.main_window.loading_label.setMovie(movie)
        movie.start()
        self.main_window.loading_label.show()

    def parse_through_processes_table(self):
        self.main_window.model_train_button.setEnabled(False)
        self.main_window.undo_button.setEnabled(False)
        self.main_window.page_navigation_handler.save_pruning_changes_and_goback()

        data=self.main_window.reorder_table_view2.model().get_table_data()
        if len(data)<1:
            self.main_window.show_warning(
                title="No Data Found",
                message="The table is empty. Please add data before proceeding.",
                actions=None,
                buttons=["ok"]
            )
            return

        if not self.validate_table_data(self.main_window.reorder_table_view2.model()):
            return

        print("Data from pruning table:", data)
        try:
            for row in data:
                if row and len(row)>0:
                    success=self.process_table_row(row)
                    if not success:
                        self.reset_UI()
                        return # <-- Important: stop if invalid row

            self.current_model_index=0
            self.main_train_loop()

        except Exception as e:
            print(f"Error processing table data: {str(e)}")
            self.main_window.show_warning(
                title="Invalid Data",
                message=f"Error processing table data: {str(e)}",
                actions=None,
                buttons=["ok"]
            )
            self.reset_UI()
            return

    def validate_table_data(self, table_model):
        for row_index in range(table_model.rowCount()):
            try:
                row_data = [
                    table_model.index(row_index, col).data() for col in range(table_model.columnCount())
                ]
                self.extract_training_parameters(row_data[2:])
            except Exception as e:
                self.main_window.show_warning(
                    title="Invalid Data",
                    message=f"Row {row_index + 1} contains invalid data: {str(e)}",
                    actions=None,
                    buttons=["ok"]
                )
                return False
        return True

    def process_table_row(self, row):
        try:
            index=row[0]
            if not index:
                raise ValueError("Index is missing or invalid.")

            model=row[1]
            valid_models=["MLP", "LSTM"]
            if not model or model not in valid_models:
                raise ValueError("Model type is missing or invalid.")

            start=row[2]
            valid_starts=["Prior", "Prune"]
            if not start or start not in valid_starts:
                raise ValueError("Start type is missing or invalid.")

            dataset=row[3]
            valid_datasets=["MNIST", "CIFAR10", "CIFAR100"]
            if not dataset or dataset not in valid_datasets:
                raise ValueError("Dataset is missing or invalid.")

            N=self.parse_n_value(row[4], start)
            if not N:
                raise ValueError("N value is missing or invalid.")

            loss=row[5]
            valid_losses=["CrossEntropy", "MSE"]
            if not loss or loss not in valid_losses:
                raise ValueError("Loss function is missing or invalid.")

            optimizer=row[6]
            valid_optimizers=["SGD", "Adam", "RMSprop", "Adagrad", "Adadelta"]
            if not optimizer or optimizer not in valid_optimizers:
                raise ValueError("Optimizer is missing or invalid.")

            epochs=row[7]
            if not epochs or not str(epochs).isdigit():
                raise ValueError("Epochs value is missing or invalid.")
            epochs=int(epochs)

            k=row[8]
            if start != "Prune":
                if not k or not str(k).isdigit():
                    raise ValueError("K value is missing or invalid.")
                k=int(k)

            p=row[9]
            if start != "Prune":
                try:
                    p=float(p)
                except (ValueError, TypeError):
                    raise ValueError("P value is missing or invalid.")

            batch_size=row[10]
            if not batch_size or not str(batch_size).isdigit():
                raise ValueError("Batch size is missing or invalid.")
            batch_size=int(batch_size)

            learning_rate=row[11]
            try:
                learning_rate=float(learning_rate)
            except (ValueError, TypeError):
                raise ValueError("Learning rate is missing or invalid.")

            graph_type=row[12]
            valid_graph_types=["Full", "WS", "BA"]
            if not graph_type or graph_type not in valid_graph_types:
                raise ValueError("Graph type is missing or invalid.")

            self.main_window.neural_networks.append([
                index, model, start, dataset, N, loss, optimizer,
                epochs, k, p, batch_size, learning_rate, graph_type
            ])
            return True

        except ValueError as ve:
            self.main_window.show_warning(
                title="Invalid Row Data",
                message=f"Error processing row {row[0]}: {str(ve)}",
                actions=None,
                buttons=["ok"]
            )
            return False

        except Exception as e:
            print(f"Error processing row {row}: {str(e)}")
            self.main_window.show_warning(
                title="Invalid Data",
                message=f"Row {row} contains invalid data {row[0]}: {str(e)}",
                actions=None,
                buttons=["ok"]
            )
            return False

    def parse_n_value(self, n_value, start_type):
        if n_value == "":
            return [6,6,6] if start_type == "Prune" else 250
            
        if start_type == "Prune":
            if "[" in n_value and "]" in n_value:
                return [int(i) for i in n_value[1:-1].split(",") if i.strip() != ""]
            elif "," in n_value: 
                return [int(i) for i in n_value.split(",") if i.strip() != ""]
            else:
                return [int(n_value)] * int(n_value)
        else:
            return int(n_value)

    def main_train_loop(self):
        self.main_window.page_navigation_handler.show_model_page()
        self.setup_training_UI()
        #print(self.main_window.neural_networks)
        if self.current_model_index < len(self.main_window.neural_networks):
            self.train_one_model(self.main_window.neural_networks[self.current_model_index])

    def train_one_model(self, modelrow):
        params = self.extract_training_parameters(modelrow)
        
        if modelrow[2] == "Prior":
            self.train_prior_model(params)
        else:
            self.train_pruned_model(params)

    def extract_training_parameters(self, modelrow):
        return {
            'index': int(modelrow[0]),
            'model': modelrow[1],
            'start': modelrow[2],
            'dataset': modelrow[3],
            'N': modelrow[4],
            'loss': modelrow[5],
            'optimizer': modelrow[6],
            'epochs': modelrow[7],
            'k': modelrow[8],
            'p': modelrow[9],
            'batch_size': modelrow[10],
            'learning_rate': modelrow[11],
            'graph_type': modelrow[12]
        }

    def train_prior_model(self, params):
        self.trainer = Trainer(
            params['model'], params['dataset'], 
            hidden_sizes=params['N'], loss=params['loss'], 
            optimizer=params['optimizer'], epochs=params['epochs'], 
            k=params['k'], p=params['p'], batch_size=params['batch_size'], 
            lr=params['learning_rate'], graph_type=params['graph_type'], 
            index=params['index']
        )
        self.trainer.message.connect(self.update_training_process_label)
        self.trainer.load_data_and_create_graph()
        self.trainer.start()
        self.trainer.finished.connect(self.on_training_finished)

    def train_pruned_model(self, params):
        self.trainer = Trainer(
            params['model'], params['dataset'], 
            hidden_sizes=params['N'], loss=params['loss'], 
            optimizer=params['optimizer'], epochs=params['epochs'], 
            graph_type=params['graph_type'], batch_size=params['batch_size'], 
            index=params['index']
        )
        
        self.trainer.message.connect(self.update_training_process_label)
        self.trainer.load_data_and_create_graph()
        self.handle_reading_pruning_table(self.trainer)
        self.trainer.finished.connect(self.on_training_finished)


    def update_training_process_label(self, message):
        previous_text = self.main_window.training_process_label.text()

        self.main_window.training_process_label.setText(previous_text + "\n" + message)

    def on_training_finished(self):
        if self.action_queue and len(self.action_queue) > 0:
            self.process_next_action()
            return


        #Printage
        self.trainer.message.emit#   (f"Final accuracy: {self.trainer.training_metrics.get('final_train_accuracy'):.4f}, Validation accuracy: {self.training_metrics.get('final_val_accuracy'):.4f}")

        print(f"\n=========\nTraining for model {self.current_model_index + 1} finished\n=========\n")
        self.trainer.message.emit(f"\n=========\nTraining for model {self.current_model_index + 1} finished\n=========\n")

        self.current_model_index += 1
        self.main_window.previous_results.append(self.trainer.get_state())
        if self.current_model_index < len(self.main_window.neural_networks):
            self.train_one_model(self.main_window.neural_networks[self.current_model_index])
            pass
        else: # Training finalized
            print("All models training completed")
            self.main_window.loading_label.hide()
            self.main_window.results_handler.visualize_results(self.main_window.previous_results)
            self.main_window.model_train_button.setEnabled(True)
            self.main_window.model_train_button.setText("Show Results")
            self.main_window.save_results_button.show()


    def handle_reading_pruning_table(self, model):
        try:
            hidden_data = self.main_window.timelineTableModel.get_hidden_data(model.index - 2)
            if hidden_data == '' or hidden_data is None:
                self.trainer.message.emit("No hidden data for model. Skipping pruning actions.")
                self.trainer.finished.emit()
                self.on_training_finished()
                return
            self.action_queue = []
            for row in hidden_data:
                row = row[2:]
                if row == '':
                    continue
                action = row[1]
                if action == "Prune":
                    self.action_queue.append(("Prune", row))
                elif action == "Retrain":
                    self.action_queue.append(("Retrain", row))
                else:
                    print(f"Unknown action: {action}. Skipping this action.")

            if not self.action_queue:
                self.trainer.finished.emit()
                self.on_training_finished()
                return

            self.process_next_action()
        except Exception as e:
            print(f"Error reading pruning table: {str(e)}")
            self.trainer.message.emit(f"Error reading pruning table: {str(e)}")
            self.trainer.finished.emit()
            self.on_training_finished()
            return
        
    def process_next_action(self):
        if not self.action_queue or len(self.action_queue) == 0 or self.trainer is None:
            return
        action, row = self.action_queue.pop(0)
        if action == "Prune":
            self.handle_prune_action(row)
        elif action == "Retrain":
            self.handle_retrain_action(row)
        

    def handle_prune_action(self, row):
        layer = row[2]
        prune_ratio = float(row[3]) / 100
        prune_method = row[4]

        self.trainer.async_prune(prune_ratio, layer, prune_method)
        self.process_next_action()

    def handle_retrain_action(self, row):
        epochs = row[5]
        learning_rate = row[6]
        self.trainer.epochs = int(epochs)
        self.trainer.lr = float(learning_rate)
        self.trainer.start()
