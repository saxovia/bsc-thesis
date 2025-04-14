
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
        
    def setupTrainingUI(self):
        self.main_window.model_train_button.setEnabled(False)
        self.main_window.undo_button.setEnabled(False)
        self.main_window.stackedWidget_2.setCurrentWidget(self.main_window.model_training_page)
        self.main_window.model_train_button.setText("...")
        self.main_window.model_train_button.disconnect()
        self.main_window.model_train_button.clicked.connect(self.main_window.page_navigation_handler.showChooseResultsPage)
        self.showLoadingAnimation()

    def resetUI(self):
        self.main_window.model_train_button.setEnabled(True)
        self.main_window.undo_button.setEnabled(True)
        self.main_window.model_train_button.setText("Start Training")
        self.main_window.model_train_button.disconnect()
        self.main_window.model_train_button.clicked.connect(self.parseThroughProcessesTable)
    
    def showLoadingAnimation(self):
        self.main_window.loading_label.setFixedSize(50, 50)
        movie = QMovie("./resources/icons/loading.gif")
        self.main_window.loading_label.setVisible(True)
        self.main_window.loading_label.setMovie(movie)
        movie.start()
        self.main_window.loading_label.show()

    def parseThroughProcessesTable(self):
        self.main_window.model_train_button.setEnabled(False)
        self.main_window.undo_button.setEnabled(False)

        data = self.main_window.reorder_table_view2.model().get_table_data()
        print("Data from pruning table:", data)

        for row in data:
            self.processTableRow(row)

        self.current_model_index = 0
        self.mainTrainLoop()

    def processTableRow(self, row):
        index = row[0]
        model = row[1] if row[1] != "" else "MLP"
        start = row[2] if row[2] != "" else "Prior"
        dataset = row[3] if row[3] != "" else "MNIST"
        N = self.parseNValue(row[4], start)
        loss = row[5] if row[5] != "" else "CrossEntropy"
        optimizer = row[6] if row[6] != "" else "Adam"
        epochs = int(row[7]) if row[7] != "" else 30
        k = int(row[8]) if row[8] != "" else 2 
        p = float(row[9]) if row[9] != "" else 0.5 
        batch_size = int(row[10]) if row[10] != "" else 64 
        learning_rate = float(row[11]) if row[11] != "" else 0.001
        graph_type = row[12] if row[12] != "" else "WS"

        self.main_window.neural_networks.append([
            index, model, start, dataset, N, loss, optimizer, 
            epochs, k, p, batch_size, learning_rate, graph_type
        ])

    def parseNValue(self, n_value, start_type):
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

    def mainTrainLoop(self):
        self.main_window.page_navigation_handler.showModelPage()
        self.setupTrainingUI()
        print(self.main_window.neural_networks)
        if self.current_model_index < len(self.main_window.neural_networks):
            self.trainOneModel(self.main_window.neural_networks[self.current_model_index])

    def trainOneModel(self, modelrow):
        params = self.extractTrainingParameters(modelrow)
        
        if modelrow[2] == "Prior":
            self.trainPriorModel(params)
        else:
            self.trainPrunedModel(params)

    def extractTrainingParameters(self, modelrow):
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

    def trainPriorModel(self, params):
        self.trainer = Trainer(
            params['model'], params['dataset'], 
            hidden_sizes=params['N'], loss=params['loss'], 
            optimizer=params['optimizer'], epochs=params['epochs'], 
            k=params['k'], p=params['p'], batch_size=params['batch_size'], 
            lr=params['learning_rate'], graph_type=params['graph_type'], 
            index=params['index']
        )
        self.trainer.message.connect(self.updateTrainingProcessLabel)
        self.trainer.load_data_and_create_graph()
        self.trainer.start()
        self.trainer.finished.connect(self.onTrainingFinished)

    def trainPrunedModel(self, params):
        self.trainer = Trainer(
            params['model'], params['dataset'], 
            hidden_sizes=params['N'], loss=params['loss'], 
            optimizer=params['optimizer'], epochs=params['epochs'], 
            graph_type=params['graph_type'], batch_size=params['batch_size'], 
            index=params['index']
        )
        
        self.trainer.message.connect(self.updateTrainingProcessLabel)
        self.trainer.load_data_and_create_graph()
        self.handleReadingPruningTable(self.trainer)
        self.trainer.finished.connect(self.onTrainingFinished)


    def updateTrainingProcessLabel(self, message):
        previous_text = self.main_window.training_process_label.text()

        self.main_window.training_process_label.setText(previous_text + "\n" + message)

    def onTrainingFinished(self):
        if self.action_queue and len(self.action_queue) > 0:
            self.processNextAction()
            return
        try:
            self.trainer.print_summary()
        except Exception as e:
            print(f"Error printing summary: {str(e)}")
        #Printage
        print(f"\n=========\nTraining for model {self.current_model_index + 1} finished\n=========\n")
        self.trainer.message.emit(f"\n=========\nTraining for model {self.current_model_index + 1} finished\n=========\n")

        self.current_model_index += 1
        self.main_window.previous_results.append(self.trainer.get_state())

        if self.current_model_index < len(self.main_window.neural_networks):
            self.trainOneModel(self.main_window.neural_networks[self.current_model_index])
            pass
        else: # Training finalized
            print("All models training completed")
            self.main_window.loading_label.hide()
            self.main_window.page_navigation_handler.visualize_results()
            self.main_window.model_train_button.setEnabled(True)
            self.main_window.model_train_button.setText("Show Results")


    def handleReadingPruningTable(self, model):
        hidden_data = self.main_window.timelineTableModel.get_hidden_data(model.index-2)
        if hidden_data == '' or hidden_data is None:
            self.trainer.message.emit("No hidden data for model. Skipping pruning actions.")
            self.trainer.finished.emit()
            self.onTrainingFinished()
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
                print(f"Unknown action: {action}")

        if not self.action_queue:
            self.trainer.finished.emit()
            self.onTrainingFinished()
            return

        self.processNextAction()

    def processNextAction(self):
        if not self.action_queue or len(self.action_queue) == 0 or self.trainer is None:
            return
        action, row = self.action_queue.pop(0)
        if action == "Prune":
            self.handlePruneAction(row)
        elif action == "Retrain":
            self.handleRetrainAction(row)
        

    def handlePruneAction(self, row):
        layer = row[2]
        prune_ratio = float(row[3]) / 100
        prune_method = row[4]
        #print(f"Pruning {layer} with ratio {prune_ratio}% using {prune_method} method.")

        if prune_method == "Magnitude":
            #print("Magnitude Pruning")
            self.trainer.magnitude_prune(prune_ratio, layer)
        elif prune_method == "Random":
            pass #TODO make sure this goes over the methods of all prunings by classes of pruner.py
            #print("Random Pruning")
        self.processNextAction()

    def handleRetrainAction(self, row):
        epochs = row[5]
        learning_rate = row[6]
        #print(f"Training {layer} for {epochs} epochs with learning rate {learning_rate}.")
        self.trainer.epochs = int(epochs)
        self.trainer.lr = float(learning_rate)
        self.trainer.start()
        #self.trainer.finished.connect(self.processNextAction)

    def saveModel(self, model):
        if hasattr(self, 'trainer') and self.trainer is not None:
            self.trainer.running = False
            self.trainer.quit()
            self.trainer.wait()

        temp_neural_networks = self.main_window.neural_networks.copy()
        self.main_window.neural_networks = []
        temp_trainer = self.trainer.get_state()

        # Get default path from settings.txt
        settings_file = os.path.join(os.path.dirname(__file__), "..\settings.txt")
        fallback_path = os.path.join(os.path.dirname(__file__), "..\savedmodels")
        
        default_dir = fallback_path
        
        try:
            with open(settings_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        if not os.path.isabs(line):
                            line = os.path.join(os.path.dirname(__file__), line)
                        if os.path.isdir(line):
                            default_dir = line
                        break
        except (FileNotFoundError, IOError) as e:
            print(f"Note: Using fallback path ({fallback_path}) because: {str(e)}")
        
        os.makedirs(default_dir, exist_ok=True)
        
        timestamp = QtCore.QDateTime.currentDateTime().toString('yyyyMMdd_hhmmss')
        default_name = os.path.join(default_dir, f"model_{timestamp}.pt")
        
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.main_window,
            "Save Model",
            default_name,
            "PyTorch Model Files (*.pt);;All Files (*)"
        )
        
        if not file_path:
            self.main_window.neural_networks = temp_neural_networks.copy()
            return
        if not file_path.endswith('.pt'):
            file_path += '.pt'
        try:
            if self.trainer.running:
                self.trainer.stop()
            self.trainer.save_model(file_path, self.main_window.neural_networks)
            print("Model saved successfully!")
        except Exception as e:
            print(f"Failed to save model:\n{str(e)}")

        self.main_window.neural_networks = temp_neural_networks.copy()
        self.trainer = Trainer("MLP", "MNIST")
        self.trainer.set_state(temp_trainer)
        self.trainer.start()

    def loadModel(self, model):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.main_window,
            "Load Model",
            "",
            "PyTorch Model Files (*.pt);;All Files (*)"
        )
        
        if not file_path:
            return
        
        try:
            self.trainer = Trainer("MLP", "MNIST")
            self.trainer.load_data_and_create_graph()
            data = self.trainer.load_model(file_path, self.main_window.neural_networks)
            
            if data and len(data) >= 2:
                self.main_window.neural_networks = data[0]
                self.main_window.current_model_index = data[1]
                print("Model loaded successfully!")
            else:
                print("Invalid model file format")
                
        except Exception as e:
            self.main_window.neural_networks =[]
            self.main_window.current_model_index = -1
            print(f"Failed to load model:\n{str(e)}")
            #self.show_message("Error", f"Failed to load model:\n{str(e)}", "critical")