import PyQt6 as Qt
from src.trainer import Trainer
from PyQt6.QtGui import QMovie

from PyQt6.QtCore import Qt
from PyQt6.QtCore import QTimer

#TODO To make this class less complicated, seperate the trainings into a new class later. This class is too long and complicated. The new class name should be something like MasterModelTrainer or ModelTrainingHandler.
class PageNavigator:
    def __init__(self, main_window):
        self.main_window = main_window
        self.trainer = None
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

    def showDatasetPage(self):
        self.fadeToPage(self.main_window.dataset_page)
        self.main_window.previous_page = self.main_window.current_page
        self.main_window.current_page = "Dataset"
        self.main_window.restart_button.show()
        self.resetSettingsButton()
        #If i press that the training button, read through the table OR the self.main_window.neural_networks and parse through it to train. save progress! save model should it get interrupted
        self.main_window.model_train_button.setText("Start Training")
        self.main_window.model_train_button.clicked.connect(self.parseThroughProcessesTable)

    def parseThroughProcessesTable(self):
        data = self.main_window.reorder_table_view2.model().get_table_data()
        print("Data from pruning table:", data)


        for row in data:
            pass
    


    def mainTrainLoop(self):
        self.main_window.model_train_button.setEnabled(False)
        self.main_window.modify_dataset_button.setEnabled(False)
        self.main_window.model_undo_button.setEnabled(False)
        
        self.main_window.stackedWidget_2.setCurrentWidget(self.main_window.model_training_page)
        self.main_window.model_train_button.setText("...")
        self.main_window.model_train_button.disconnect()
        self.main_window.model_train_button.clicked.connect(self.showChooseResultsPage)# ????
        
        self.main_window.loading_label.setFixedSize(50, 50)
        movie = QMovie("./resources/icons/loading.gif")
        self.main_window.loading_label.setVisible(True)
        self.main_window.loading_label.raise_() #TODO simplify later
        self.main_window.loading_label.setMovie(movie)
        movie.start()
        self.main_window.loading_label.show()
        self.main_window.loading_label.show()
        self.main_window.repaint()

        # The NN gets filled at the stage of clicking on "Done", in place of the previous training button. It redirects to the process timeline
        for row in self.main_window.neural_network:
            self.trainOneModel(row)




    def trainOneModel(self, modelrow): # Already have the data. Its job is not to format the data to the interpretable type for the trainer. Nor give default values
        #Paste this here later. I suggest going through the pages first, the rest will come to you later
        model = modelrow[0] # Will not be correct by ordering because of templatetable
        start =  modelrow[1]
        dataset = modelrow[2]
        N = modelrow[3]
        loss = modelrow[4]
        optimizer = modelrow[5]
        epochs = modelrow[6]
        k = modelrow[7]
        p = modelrow[8]
        batch_size = modelrow[9]
        learning_rate = modelrow[10]
        graph_type = modelrow[11]
        
        if start == "Prior":
            #Dont forget to make number of layers or number of nodes or N the same
            # self.trainer???
            trainer = Trainer(model, dataset, hidden_sizes= N, loss=loss, optimizer=optimizer, epochs=epochs, k=k, p=p, batch_size=batch_size, learning_rate=learning_rate, graph_type=graph_type)
            trainer.message.connect(self.updateTrainingProcessLabel)
            trainer.load_data_and_create_graph()
            trainer.start()
            trainer.finished.connect(self.onTrainingFinished)
        else:
            trainer = Trainer(model, dataset, hidden_sizes=N, loss=loss, optimizer=optimizer, epochs=epochs, graph_type=graph_type, batch_size=batch_size)
            trainer.message.connect(self.updateTrainingProcessLabel)
            trainer.load_data_and_create_graph()
            self.handleReadingPruningTable(self.main_window.reorder_table_view.model().get_table_data(), trainer.model)

        #TODO make a functionality for Undo button between model stages



    def showModelPage(self):
        self.fadeToPage(self.main_window.model_page)
        self.main_window.previous_page = self.main_window.current_page
        self.main_window.current_page = "Model"
        self.main_window.restart_button.show()

        self.resetSettingsButton()

    def showChooseResultsPage(self):
        self.fadeToPage(self.main_window.choose_results_page)
        self.main_window.previous_page = self.main_window.current_page
        self.main_window.current_page = "Results"
        self.main_window.restart_button.show()
        self.resetSettingsButton()

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
        elif self.main_window.previous_page == "Dataset":
            self.main_window.settings_button.clicked.connect(self.showDatasetPage)
        else:
            print("Error: No previous page found")

    def showModelStartingPage(self):
        #self.fadeToPage(self.main_window.model_starting_page)
        self.main_window.stackedWidget_2.setCurrentWidget(self.main_window.model_starting_page)
        self.main_window.model_train_button.setText("Continue")
        if self.main_window.model_train_button.signalsBlocked():
            self.main_window.model_train_button.disconnect()
        self.main_window.model_train_button.clicked.connect(self.modelTransitioner)

    def modelTransitioner(self):
        if self.main_window.GLOBAL_CHOSEN_START == "Full":
            self.showPruningStartPage()
        else:
            self.showModelPriorPage()


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

    def showModelTrainingPage(self):
        self.main_window.model_train_button.setEnabled(False)
        self.main_window.modify_dataset_button.setEnabled(False)
        self.main_window.model_undo_button.setEnabled(False)
        self.main_window.chosen_dataset = self.main_window.input_dataset.currentText()

        self.main_window.stackedWidget_2.setCurrentWidget(self.main_window.model_training_page)
        self.main_window.model_train_button.setText("...")
        self.main_window.model_train_button.disconnect()
        self.main_window.model_train_button.clicked.connect(self.showChooseResultsPage)

        self.main_window.loading_label.setFixedSize(50, 50)
        movie = QMovie("./resources/icons/loading.gif")
        self.main_window.loading_label.setVisible(True)
        self.main_window.loading_label.raise_()  # Ensure it's on top so it wont disappear behind other widgets

        self.main_window.loading_label.setMovie(movie)
        movie.start()
        self.main_window.loading_label.show()


        # Show the label and refresh UI
        self.main_window.loading_label.show()
        self.main_window.repaint()



        #TODO make a functionality for Undo button between model stages

        if self.main_window.GLOBAL_CHOSEN_START == "Prior":
            self.main_window.update_variable(self.main_window.input_prior_numofnodes, 'number_of_nodes')
            #self.main_window.update_variable(self.main_window.input_prior_numoflayer, 'number_of_layers')
            self.main_window.update_variable(self.main_window.input_prior_optimizer, 'optimizer')
            self.main_window.update_variable(self.main_window.input_prior_lr, 'learning_rate')
            self.main_window.update_variable(self.main_window.input_prior_epochs, 'epochs')
            self.main_window.update_variable(self.main_window.input_prior_k, 'k')
            self.main_window.update_variable(self.main_window.input_prior_p, 'p')
            self.main_window.update_variable(self.main_window.input_prior_loss, 'loss_function')
            self.main_window.update_variable(self.main_window.input_prior_batch, 'batch_size')
            
            num_layers = int(self.main_window.number_of_layers)

            if isinstance(self.main_window.number_of_nodes, list):
                layer_sizes = self.main_window.number_of_nodes
            elif isinstance(self.main_window.number_of_nodes, int):
                layer_sizes = [self.main_window.number_of_nodes] * num_layers
            else:
                layer_sizes_input = self.main_window.number_of_nodes.strip()
                if ',' in layer_sizes_input:
                    layer_sizes = [int(node) for node in layer_sizes_input.split(',') if node.strip()]
                elif layer_sizes_input == "":
                    layer_sizes = [6, 6, 6] #default value for the model
                    num_layers = 3
                    self.main_window.number_of_layers = 3
                    self.main_window.number_of_nodes = 250
                else:
                    layer_sizes = [int(layer_sizes_input)] * num_layers
            # error!

            #if len(layer_sizes) != num_layers:
            #    raise ValueError(f"Number of nodes ({len(layer_sizes)}) does not match the number of layers ({num_layers}).")

            self.main_window.hidden_sizes = [int(self.main_window.number_of_nodes) for _ in range(int(self.main_window.number_of_layers))]
            learning_rate = self.main_window.learning_rate
            
            if isinstance(self.main_window.input_prior_epochs, str) and self.main_window.input_prior_epochs.strip() != '':
                self.main_window.epochs = int(self.main_window.input_prior_epochs)
            else:
                self.main_window.epochs = 10

            if isinstance(learning_rate, str) and learning_rate.strip() != '':
                learning_rate = float(learning_rate)
            else:
                learning_rate = 0.001 

            p_value = self.main_window.p
            if isinstance(p_value, str) and p_value.strip() != '':
                p_value = float(p_value)
            else:
                p_value = 0.5
            if self.main_window.k == "":
                self.main_window.k = 2
            if isinstance(self.main_window.k, str) and self.main_window.k.strip() != '':
                self.main_window.k = int(self.main_window.k)
            else:
                self.main_window.k = 2

            if self.main_window.batch_size == "" or self.main_window.batch_size == 0:
                self.main_window.batch_size = 32


            #for i in range(self.main_window.k):
            #    self.trainers.append(Trainer(self.main_window.GLOBAL_CHOSEN_MODEL, self.main_window.GLOBAL_CHOSEN_DATASET, layer_sizes, lr=learning_rate, loss=self.main_window.loss_function, optimizer=self.main_window.optimizer, epochs=self.main_window.epochs, k=self.main_window.k, p=p_value))
            self.trainer = Trainer(self.main_window.GLOBAL_CHOSEN_MODEL, self.main_window.chosen_dataset, hidden_sizes=layer_sizes, lr=learning_rate, loss=self.main_window.loss_function, optimizer=self.main_window.optimizer, epochs=self.main_window.epochs, k=self.main_window.k, p=p_value, graph_type="WS", N=self.main_window.number_of_nodes, batch_size=self.main_window.batch_size)
            
            self.trainer.message.connect(self.updateTrainingProcessLabel)
            self.trainer.load_data_and_create_graph()
            # start training 
            self.trainer.start()

            self.trainer.finished.connect(self.onTrainingFinished)



        elif self.main_window.GLOBAL_CHOSEN_START == "Full":
            self.main_window.update_variable(self.main_window.input_prior_numofnodes_2, 'number_of_nodes')
            self.main_window.update_variable(self.main_window.input_prior_numoflayer_2, 'number_of_layers')
            self.main_window.update_variable(self.main_window.input_prior_optimizer_2, 'optimizer')
            self.main_window.update_variable(self.main_window.input_prior_lr_2, 'learning_rate') #the conversion is not good. always raises exceptions
            self.main_window.update_variable(self.main_window.input_prior_epochs_2, 'epochs')
            self.main_window.update_variable(self.main_window.input_prior_loss_2, 'loss_function')
            self.main_window.update_variable(self.main_window.input_prior_batch_2, 'batch_size')

            num_layers = int(self.main_window.number_of_layers)
            if isinstance(self.main_window.number_of_nodes, list):
                self.main_window.number_of_nodes = int(self.main_window.number_of_nodes.strip())

            layer_sizes_input = self.main_window.number_of_nodes
            if layer_sizes_input == "":
                self.main_window.hidden_sizes = [6, 6, 6] #default value for the model
            else:
                self.main_window.hidden_sizes = [int(self.main_window.number_of_nodes)] * num_layers
            #self.main_window.hidden_sizes = [int(self.main_window.number_of_nodes) for _ in range(int(self.main_window.number_of_layers))]


            print(self.main_window.hidden_sizes) # [6,6,6]
            if self.main_window.batch_size == "" or self.main_window.batch_size == 0:
                self.main_window.batch_size = 32


            #will need to go over teh pruning settings of the model and add them here
            print(self.main_window.hidden_sizes, self.main_window.loss_function, self.main_window.optimizer, self.main_window.epochs, self.main_window.batch_size, self.main_window.GLOBAL_CHOSEN_START, self.main_window.chosen_dataset, self.main_window.GLOBAL_CHOSEN_MODEL)
            self.trainer = Trainer(self.main_window.GLOBAL_CHOSEN_MODEL, self.main_window.chosen_dataset,  hidden_sizes=self.main_window.hidden_sizes, loss=self.main_window.loss_function, optimizer=self.main_window.optimizer, epochs=self.main_window.epochs, graph_type=self.main_window.GLOBAL_CHOSEN_START, batch_size=int(self.main_window.batch_size))
            
            self.trainer.message.connect(self.updateTrainingProcessLabel)
            self.trainer.load_data_and_create_graph()

            self.handleReadingPruningTable(self.main_window.reorder_table_view.model().get_table_data(), self.trainer.model)





    def updateTrainingProcessLabel(self, message):
        previous_text = self.main_window.training_process_label.text()

        self.main_window.training_process_label.setText(previous_text + "\n" + message)


    def handleReadingPruningTable(self, data, model):
        print("Data from pruning table:", data)
        self.main_window.pruning_data = data
        # TODO change the page too
        
        for row in data:
            action = row[0]
            if action == "Prune": 
                scope = row[1]
                layer = row[2]
                prune_ratio = float(row[3]) / 100
                prune_method = row[4]
                print(f"Pruning {layer} with ratio {prune_ratio}% using {prune_method} method.")

                if prune_method == "Magnitude": #TODO make sure this goes over the methods of all prunings by classes of pruner.py
                    print("Magnitude Pruning")
                    self.trainer.magnitude_prune(prune_ratio, layer)

                elif prune_method == "Random":
                    print("Random Pruning")
            elif action == "Retrain":
                epochs = row[5]
                learning_rate = row[6]
                print(f"Training {layer} for {epochs} epochs with learning rate {learning_rate}.")
                self.trainer.epochs = int(epochs)
                self.trainer.lr = float(learning_rate)
                self.trainer.optimizer = self.main_window.optimizer
                self.trainer.start()
                    
            


    def onTrainingFinished(self):
        print("Training Finished")
        
        self.main_window.loading_label.hide()
        self.main_window.model_train_button.setEnabled(True)
        self.main_window.model_train_button.setText("Training Completed")
        
        if self.main_window.GLOBAL_CHOSEN_START == None:
            self.main_window.model_train_button.disconnect()
            #self.main_window.model_train_button.clicked.connect(self.showChooseResultsPage)
        self.main_window.modify_dataset_button.setEnabled(True)
        
        self.trainer.wait()



    
    def update_button_state(self): #TODO make other mechanism for this!!!
        #if self.main_window.GLOBAL_STAGE == 1: # havent started training yet and the model is not chosen yet
        if self.main_window.GLOBAL_CHOSEN_MODEL is not None and self.main_window.GLOBAL_CHOSEN_START is not None:
            self.main_window.model_train_button.setEnabled(True)
            self.main_window.current_model_architecture_label.setText(self.main_window.GLOBAL_CHOSEN_MODEL)
            self.main_window.current_model_architecture_label_2.setText(self.main_window.GLOBAL_CHOSEN_MODEL)
            self.main_window.current_model_architecture_label_3.setText(self.main_window.GLOBAL_CHOSEN_MODEL)

            #self.main_window.GLOBAL_STAGE = 2
        else: # unless the model is chosen, disable the button
            self.main_window.model_train_button.setEnabled(False)
        
        # elif self.main_window.GLOBAL_STAGE == 2:
        """
            if self.main_window.GLOBAL_CHOSEN_MODEL is not None and self.main_window.GLOBAL_CHOSEN_START is not None:
                self.main_window.model_train_button.setEnabled(True)
                self.GLOBAL_STAGE = 2
            else:
                self.main_window.model_train_button.setEnabled(False)
        """