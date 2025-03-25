import PyQt6 as Qt
from src.trainer import Trainer
from PyQt6.QtGui import QMovie
import os

from PyQt6.QtCore import Qt
from PyQt6.QtCore import QTimer
#import the config file


class PageNavigator:
    def __init__(self, main_window):
        self.main_window = main_window
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
        if self.main_window.GLOBAL_CHOSEN_START == "Prune":
            self.showPruningStartPage()
            self.showModelTrainingPage()
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
        self.main_window.model_train_button.setText("Training...")
        self.main_window.GLOBAL_CHOSEN_DATASET = self.main_window.input_dataset.currentText()
        if self.main_window.GLOBAL_CHOSEN_START == "Prior":
            self.main_window.update_variable(self.main_window.input_prior_numofnodes, 'number_of_nodes')
            self.main_window.update_variable(self.main_window.input_prior_numoflayer, 'number_of_layers')
            self.main_window.update_variable(self.main_window.input_prior_optimizer, 'optimizer')
            self.main_window.update_variable(self.main_window.input_prior_lr, 'learning_rate')
            self.main_window.update_variable(self.main_window.input_prior_epochs, 'epochs')
            self.main_window.update_variable(self.main_window.input_prior_k, 'k')
            self.main_window.update_variable(self.main_window.input_prior_p, 'p')
            self.main_window.update_variable(self.main_window.input_prior_loss, 'loss_function')
        print("FPQAJKNFLAKFAKLSNF",self.main_window.number_of_nodes, self.main_window.number_of_layers, self.main_window.learning_rate, self.main_window.optimizer, self.main_window.epochs, self.main_window.k, self.main_window.p, self.main_window.loss_function)
        
        print(os.path.abspath("../resources/icons/loading.gif"))
        layer_sizes = [int(self.main_window.number_of_nodes) for _ in range(int(self.main_window.number_of_layers))]
        learning_rate = self.main_window.learning_rate
        
        if isinstance(learning_rate, str) and learning_rate.strip() != '':  # Check if it's a non-empty string
            learning_rate = float(learning_rate)
        else:
            learning_rate = 0.001  # Default value if empty or invalid

        # Handle the value of 'p'
        p_value = self.main_window.p
        if isinstance(p_value, str) and p_value.strip() != '':
            p_value = float(p_value)
        else:
            p_value = 0.5  # Default value if empty or invalid

        #def __init__(self, model_type, dataset_type, hidden_sizes=None, lr=0.001, loss="CrossEntropy", optimizer="Adam", epochs=30, k=2, p=0.05):
        #self.trainer = Trainer(self.main_window.GLOBAL_CHOSEN_MODEL, self.main_window.GLOBAL_CHOSEN_DATASET, [6, 6, 6], lr=0.001)

        self.trainer = Trainer(self.main_window.GLOBAL_CHOSEN_MODEL, self.main_window.GLOBAL_CHOSEN_DATASET, layer_sizes, lr=learning_rate, loss=self.main_window.loss_function, optimizer=self.main_window.optimizer, epochs=self.main_window.epochs, k=self.main_window.k, p=p_value)
        self.main_window.stackedWidget_2.setCurrentWidget(self.main_window.model_training_page)
        self.main_window.model_train_button.setText("...")
        self.main_window.model_train_button.disconnect()
        self.main_window.model_train_button.clicked.connect(self.showChooseResultsPage)


        self.main_window.loading_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.main_window.loading_label.setFixedSize(30, 30)
        movie = QMovie("./resources/icons/loading.gif")
        print(os.path.abspath("./resources/icons/loading.gif"))
        self.main_window.loading_label.setMovie(movie)
        movie.start()




        self.trainer.message.connect(self.updateTrainingProcessLabel)
        self.trainer.finished.connect(self.onTrainingFinished)

        # start training 
        self.trainer.start()


    def updateTrainingProcessLabel(self, message):
        previous_text = self.main_window.training_process_label.text()

        self.main_window.training_process_label.setText(previous_text + "\n" + message)

    def onTrainingFinished(self):
        print("Training Finished")
        self.main_window.model_train_button.setEnabled(True)
        self.main_window.model_train_button.setText("Training Completed")
        
        self.showChooseResultsPage()

        self.trainer.wait()
        self.trainer = None
