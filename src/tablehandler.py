from PyQt6 import QtWidgets, QtCore 
from src.reordertable import ReorderTableView, ReorderTableModel


class TableHandler:
    def __init__(self, main_window):
        self.main_window = main_window
        self.data = [
            ["", "1", "MLP", "Prune", "MNIST", "[89, 44, 22, 11, 4, 80]", "CrossEntropy", "Adam", "", "", "", 32, "", "Full"],
            ["", "2", "MLP", "Prune", "MNIST", "[15,9,6,4,2,12]", "CrossEntropy", "SGD", "", "", "", 32, "", "Full"],
            ["", "3", "MLP", "Prune", "MNIST", "[11,3,6,4,2,8]", "CrossEntropy", "RMSprop", "", "", "", 32, "", "Full"],
            ["", "4", "MLP", "Prior", "MNIST", "48", "CrossEntropy", "Adam", "2", 2, 1.0, 64,0.001, "WS"],
            ["", "5", "MLP", "Prior", "MNIST", "70", "CrossEntropy", "Adam", "2", 2, 0.8, 64,0.01, "WS"],
            ["", "6", "MLP", "Prior", "MNIST", "100", "CrossEntropy", "Adam", "2", 2, 0.7, 64,0.01, "WS"],
            ["", "7", "MLP", "Prior", "MNIST", "250", "CrossEntropy", "Adam", "2", 2, 0.5, 64,0.01, "WS"],
        ]
        self.hiddendata = [
            [ ["","", "1", "Retrain", "-", "-", "-", "1", "0.001"], ["", "", "2", "Prune", "FULL", "10", "Magnitude", "-", "-"], ["", "", "3", "Retrain", "-", "-", "-", "2", "0.001"]],
        ]

    def show_table_widget(self):
        # sample data
        processed_data = [[False] + row for row in self.hiddendata]

        self.main_window.pruningTableModel = ReorderTableModel(processed_data, headers=["", "Step", "Action", "Scope", "Pruning %", "Method", "Epochs", "Learning Rate"], show_edit_column=False)

        self.main_window.reorder_table_view = ReorderTableView(self.main_window)
        self.main_window.reorder_table_view.setModel(self.main_window.pruningTableModel)
        self.main_window.reorder_table_view.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.DoubleClicked)

        self.main_window.reorder_table_view.selectionModel().selectionChanged.connect(self.handle_pruning_selection_changed)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.main_window.reorder_table_view)

        header = self.main_window.reorder_table_view.horizontalHeader()
        header.setDefaultAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Interactive)
        header.setStyleSheet("QHeaderView::section {"
                            "   qproperty-alignment: AlignCenter;"
                            "   padding: 4px;"
                            "   font-size: 7pt;"
                            "   white-space: normal;"
                            "}")
        # Adjust font size
        font = header.font()
        font.setPointSize(8)  # Set to a smaller font size
        header.setFont(font)
        self.main_window.reorder_table_view.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.main_window.reorder_table_view.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.MultiSelection)

        self.main_window.reorder_table_view.verticalHeader().hide()
        self.main_window.reorder_table_view.resizeColumnsToContents()
        self.main_window.reorder_table_view.setColumnWidth(0, 20)
        if self.main_window.tableWidgetPruning.layout():
            QtWidgets.QWidget().setLayout(self.main_window.tableWidgetPruning.layout()) 

        header = self.main_window.reorder_table_view.horizontalHeader()
        for col in range(2, self.main_window.pruningTableModel.columnCount() - 2):
            header.setSectionResizeMode(col, QtWidgets.QHeaderView.ResizeMode.Stretch)

        self.main_window.reorder_table_view.verticalHeader().hide()
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.main_window.reorder_table_view)

        if self.main_window.tableWidgetPruning.layout():
            QtWidgets.QWidget().setLayout(self.main_window.tableWidgetPruning.layout()) 
        self.main_window.tableWidgetPruning.setLayout(layout)
        self.main_window.tableWidgetPruning.resizeColumnsToContents()

    def show_table_widget2(self):
        # sample data
        self.main_window.timelineTableModel = ReorderTableModel(self.data, headers=["", "", "Model\nType", "Start", "Dataset", "Hidden\nsizes", "Loss", "Optimizer", "Epochs", "k", "p", "Batch\nSize", "Learning\nRate", "Graph\nType"])

        self.main_window.reorder_table_view2 = ReorderTableView(self.main_window)
        self.main_window.reorder_table_view2.setModel(self.main_window.timelineTableModel)
        self.main_window.reorder_table_view2.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.DoubleClicked)

        for i in range(len(self.data)):
            if len(self.hiddendata) > 0:
                self.main_window.timelineTableModel.set_hidden_data(i, self.hiddendata[0])

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.main_window.reorder_table_view2)

        header = self.main_window.reorder_table_view2.horizontalHeader()
        header.setDefaultAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Interactive)
        header.setStyleSheet("QHeaderView::section {"
                            "   qproperty-alignment: AlignCenter;"
                            "   padding: 4px;"
                            "   font-size: 7pt;"
                            "   white-space: normal;"
                            "}")

        font = header.font()
        font.setPointSize(8)
        header.setFont(font)
        self.main_window.reorder_table_view2.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.main_window.reorder_table_view2.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.MultiSelection)

        if self.main_window.tableWidgetPruning_2.layout():
            QtWidgets.QWidget().setLayout(self.main_window.tableWidgetPruning_2.layout()) 

        self.main_window.reorder_table_view2.resizeColumnsToContents()

        self.main_window.reorder_table_view2.setColumnWidth(0,20)
        self.main_window.reorder_table_view2.setColumnWidth(1,1)
        self.main_window.reorder_table_view2.setColumnWidth(6,80)
        self.main_window.reorder_table_view2.setColumnWidth(7,80)
        self.main_window.reorder_table_view2.setColumnWidth(8,60)
        self.main_window.reorder_table_view2.setColumnWidth(11,30)
        self.main_window.reorder_table_view2.setColumnWidth(12,30)

        header = self.main_window.reorder_table_view2.horizontalHeader()
        for col in range(2, self.main_window.timelineTableModel.columnCount() - 2):
            header.setSectionResizeMode(col, QtWidgets.QHeaderView.ResizeMode.Stretch)

        self.main_window.reorder_table_view2.verticalHeader().hide()

        self.main_window.tableWidgetPruning_2.setLayout(layout)
        self.main_window.tableWidgetPruning_2.resizeColumnsToContents()
        self.main_window.multiply_rows_timeline_button.clicked.connect(lambda: self.multiply_rows_timeline(self.main_window.timelineTableModel))
        self.main_window.multiply_rows_pruning_button.clicked.connect(lambda: self.multiply_rows_timeline(self.main_window.pruningTableModel))

        self.main_window.reorder_table_view2.rowEdited.connect(lambda row: self.handle_row_edit(row))

    def multiply_rows_timeline(self, model):
        popup = QtWidgets.QDialog(self.main_window)
        popup.setWindowFlags(QtCore.Qt.WindowType.FramelessWindowHint)
        popup.setMinimumSize(300, 150)
        popup.setStyleSheet("""
            QDialog {
                background-color: #121212;
                border: 1px solid #333333;
                border-radius: 10px;
            }
            QLabel {
                color: #ffffff;
                font-size: 14px;
            }
            QSpinBox {
                background-color: #1e1e1e;
                color: #ffffff;           
                border: 1px solid #333333;
                border-radius: 5px;
                padding: 5px;
            }
            QPushButton {
                background-color: #1e1e1e;
                color: #ffffff;           
                border: 1px solid #333333;
                border-radius: 5px;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #333333;
            }
        """)

        layout = QtWidgets.QVBoxLayout(popup)
        label = QtWidgets.QLabel("How many times would you like to multiply the selected rows?")
        layout.addWidget(label)

        spin_box = QtWidgets.QSpinBox()
        spin_box.setRange(1, 100)
        spin_box.setValue(1)
        layout.addWidget(spin_box)

        button_layout = QtWidgets.QHBoxLayout()
        ok_button = QtWidgets.QPushButton("OK")
        cancel_button = QtWidgets.QPushButton("Cancel")
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

        ok_button.clicked.connect(popup.accept)
        cancel_button.clicked.connect(popup.reject)

        window_rect = self.main_window.geometry()
        dialog_rect = popup.geometry()
        x = window_rect.center().x() - dialog_rect.center().x()
        y = window_rect.center().y() - dialog_rect.center().y()
        popup.move(x, y)

        # Show the popup and handle the result
        if popup.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            count = spin_box.value()
            model.multiply_selected_items(count + 1)
                
    def handle_row_edit(self, row):
        # Select the row if it's not already selected
        if not self.main_window.timelineTableModel._data[row][0]:
            self.main_window.timelineTableModel.setData(
                self.main_window.timelineTableModel.index(row, 0),
                True,
                QtCore.Qt.ItemDataRole.EditRole
            )
            self.main_window.reorder_table_view2.selectionModel().select(
                self.main_window.timelineTableModel.index(row, 0),
                QtCore.QItemSelectionModel.SelectionFlag.Select | QtCore.QItemSelectionModel.SelectionFlag.Rows
            )
        self.main_window.reorder_table_view2.setCurrentIndex(
            self.main_window.reorder_table_view2.model().index(row , 0)
        )        
        print(self.main_window.reorder_table_view2.selectionModel())
        data = self.main_window.timelineTableModel.get_hidden_data(row)

        if not data or not isinstance(data, list):
            data = self.hiddendata[0]
        elif len(data) > 0 and not isinstance(data[0], list):
            data = [data]
            
        self.main_window.pruningTableModel.beginResetModel()
        self.main_window.pruningTableModel._data = []

        for hidden_row in data:
            if isinstance(hidden_row, list):
                new_row = [""] * self.main_window.pruningTableModel.columnCount()
                for j in range(min(len(hidden_row), self.main_window.pruningTableModel.columnCount())):
                    new_row[j] = hidden_row[j]
                self.main_window.pruningTableModel._data.append(new_row)

        # Check if the last row is empty and add one if it isn't
        if not self.main_window.pruningTableModel._data or not all(cell == "" for cell in self.main_window.pruningTableModel._data[-1]):
            self.main_window.pruningTableModel._data.append([""] * self.main_window.pruningTableModel.columnCount())

        self.main_window.pruningTableModel.endResetModel()
        self.main_window.page_navigation_handler.show_model_page()
        self.main_window.page_navigation_handler.show_model_pruning_table_page()

    def overwrite_table_data(self, table, data):
        table.beginResetModel()
        table._data = []

        for row in data:
            new_row = [""] * table.columnCount()
            for j in range(min(len(row), table.columnCount())):
                new_row[j] = row[j]
            table._data.append(row)
        table.endResetModel()

    def reset_timeline_table(self):
        table = self.main_window.timelineTableModel
        data = self.data
        table.beginResetModel()
        table._data = []
        
        for row in data:
            new_row = [False] + list(row) + ['', '', []]
            table._data.append(new_row)
            
            for i in range(len(self.hiddendata)):
                table.set_hidden_data(len(table._data)-1, self.hiddendata[i])
            
        table._data.append([False] + [''] * (len(table._headers) - 3) + ['', '', {"hidden_key": "default_value"}])
        
        table.endResetModel()


    def handle_pruning_selection_changed(self, selected, deselected):
        selected_rows = [index.row() for index in self.main_window.reorder_table_view.selectionModel().selectedRows()]
        deselected_rows = [index.row() for index in deselected.indexes()]
        
        #print(f"Selected rows: {selected_rows}")
        #print(f"Deselected rows: {deselected_rows}")

