from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import QThread, pyqtSignal

class ReorderTableModel(QtCore.QAbstractTableModel):
    def __init__(self, data, headers=None, editable=True, show_edit_column=True, parent=None):
        super().__init__(parent)
        self._show_edit_column = show_edit_column
        self._data = [[False] + list(row) + ['', ''] for row in data]
        self._headers = [''] + (headers if headers else [f"Column {i+1}" for i in range(len(self._data[0]) - 3)]) + ['', '']
        self._editable = editable
        for row in self._data:
            row.append([])
        self._data.append([False] + [''] * (len(self._headers) - 3) + ['', '', {"hidden_key": "default_value"}])

    def columnCount(self, parent=None) -> int:
        return len(self._headers)

    def rowCount(self, parent=None) -> int:
        return len(self._data)
    
    def get_table_data(self):
        return [row[2:] for row in self._data]

    def headerData(self, column: int, orientation, role: QtCore.Qt.ItemDataRole):
        if role == QtCore.Qt.ItemDataRole.DisplayRole and orientation == QtCore.Qt.Orientation.Horizontal:
            return self._headers[column]
        return None
    
    def data(self, index: QtCore.QModelIndex, role: QtCore.Qt.ItemDataRole):
        if not index.isValid():
            return None
        
        row, col = index.row(), index.column()
        if row < 0 or row >= len(self._data) or col < 0 or col >= len(self._data[row]):
            return None

        if col==0:
            if role == QtCore.Qt.ItemDataRole.DecorationRole:
                return QIcon("./resources/icons/checked.png") if self._data[row][col] else QIcon("./resources/icons/unchecked.png")
            if role == QtCore.Qt.ItemDataRole.DisplayRole:
                return None  # Hide the True/False text
        if col == 2 and role == QtCore.Qt.ItemDataRole.DisplayRole:
            return str(row + 1)
        if col >= len(self._headers) - 2:
            if role == QtCore.Qt.ItemDataRole.DecorationRole:
                if col == len(self._headers) - 2:
                    return QIcon("./resources/icons/edit.png")
                elif col == len(self._headers) - 1:
                    return QIcon("./resources/icons/delete.png")
            return None

        if role in {QtCore.Qt.ItemDataRole.DisplayRole, QtCore.Qt.ItemDataRole.EditRole}:
            return self._data[row][col]

        return None
    
    def get_hidden_data(self, row):
        if not isinstance(row, int):
            return None
        if 0 <= row < len(self._data):
            return self._data[row][-1]
        return None

    def set_hidden_data(self, row, value):
        if not isinstance(row, int) or not isinstance(value, dict):
            return False
        if 0 <= row < len(self._data):
            self._data[row][-1] = value
            return True
        return False
    
    def multiply_selected_items(self, count):
        if count<1 or count>100:
            return False 
        selected_rows = [i for i, row in enumerate(self._data[:-1]) if row[0]]
        
        if not selected_rows:
            return False
        self.beginResetModel()
        
        for rowi in reversed(selected_rows):
            if not isinstance(count, int) or count <= 1:
                self.endResetModel()
                return False
            row_data = self._data[rowi].copy()
            for _ in range(count - 1):
                self._data.insert(rowi + 1, row_data.copy())
        
        for row in range(len(self._data)):
            self._data[row][0] = False
        
        self.endResetModel()
        return True
    
    def remove_selected_items(self):
        selected_rows = [i for i, row in enumerate(self._data[:-1]) if row[0]]
        if not selected_rows:
            return False
        selected_rows.sort(reverse=True)

        for row in selected_rows:
            self._data[row][0] = False

        view = self.parent()
        if isinstance(view, QtWidgets.QTableView):
            view.selectionModel().clearSelection()

        for rowi in selected_rows:
            self.beginRemoveRows(QtCore.QModelIndex(), rowi, rowi)
            self._data.pop(rowi)
            self.endRemoveRows()

        if isinstance(view, QtWidgets.QTableView):
            view.viewport().update()

        return True
    

    def setData(self, index: QtCore.QModelIndex, value, role: QtCore.Qt.ItemDataRole):
        if not index.isValid():
            return False

        row, col = index.row(), index.column()

        if col == 0 and role == QtCore.Qt.ItemDataRole.EditRole:
            self._data[row][col] = not self._data[row][col]
            self.dataChanged.emit(index, index, [QtCore.Qt.ItemDataRole.DecorationRole])
            return True

        if col >= len(self._headers) - 2:
            return False

        if role == QtCore.Qt.ItemDataRole.EditRole and col > 0 and self._editable:
            self._data[row][col] = value
            self.dataChanged.emit(index, index, [QtCore.Qt.ItemDataRole.EditRole])
            return True

        return False


    def flags(self, index: QtCore.QModelIndex) -> QtCore.Qt.ItemFlag:
        if not index.isValid():
            return QtCore.Qt.ItemFlag.ItemIsDropEnabled

        col = index.column()
        row = index.row()
        
        if row == self.rowCount() - 1:
            flags = QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable
            if col == 0:
                flags |= QtCore.Qt.ItemFlag.ItemIsEditable
            if self._editable and col > 0 and col < len(self._headers) - 2:
                flags |= QtCore.Qt.ItemFlag.ItemIsEditable
            return flags

        if col == 0:
            return QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable
        if col >= len(self._headers) - 2:
            return QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable | QtCore.Qt.ItemFlag.ItemIsDropEnabled

        flags = QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable | QtCore.Qt.ItemFlag.ItemIsDragEnabled | QtCore.Qt.ItemFlag.ItemIsDropEnabled
        if self._editable:
            flags |= QtCore.Qt.ItemFlag.ItemIsEditable
        
        
        return flags


    def relocateRow(self, row_source, row_target):
        if getattr(self, '_is_moving', False):
            return
        if not isinstance(row_source, int) or not isinstance(row_target, int):
            return
        if row_source==row_target or row_source<0 or row_target<0:
            return
        if self._is_moving or row_source == row_target or row_source < 0 or row_target < 0:
            return

        if row_target >= self.rowCount():
            row_target = self.rowCount() - 1 

        if row_source >= self.rowCount():
            return  # Don't move a non-existent row!!

        self._is_moving = True
        if row_target > row_source:
            # If moving row ddown
            row_target += 1

        self.beginMoveRows(QtCore.QModelIndex(), row_source, row_source, QtCore.QModelIndex(), row_target)

        self._data.insert(row_target if row_target > row_source else row_target, self._data.pop(row_source))

        self.endMoveRows()
        self._is_moving = False

    def supportedDropActions(self):
        return QtCore.Qt.DropAction.MoveAction

    def mimeTypes(self):
        return ["application/x-qabstractitemmodeldatalist"]

    def mimeData(self, indexes):
        data = QtCore.QMimeData()
        stream = QtCore.QDataStream(QtCore.QByteArray(), QtCore.QIODevice.OpenModeFlag.WriteOnly)
        rows = sorted(set(index.row() for index in indexes)) #unique rows only
        stream.writeInt32(len(rows))
        for row in rows:
            stream.writeInt32(row)

        data.setData("application/x-qabstractitemmodeldatalist", stream.device().data())
        return data

    def dropMimeData(self, data, action, row, column, parent):
        if not data.hasFormat("application/x-qabstractitemmodeldatalist"):
            return False
            
        stream = QtCore.QDataStream(data.data("application/x-qabstractitemmodeldatalist"), QtCore.QIODevice.OpenModeFlag.ReadOnly)
        if stream.atEnd():
            return False
        count = stream.readInt32()
        from_rows = [stream.readInt32() for _ in range(count)]

        from_rows = [r for r in from_rows if 0 <= r < self.rowCount() - 1]
        if not from_rows:
            return False        
        
        moved_data = []
        for r in sorted(from_rows, reverse=True):
            moved_data.insert(0, self._data.pop(r))
            self.beginRemoveRows(QtCore.QModelIndex(), r, r)
            self.endRemoveRows()

        if not isinstance(moved_data, list):
            return False
        # drag to first row
        if row == -1:
            row = 0

        adjusted_row = row - sum(1 for r in from_rows if r < row)
        adjusted_row = max(0, min(adjusted_row, self.rowCount()))
        
        if moved_data:
            self.beginInsertRows(QtCore.QModelIndex(), adjusted_row, adjusted_row + len(moved_data) - 1)
            for i, item in enumerate(moved_data):
                self._data.insert(adjusted_row + i, item)
            self.endInsertRows()

        return True
    
    def supportedDropActions(self):
        return QtCore.Qt.DropAction.MoveAction

class ReorderTableView(QtWidgets.QTableView):
    rowEdited = pyqtSignal(int)
    def __init__(self, parent):
        super().__init__(parent)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.MultiSelection)
        self.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.InternalMove)
        self.setDefaultDropAction(QtCore.Qt.DropAction.MoveAction)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.setDragDropOverwriteMode(False)
        self.setAcceptDrops(True)
        self.setDragEnabled(True)
        self.setDropIndicatorShown(True)
        self.clicked.connect(self.on_click) # Click signal

        header = self.horizontalHeader()
        header.setDefaultAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Interactive)
        header.setStretchLastSection(False)
            
    def mousePressEvent(self, event):
        if not self.model():
            return

        index = self.indexAt(event.position().toPoint())
        if index.isValid():
            col = index.column()
            model = self.model()
            if model and (col == model.columnCount() - 1 or col == model.columnCount() - 2):
                self.clicked.emit(index)
                return

            if event.button() == QtCore.Qt.MouseButton.RightButton:
                if model and col < model.columnCount() - 2:
                    if index.row() not in [selected.row() for selected in self.selectionModel().selectedRows()]:
                        return
                    self.edit(index)
                    return

        # Allow default behavior for selection
        super().mousePressEvent(event)

    def setModel(self, model):
        super().setModel(model)
        if model:
            self.setColumnWidth(0, 20)
            if model._show_edit_column:
                self.setColumnWidth(model.columnCount() - 2, 30)
            else:
                self.setColumnWidth(model.columnCount() - 2, 0)
            self.setColumnWidth(model.columnCount() - 1, 30)  #Delete col
            self.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Fixed)
            if model._show_edit_column:
                self.horizontalHeader().setSectionResizeMode(model.columnCount() - 2, QtWidgets.QHeaderView.ResizeMode.Fixed)
            self.horizontalHeader().setSectionResizeMode(model.columnCount() - 1, QtWidgets.QHeaderView.ResizeMode.Fixed)
            for col in range(1, model.columnCount() - 2):
                self.resizeColumnToContents(col)

    def on_click(self, index):
        if not index.isValid():
            return

        row, col = index.row(), index.column()
        model = self.model()
        if not model:
            return
        if row == model.rowCount()-1:
            return

        if col == model.columnCount()-1:
            selected_rows = [i for i, row_data in enumerate(model._data[:-1]) if row_data[0]]
            if selected_rows:
                model.remove_selected_items()
            else:
                model.beginRemoveRows(QtCore.QModelIndex(), row, row)
                model._data.pop(row)
                model.endRemoveRows()

        elif col == model.columnCount() - 2 and row < model.rowCount() - 1:
            self.rowEdited.emit(row)

    def dragEnterEvent(self, event):
        if event.source() is self:
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        index = self.indexAt(event.position().toPoint())
        if index.isValid() and index.row() == self.model().rowCount() - 1:
            event.ignore()
        else:
            # Show the drop indicator between rows
            self.setDropIndicatorShown(True)
            event.acceptProposedAction()

    def unlockDragging(self):
        self.is_moving = False

    def dropEvent(self, event):
        if event.source() is self:
            index = self.indexAt(event.position().toPoint())
            if not index.isValid():
                return

            item_rect = self.visualRect(index)
            mouse_pos = event.position().toPoint()
            relative_pos = mouse_pos.y() - item_rect.top()
            
            drop_row = index.row()
            if relative_pos > item_rect.height() / 2:
                drop_row += 1

            model = self.model()
            if model:
                data = event.mimeData()
                stream = QtCore.QDataStream(data.data("application/x-qabstractitemmodeldatalist"), 
                                        QtCore.QIODevice.OpenModeFlag.ReadOnly)
                count = stream.readInt32()
                from_rows = [stream.readInt32() for _ in range(count)]
                from_rows = [r for r in from_rows if 0 <= r < model.rowCount() - 1]
                
                if not from_rows:
                    return

                for r in sorted(from_rows):
                    if r < drop_row:
                        drop_row -= 1

                moved_data = []
                for r in sorted(from_rows, reverse=True):
                    model.beginRemoveRows(QtCore.QModelIndex(), r, r)
                    moved_data.insert(0, model._data.pop(r))
                    model.endRemoveRows()


                if moved_data:
                    drop_row = max(0, min(drop_row, model.rowCount()))
                    model.beginInsertRows(QtCore.QModelIndex(), drop_row, drop_row + len(moved_data) - 1)
                    for i, item in enumerate(moved_data):
                        model._data.insert(drop_row + i, item)
                    model.endInsertRows()

            event.accept()
        else:
            event.ignore()


    def selectionChanged(self, selected, deselected):
        super().selectionChanged(selected, deselected)
        model = self.model()
        if not model:
            return
        
        selected_indexes = self.selectionModel().selectedRows()

        if not selected_indexes:
            return
        selected_rows = {index.row() for index in selected_indexes}

        for row in range(model.rowCount()):
            index = model.index(row, 0)
            current_state = model.data(index, QtCore.Qt.ItemDataRole.EditRole)
            should_be_checked = row in selected_rows

            if current_state != should_be_checked:
                model.blockSignals(True)
                model.setData(index, should_be_checked, QtCore.Qt.ItemDataRole.EditRole)
                model.blockSignals(False)


        self.viewport().update()

    def shift_selection(self, start_index, end_index):
        if not start_index.isValid() or not end_index.isValid():
            return

        start_row, end_row = start_index.row(), end_index.row()
        if start_row > end_row:
            start_row, end_row = end_row, start_row

        model = self.model()
        if not model:
            return

        selection = QtCore.QItemSelection()
        first_col = model.index(start_row, 0)
        last_col = model.index(end_row, model.columnCount() - 1)
        selection.select(first_col, last_col)

        selection_model = self.selectionModel()
        selection_model.select(selection, QtCore.QItemSelectionModel.SelectionFlag.Select)

        for row in range(start_row, end_row + 1):
            index = model.index(row, 0)
            model.setData(index, True, QtCore.Qt.ItemDataRole.EditRole)

        self.viewport().update()

    def clear_selection(self):
        model = self.model()
        if model:
            for row in range(model.rowCount()):
                index = model.index(row, 0)
                model.setData(index, False, QtCore.Qt.ItemDataRole.EditRole)
            
            self.selectionModel().clearSelection()
            self.viewport().update()

    
    def select_all(self):
        model = self.model()
        if not model:
            return

        for row in range(model.rowCount()):
            index = model.index(row, 0)
            model.blockSignals(True)
            model.setData(index, True, QtCore.Qt.ItemDataRole.EditRole)
            model.blockSignals(False)

        selection = QtCore.QItemSelection()
        for row in range(model.rowCount()):
            first_col = model.index(row, 0)
            last_col = model.index(row, model.columnCount() - 1)
            selection.select(first_col, last_col)
        
        self.selectionModel().select(selection, QtCore.QItemSelectionModel.SelectionFlag.Select)
        self.viewport().update()


class Testing(QtWidgets.QMainWindow): #just in case for testing this by itself
    def __init__(self, editable=True, show_edit_column=True):
        super().__init__()
        data = [
            ("1","A","Extra 1"),
            ("2","B","Extra 2"),
            ("3","C","Extra 3"),
            ("4","D","Extra 4"),
        ]
        headers = ["Numbers", "ABCD", "Extra"]

        view = ReorderTableView(self)
        self.model = ReorderTableModel(data, headers, editable, show_edit_column)
        view.setModel(self.model)

        hidden_data_row_0 = self.model.get_hidden_data(0)
        print(f"Hidden data for row 0: {hidden_data_row_0}")
        self.model.set_hidden_data(0, {"hidden_key": "new_value"})
        if editable:
            view.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.DoubleClicked)
        else:
            view.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)

        toolbar = self.addToolBar("Tools")
        multiply_action = QtGui.QAction("Multiply Selected", self)
        multiply_action.triggered.connect(self.multiply_items)
        toolbar.addAction(multiply_action)
        delete_action = QtGui.QAction("Delete Selected", self)
        delete_action.triggered.connect(self.delete_items)
        toolbar.addAction(delete_action)


        view.rowEdited.connect(self.handle_row_edit)


        self.setCentralWidget(view)
        self.show()
    def handle_row_edit(self, row):
        print(f"Row {row} was edited")

    def multiply_items(self):
        count, ok = QtWidgets.QInputDialog.getInt(
            self, "Multiply Items", "How many copies?", 2, 1, 100, 1
        )
        if ok:
            self.model.multiply_selected_items(count)
    def delete_items(self):
        reply = QtWidgets.QMessageBox.question(
            self, 'Delete Items',
            'Are you sure?',
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No
        )
        
        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            self.model.remove_selected_items()

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = Testing(editable=True, show_edit_column=True) 
    sys.exit(app.exec())