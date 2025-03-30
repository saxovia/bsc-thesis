from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtCore import pyqtSlot

class MoveRowTask(QtCore.QRunnable):
    """Handles row move operation in a separate thread."""
    def __init__(self, model, row_source, row_target, table_view):
        super().__init__()
        self.model = model # Model to be updated
        self.row_source = row_source # Source row index
        self.row_target = row_target # Target row index
        self.table_view = table_view # Table view to unlock dragging

    def run(self):
        if self.row_source == self.row_target or self.row_source < 0 or self.row_target < 0:
            return
        print(f"Moving row {self.row_source} to {self.row_target}")
        QtCore.QMetaObject.invokeMethod(self.model, "relocateRow", QtCore.Qt.ConnectionType.QueuedConnection,
                                        QtCore.Q_ARG(int, self.row_source), QtCore.Q_ARG(int, self.row_target)) # Move the row in the model
        QtCore.QMetaObject.invokeMethod(self.table_view, "unlockDragging", QtCore.Qt.ConnectionType.QueuedConnection) # Unlock dragging in the table view

class ReorderTableModel(QtCore.QAbstractTableModel):

    def __init__(self, data, headers=None, editable=True, parent=None):
        super().__init__(parent)
        self._data = [[False] + list(row) for row in data]  # Add checkmark column
        self._headers = ["Select"] + (headers if headers else [f"Column {i+1}" for i in range(len(self._data[0]) - 1)])
        self._is_moving = False
        self._editable = editable
        self.thread_pool = QtCore.QThreadPool.globalInstance()

    def columnCount(self, parent=None) -> int:
        return len(self._headers)

    def rowCount(self, parent=None) -> int:
        return len(self._data)
    
    def get_table_data(self):
        return [row[1:] for row in self._data]  # Exclude the first column (checkbox)


    def headerData(self, column: int, orientation, role: QtCore.Qt.ItemDataRole):
        if role == QtCore.Qt.ItemDataRole.DisplayRole and orientation == QtCore.Qt.Orientation.Horizontal:
            return self._headers[column]
        return None

    def data(self, index: QtCore.QModelIndex, role: QtCore.Qt.ItemDataRole):
        if not index.isValid():
            return None
        
        row, col = index.row(), index.column()
        
        # Handle checkbox column (first column)
        if col == 0:
            if role == QtCore.Qt.ItemDataRole.CheckStateRole: # Check state role
                 return QtCore.Qt.CheckState.Checked if self._data[row][col] else QtCore.Qt.CheckState.Unchecked # Check if the checkbox is checked or not
        
        # Handle text data
        if role in {QtCore.Qt.ItemDataRole.DisplayRole, QtCore.Qt.ItemDataRole.EditRole}: # Display and edit roles
            return self._data[row][col]  # Return the data for the specified row and column

        return None

    def setData(self, index: QtCore.QModelIndex, value, role: QtCore.Qt.ItemDataRole) -> bool:
        if not index.isValid():
            return False
        
        row, col = index.row(), index.column()

        # Handle checkbox interaction
        # Handle checkbox interaction
        if col == 0 and role == QtCore.Qt.ItemDataRole.CheckStateRole:

            if value == int(QtCore.Qt.CheckState.Checked.value):
                print(f"Checkbox at row {row} checked")
                self._data[row][col] = True
            else:
                print(f"Checkbox at row {row} unchecked")
                self._data[row][col] = False

            self.dataChanged.emit(index, index, [QtCore.Qt.ItemDataRole.CheckStateRole])
            return True
        
        # Handle text editing (if editable)
        if role == QtCore.Qt.ItemDataRole.EditRole and col > 0 and self._editable:
            self._data[row][col] = value
            self.dataChanged.emit(index, index, [QtCore.Qt.ItemDataRole.EditRole])
            return True
        
        return False


    def flags(self, index: QtCore.QModelIndex) -> QtCore.Qt.ItemFlag:
        if not index.isValid():
            return QtCore.Qt.ItemFlag.ItemIsDropEnabled

        col = index.column()

        # Checkbox column: enable checking/unchecking
        if col == 0:
            return QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsUserCheckable | QtCore.Qt.ItemFlag.ItemIsSelectable

        # Other columns: allow editing if enabled
        flags = QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable | QtCore.Qt.ItemFlag.ItemIsDragEnabled | QtCore.Qt.ItemFlag.ItemIsDropEnabled
        print (f"Flags for column {col}: {flags}")
        if self._editable:
            flags |= QtCore.Qt.ItemFlag.ItemIsEditable  # Allow editing only if _editable is True
        
        return flags


    @pyqtSlot(int, int)
    def relocateRow(self, row_source, row_target) -> None:
        print(f"Moving row {row_source} to {row_target}")
        if self._is_moving or row_source == row_target or row_source < 0 or row_target < 0:
            return

        self._is_moving = True
        self.beginMoveRows(QtCore.QModelIndex(), row_source, row_source, QtCore.QModelIndex(), row_target)
        self._data.insert(row_target, self._data.pop(row_source))
        self.endMoveRows()
        self._is_moving = False

    def queueMove(self, row_source, row_target, table_view):
        print(f"Queueing move from {row_source} to {row_target}")
        if not self._is_moving:
            self.thread_pool.start(MoveRowTask(self, row_source, row_target, table_view))


    def supportedDropActions(self):
        return QtCore.Qt.DropAction.MoveAction

    def mimeTypes(self):
        return ["application/x-qabstractitemmodeldatalist"]

    def mimeData(self, indexes):
        """Serialize data when dragging."""
        data = QtCore.QMimeData()
        stream = QtCore.QDataStream(QtCore.QByteArray(), QtCore.QIODevice.OpenModeFlag.WriteOnly)
        stream.writeInt(indexes[0].row())  # Store the row index
        data.setData("application/x-qabstractitemmodeldatalist", stream.device().data())
        return data

    def dropMimeData(self, data, action, row, column, parent):
        """Deserialize data when dropping."""
        if not data.hasFormat("application/x-qabstractitemmodeldatalist"):
            return False
        stream = QtCore.QDataStream(data.data("application/x-qabstractitemmodeldatalist"), QtCore.QIODevice.OpenModeFlag.ReadOnly)
        from_index = stream.readInt()
        if from_index < 0 or from_index >= self.rowCount():
            return False
        self.queueMove(from_index, row, parent)
        return True


class ReorderTableView(QtWidgets.QTableView):
    def __init__(self, parent):
        super().__init__(parent)
        #self.verticalHeader().hide()
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.InternalMove)
        self.setDragDropOverwriteMode(False)
        self.setAcceptDrops(True)
        self.setDragEnabled(True)
        self.setDropIndicatorShown(True)
        self.is_moving = False

    def dragEnterEvent(self, event):
        if event.source() is self:
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        event.acceptProposedAction()

    @pyqtSlot()  # Expose unlockDragging to the QMetaObject system
    def unlockDragging(self):
        print(f"Unlocking dragging")
        self.is_moving = False

    def dropEvent(self, event):
        print(f"Drop event triggered")
        if event.source() is not self or self.is_moving or self.model()._is_moving:
            event.ignore()
            return

        from_index = self.selectionModel().currentIndex().row()
        to_index = self.indexAt(event.position().toPoint()).row()

        if 0 <= from_index < self.model().rowCount() and 0 <= to_index < self.model().rowCount() and from_index != to_index: # Check if the indices are valid and not the same
            self.is_moving = True
            print(f"Moving from {from_index} to {to_index}")
            self.model().queueMove(from_index, to_index, self)
            event.acceptProposedAction()
        else:
            event.ignore()



class Testing(QtWidgets.QMainWindow):
    def __init__(self, editable=True):
        super().__init__()
        data = [
            ("Regex 1", "Category A", "Extra 1", "extra"),
            ("Regex 2", "Category B", "Extra 2", "extra"),
            ("Regex 3", "Category C", "Extra 3", "extra"),
            ("Regex 4", "Category D", "Extra 4", "extra"),
        ]
        headers = ["Regex", "Category", "Additional Info", "Extra"]

        view = ReorderTableView(self)
        model = ReorderTableModel(data, headers, editable)
        view.setModel(model)

        if editable:
            view.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.DoubleClicked)
        else:
            view.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)

        self.setCentralWidget(view)
        self.show()

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = Testing(editable=True)  # Change to False to disable editing
    sys.exit(app.exec())
