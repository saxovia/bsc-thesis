from PyQt6 import QtWidgets, QtCore

class MoveRowTask(QtCore.QRunnable):
    """Handles row move operation in a separate thread."""
    def __init__(self, model, row_source, row_target, table_view):
        super().__init__()
        self.model = model
        self.row_source = row_source
        self.row_target = row_target
        self.table_view = table_view

    def run(self):
        if self.row_source == self.row_target or self.row_source < 0 or self.row_target < 0:
            return

        QtCore.QMetaObject.invokeMethod(self.model, "relocateRow", QtCore.Qt.ConnectionType.QueuedConnection,
                                        QtCore.Q_ARG(int, self.row_source), QtCore.Q_ARG(int, self.row_target))
        QtCore.QMetaObject.invokeMethod(self.table_view, "unlockDragging", QtCore.Qt.ConnectionType.QueuedConnection)

class ReorderTableModel(QtCore.QAbstractTableModel):
    def __init__(self, data, headers=None, parent=None, *args):
        super().__init__(parent, *args)
        self._data = [list(row) for row in data]
        self._headers = headers if headers else [f"Column {i+1}" for i in range(len(self._data[0]))]
        self._is_moving = False
        self.thread_pool = QtCore.QThreadPool.globalInstance()

    def columnCount(self, parent=None) -> int:
        return len(self._headers)

    def rowCount(self, parent=None) -> int:
        return len(self._data)  # No extra row to avoid drop issues

    def headerData(self, column: int, orientation, role: QtCore.Qt.ItemDataRole):
        if role == QtCore.Qt.ItemDataRole.DisplayRole and orientation == QtCore.Qt.Orientation.Horizontal:
            return self._headers[column]
        return None

    def data(self, index: QtCore.QModelIndex, role: QtCore.Qt.ItemDataRole):
        if not index.isValid():
            return None

        if role in {QtCore.Qt.ItemDataRole.DisplayRole, QtCore.Qt.ItemDataRole.EditRole}:
            return self._data[index.row()][index.column()]
        return None

    def flags(self, index: QtCore.QModelIndex) -> QtCore.Qt.ItemFlag:
        if not index.isValid():
            return QtCore.Qt.ItemFlag.ItemIsDropEnabled
        return (QtCore.Qt.ItemFlag.ItemIsEnabled |
                QtCore.Qt.ItemFlag.ItemIsEditable |
                QtCore.Qt.ItemFlag.ItemIsSelectable |
                QtCore.Qt.ItemFlag.ItemIsDragEnabled |
                QtCore.Qt.ItemFlag.ItemIsDropEnabled)

    def relocateRow(self, row_source, row_target) -> None:
        if self._is_moving or row_source == row_target or row_source < 0 or row_target < 0:
            return

        self._is_moving = True
        self.beginMoveRows(QtCore.QModelIndex(), row_source, row_source, QtCore.QModelIndex(), row_target)
        self._data.insert(row_target, self._data.pop(row_source))
        self.endMoveRows()
        self._is_moving = False

    def queueMove(self, row_source, row_target, table_view):
        if not self._is_moving:
            self.thread_pool.start(MoveRowTask(self, row_source, row_target, table_view))

    def setData(self, index: QtCore.QModelIndex, value, role: QtCore.Qt.ItemDataRole) -> bool:
        if role == QtCore.Qt.ItemDataRole.EditRole and index.isValid():
            self._data[index.row()][index.column()] = value
            self.dataChanged.emit(index, index, [QtCore.Qt.ItemDataRole.EditRole])
            return True
        return False

class ReorderTableView(QtWidgets.QTableView):
    def __init__(self, parent):
        super().__init__(parent)
        self.verticalHeader().hide()
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.InternalMove)
        self.setDragDropOverwriteMode(False)
        self.setAcceptDrops(True)
        self.viewport().setAcceptDrops(True)
        self.setDragEnabled(True)
        self.setDropIndicatorShown(True)
        self.is_moving = False

    def dragEnterEvent(self, event):
        if event.source() is self:
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.source() is self:
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.source() is not self or self.is_moving or self.model()._is_moving:
            event.ignore()
            return

        from_index = self.selectionModel().currentIndex().row()
        to_index = self.indexAt(event.position().toPoint()).row()

        if 0 <= from_index < self.model().rowCount() and 0 <= to_index < self.model().rowCount() and from_index != to_index:
            self.is_moving = True
            self.model().queueMove(from_index, to_index, self)
            event.acceptProposedAction()
        else:
            event.ignore()

    def unlockDragging(self):
        self.is_moving = False

class Testing(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        data = [
            ("Regex 1", "Category A", "Extra 1", "extra"),
            ("Regex 2", "Category B", "Extra 2", "extra"),
            ("Regex 3", "Category C", "Extra 3", "extra"),
            ("Regex 4", "Category D", "Extra 4", "extra"),
        ]
        headers = ["Regex", "Category", "Additional Info", "Extra"]

        view = ReorderTableView(self)
        view.setModel(ReorderTableModel(data, headers))
        view.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.DoubleClicked)

        self.setCentralWidget(view)
        self.show()

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = Testing()
    sys.exit(app.exec())
