from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import QThread, pyqtSignal

class ReorderTableModel(QtCore.QAbstractTableModel):
    def __init__(self, data, headers=None, editable=True, show_edit_column=True, parent=None):
        super().__init__(parent)
        self._editable = editable
        self._show_edit_column = show_edit_column
        self._data = [[False] + list(row) + ['', ''] for row in data]
        self._headers = [''] + (headers if headers else [f"Column {i+1}" for i in range(len(data[0]))]) + ['', '']
        for row in self._data:
            row.append({})
        self._data.append([False] + [''] * (len(self._headers) - 3) + ['', '', {}])

    def rowCount(self, parent=None):
        return len(self._data)

    def columnCount(self, parent=None):
        return len(self._headers)

    def get_table_data(self):
        return [row[2:] for row in self._data[:-1]]

    def headerData(self, section, orientation, role):
        if role == QtCore.Qt.ItemDataRole.DisplayRole and orientation == QtCore.Qt.Orientation.Horizontal:
            return self._headers[section]
        return None

    def data(self, index, role):
        if not index.isValid():
            return None
        row, col = index.row(), index.column()

        if role == QtCore.Qt.ItemDataRole.BackgroundRole:
            if self._data[row][0]:
                view = self.parent()
                if isinstance(view, QtWidgets.QTableView):
                    return view.palette().brush(QtGui.QPalette.ColorGroup.Active, QtGui.QPalette.ColorRole.Highlight)
                return QtGui.QBrush(QtGui.QColor(173, 216, 230))

        if col == 0:
            if role == QtCore.Qt.ItemDataRole.DecorationRole:
                return QIcon("./resources/icons/checked.png") if self._data[row][0] else QIcon("./resources/icons/unchecked.png")
            if role == QtCore.Qt.ItemDataRole.DisplayRole:
                return None

        if col == 2 and role == QtCore.Qt.ItemDataRole.DisplayRole:
            return str(row + 1)

        if col >= len(self._headers) - 2:
            if role == QtCore.Qt.ItemDataRole.DecorationRole:
                return QIcon("./resources/icons/edit.png") if col == len(self._headers) - 2 else QIcon("./resources/icons/delete.png")

        if role in (QtCore.Qt.ItemDataRole.DisplayRole, QtCore.Qt.ItemDataRole.EditRole):
            return self._data[row][col]

        return None

    def setData(self, index, value, role):
        if not index.isValid():
            return False

        row, col = index.row(), index.column()

        if col == 0 and role == QtCore.Qt.ItemDataRole.EditRole:
            self._data[row][0] = bool(value)
            self.dataChanged.emit(index, index, [QtCore.Qt.ItemDataRole.DecorationRole])
            return True

        if col >= len(self._headers) - 2:
            return False

        if role == QtCore.Qt.ItemDataRole.EditRole and col > 0 and self._editable:
            self._data[row][col] = value
            self.dataChanged.emit(index, index, [QtCore.Qt.ItemDataRole.EditRole])

            selected_rows = [i for i, row_data in enumerate(self._data[:-1]) if row_data[0] and i != row]
            for sr in selected_rows:
                self._data[sr][col] = value
                self.dataChanged.emit(self.index(sr, col), self.index(sr, col), [QtCore.Qt.ItemDataRole.EditRole])

            if row == len(self._data) - 1:
                self.beginInsertRows(QtCore.QModelIndex(), len(self._data), len(self._data))
                self._data.append([False] + [''] * (len(self._headers) - 3) + ['', '', {}])
                self.endInsertRows()

            return True

        return False

    def flags(self, index):
        if not index.isValid():
            return QtCore.Qt.ItemFlag.ItemIsDropEnabled

        flags = QtCore.Qt.ItemFlag.ItemIsSelectable | QtCore.Qt.ItemFlag.ItemIsEnabled

        if index.row() < self.rowCount() - 1:
            flags |= QtCore.Qt.ItemFlag.ItemIsDragEnabled | QtCore.Qt.ItemFlag.ItemIsDropEnabled
            if index.column() == 0:
                flags |= QtCore.Qt.ItemFlag.ItemIsEditable
            if 0 < index.column() < len(self._headers) - 2 and self._editable:
                flags |= QtCore.Qt.ItemFlag.ItemIsEditable

        elif index.row() == self.rowCount() - 1:
            if 0 <= index.column() < len(self._headers) - 2 and self._editable:
                flags |= QtCore.Qt.ItemFlag.ItemIsEditable

        return flags

    def mimeTypes(self):
        return ["application/x-qabstractitemmodeldatalist"]

    def mimeData(self, indexes):
        mime_data = QtCore.QMimeData()
        encoded_data = QtCore.QByteArray()
        stream = QtCore.QDataStream(encoded_data, QtCore.QIODevice.OpenModeFlag.WriteOnly)

        rows = sorted(set(index.row() for index in indexes))
        stream.writeInt32(len(rows))
        for r in rows:
            stream.writeInt32(r)

        mime_data.setData("application/x-qabstractitemmodeldatalist", encoded_data)
        return mime_data

    def dropMimeData(self, mime_data, action, row, column, parent):
        if not mime_data.hasFormat("application/x-qabstractitemmodeldatalist"):
            return False

        stream = QtCore.QDataStream(mime_data.data("application/x-qabstractitemmodeldatalist"), QtCore.QIODevice.OpenModeFlag.ReadOnly)
        count = stream.readInt32()
        from_rows = [stream.readInt32() for _ in range(count)]
        from_rows = [r for r in from_rows if r < self.rowCount() - 1]

        if not from_rows:
            return False

        moved = [self._data[r] for r in from_rows]
        for r in sorted(from_rows, reverse=True):
            self.beginRemoveRows(QtCore.QModelIndex(), r, r)
            self._data.pop(r)
            self.endRemoveRows()

        if row == -1:
            row = self.rowCount() - 1

        row = min(row, self.rowCount() - 1)

        self.beginInsertRows(QtCore.QModelIndex(), row, row + len(moved) - 1)
        for i, item in enumerate(moved):
            self._data.insert(row + i, item)
        self.endInsertRows()

        return True

    def supportedDropActions(self):
        return QtCore.Qt.DropAction.MoveAction

    def multiply_selected_items(self, count):
        selected_rows = [i for i, row in enumerate(self._data[:-1]) if row[0]]
        if not selected_rows:
            return False

        self.beginResetModel()
        for rowi in reversed(selected_rows):
            for _ in range(count - 1):
                self._data.insert(rowi + 1, self._data[rowi].copy())
        for row in self._data:
            row[0] = False
        self.endResetModel()
        return True

    def remove_selected_items(self):
        selected_rows = [i for i, row in enumerate(self._data[:-1]) if row[0]]
        if not selected_rows:
            return False

        for row in sorted(selected_rows, reverse=True):
            self.beginRemoveRows(QtCore.QModelIndex(), row, row)
            self._data.pop(row)
            self.endRemoveRows()

        return True

    def get_hidden_data(self, row):
        return self._data[row][-1] if 0 <= row < len(self._data) else None

    def set_hidden_data(self, row, value):
        if 0 <= row < len(self._data):
            self._data[row][-1] = value
            return True
        return False

class ReorderTableView(QtWidgets.QTableView):
    rowEdited = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.open_context_menu)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.MultiSelection)
        self.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.InternalMove)
        self.setDragDropOverwriteMode(False)
        self.setAcceptDrops(True)
        self.setDragEnabled(True)
        self.setDropIndicatorShown(True)

        self.clicked.connect(self.handle_click)

    def setModel(self, model):
        super().setModel(model)
        if model:
            self.setColumnWidth(0, 20)
            self.setColumnWidth(model.columnCount() - 2, 30)
            self.setColumnWidth(model.columnCount() - 1, 30)
            for col in range(1, model.columnCount() - 2):
                self.resizeColumnToContents(col)

    def open_context_menu(self, position):
        index = self.indexAt(position)
        if index.isValid():
            self.edit(index)
    def handle_click(self, index):
        if not index.isValid():
            return
        model = self.model()
        row, col = index.row(), index.column()

        if col == model.columnCount() - 1 and row < model.rowCount() - 1:
            model.remove_selected_items() if any(r[0] for r in model._data[:-1]) else self.delete_single_row(row)

        elif col == model.columnCount() - 2 and row < model.rowCount() - 1:
            self.rowEdited.emit(row)

    def delete_single_row(self, row):
        model = self.model()
        model.beginRemoveRows(QtCore.QModelIndex(), row, row)
        model._data.pop(row)
        model.endRemoveRows()

    def selectionChanged(self, selected, deselected):
        super().selectionChanged(selected, deselected)
        model = self.model()
        selected_rows = {idx.row() for idx in self.selectionModel().selectedIndexes()}

        for row in range(model.rowCount()):
            idx = model.index(row, 0)
            model.setData(idx, row in selected_rows, QtCore.Qt.ItemDataRole.EditRole)

class Testing(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        data = [("1", "A", "Extra 1"), ("2", "B", "Extra 2"), ("3", "C", "Extra 3")]
        headers = ["Numbers", "Letters", "Extra"]

        self.view = ReorderTableView(self)
        self.model = ReorderTableModel(data, headers)
        self.view.setModel(self.model)
        self.view.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.DoubleClicked)

        toolbar = self.addToolBar("Tools")
        mult_action = QtGui.QAction("Multiply Selected", self)
        mult_action.triggered.connect(self.multiply)
        toolbar.addAction(mult_action)

        del_action = QtGui.QAction("Delete Selected", self)
        del_action.triggered.connect(self.delete)
        toolbar.addAction(del_action)

        self.setCentralWidget(self.view)
        self.show()

    def multiply(self):
        count, ok = QtWidgets.QInputDialog.getInt(self, "Multiply", "Copies?", 2, 1, 100, 1)
        if ok:
            self.model.multiply_selected_items(count)

    def delete(self):
        if QtWidgets.QMessageBox.question(self, "Confirm", "Delete selected?", QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No) == QtWidgets.QMessageBox.StandardButton.Yes:
            self.model.remove_selected_items()

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = Testing()
    sys.exit(app.exec())
