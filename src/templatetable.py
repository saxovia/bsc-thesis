from PyQt6 import QtWidgets, QtCore

# https://stackoverflow.com/questions/61387248/in-pyqt5-how-do-i-properly-move-rows-in-a-qtableview-using-dragdrop
class ReorderTableModel(QtCore.QAbstractTableModel):
    def __init__(self, data, headers=None, parent=None, *args):
        super().__init__(parent, *args)
        self._data = [list(row) for row in data]
        self._headers = headers if headers else [f"Column {i+1}" for i in range(len(self._data[0]))]

    def columnCount(self, parent=None) -> int:
        return len(self._headers)

    def rowCount(self, parent=None) -> int:
        return len(self._data) + 1  # Last row for new entries

    def headerData(self, column: int, orientation, role: QtCore.Qt.ItemDataRole):
        if role == QtCore.Qt.ItemDataRole.DisplayRole and orientation == QtCore.Qt.Orientation.Horizontal:
            return self._headers[column]
        return None

    def data(self, index: QtCore.QModelIndex, role: QtCore.Qt.ItemDataRole):
        if not index.isValid() or role not in {QtCore.Qt.ItemDataRole.DisplayRole, QtCore.Qt.ItemDataRole.EditRole}:
            return None

        if index.row() < len(self._data):
            return self._data[index.row()][index.column()]
        else:
            return "edit me" if role == QtCore.Qt.ItemDataRole.DisplayRole else ""

    def flags(self, index: QtCore.QModelIndex) -> QtCore.Qt.ItemFlag:
        if not index.isValid():
            return QtCore.Qt.ItemFlag.ItemIsDropEnabled
        if index.row() < len(self._data):
            return (QtCore.Qt.ItemFlag.ItemIsEnabled |
                    QtCore.Qt.ItemFlag.ItemIsEditable |
                    QtCore.Qt.ItemFlag.ItemIsSelectable |
                    QtCore.Qt.ItemFlag.ItemIsDragEnabled)
        return QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsEditable

    def supportedDropActions(self) -> QtCore.Qt.DropAction:
        return QtCore.Qt.DropAction.MoveAction | QtCore.Qt.DropAction.CopyAction
    
    def relocateRow(self, row_source, row_target) -> None:
        if row_source == row_target or row_source < 0 or row_target < 0:
            return  # Prevent invalid movement

        self.beginMoveRows(QtCore.QModelIndex(), row_source, row_source, QtCore.QModelIndex(), row_target)
        self._data.insert(row_target, self._data.pop(row_source))
        self.endMoveRows()


    def setData(self, index: QtCore.QModelIndex, value, role: QtCore.Qt.ItemDataRole) -> bool:
        if role == QtCore.Qt.ItemDataRole.EditRole and index.isValid():
            if index.row() < len(self._data):  # Editing an existing row
                old_value = self._data[index.row()][index.column()]
                self._data[index.row()][index.column()] = value
                print(f"Row {index.row()} Column {index.column()} changed from '{old_value}' to '{value}'")
            else:  # Editing the last row (adds a new row)
                print(f"New row added at index {index.row()} with value '{value}'")
                self.beginInsertRows(QtCore.QModelIndex(), len(self._data), len(self._data))
                self._data.append([""] * self.columnCount())  # Append new row with correct columns
                self.endInsertRows()
                self._data[index.row()][index.column()] = value  # Save edited value in last row

            self.dataChanged.emit(index, index, [QtCore.Qt.ItemDataRole.EditRole])
            return True
        return False


class ReorderTableView(QtWidgets.QTableView):
    """QTableView with the ability to make the model move a row with drag & drop"""

    class DropmarkerStyle(QtWidgets.QProxyStyle):
        def drawPrimitive(self, element, option, painter, widget=None):
            if element == QtWidgets.QStyle.PrimitiveElement.PE_IndicatorItemViewItemDrop and not option.rect.isNull():
                option_new = QtWidgets.QStyleOption(option)
                option_new.rect.setLeft(0)
                if widget:
                    option_new.rect.setRight(widget.width())
                option = option_new
            super().drawPrimitive(element, option, painter, widget)

    def __init__(self, parent):
        super().__init__(parent)
        self.verticalHeader().hide()
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.InternalMove)
        self.setDragDropOverwriteMode(False)
        self.setStyle(self.DropmarkerStyle())

    def dropEvent(self, event):
        if (event.source() is not self or
            (event.dropAction() != QtCore.Qt.DropAction.MoveAction and
             self.dragDropMode() != QtWidgets.QAbstractItemView.DragDropMode.InternalMove)):
            super().dropEvent(event)

        selection = self.selectedIndexes()
        from_index = selection[0].row() if selection else -1
        to_index = self.indexAt(event.position().toPoint()).row()
        if (0 <= from_index < self.model().rowCount() and
            0 <= to_index < self.model().rowCount() and
            from_index != to_index):
            self.model().relocateRow(from_index, to_index)
            event.accept()
        super().dropEvent(event)


class Testing(QtWidgets.QMainWindow):
    """Demonstrate ReorderTableView"""
    def __init__(self):
        super().__init__()
        
        # Example dataset with dynamic columns
        data = [
            ("Regex 1", "Category A", "Extra 1", "extra"),
            ("Regex 2", "Category B", "Extra 2", "extra"),
            ("Regex 3", "Category C", "Extra 3", "extra"),
            ("Regex 4", "Category D", "Extra 4", "extra"),
        ]
        headers = ["Regex", "Category", "Additional Info","extra"]

        view = ReorderTableView(self)
        view.setModel(ReorderTableModel(data, headers))
        view.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.DoubleClicked)

        self.setCentralWidget(view)
        self.show()

