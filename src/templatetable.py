from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtCore import pyqtSlot

#TODO add selection groups. when multiple things are selected move them all at once. to the desired place.
#TODO add multiplication of selected items. to desired number of times.
#TODO add a funcitonality to remove selected items. The main window can have a button to remove selected items.
class ReorderTableModel(QtCore.QAbstractTableModel):

    def __init__(self, data, headers=None, editable=True, parent=None):
        super().__init__(parent)
        self._data = [[False] + list(row) for row in data] 
        self._headers = ["Select"] + (headers if headers else [f"Column {i+1}" for i in range(len(self._data[0]) - 1)])
        self._is_moving = False
        self._editable = editable
        self._data.append([False] + [''] * (len(self._headers) - 1))

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
        if col == 0:
            if role == QtCore.Qt.ItemDataRole.CheckStateRole:
                 return QtCore.Qt.CheckState.Checked if self._data[row][col] else QtCore.Qt.CheckState.Unchecked 

        if role in {QtCore.Qt.ItemDataRole.DisplayRole, QtCore.Qt.ItemDataRole.EditRole}:
            return self._data[row][col]

        return None

    def setData(self, index: QtCore.QModelIndex, value, role: QtCore.Qt.ItemDataRole) -> bool:
        if not index.isValid():
            return False
        
        row, col = index.row(), index.column()

        if col == 0 and role == QtCore.Qt.ItemDataRole.CheckStateRole:
            self._data[row][col] = (value == int(QtCore.Qt.CheckState.Checked.value))
            self.dataChanged.emit(index, index, [QtCore.Qt.ItemDataRole.CheckStateRole])
            return True
        
        if role == QtCore.Qt.ItemDataRole.EditRole and col > 0 and self._editable:
            self._data[row][col] = value
            self.dataChanged.emit(index, index, [QtCore.Qt.ItemDataRole.EditRole])
            # Should the last row be edited, add one
            if row == len(self._data) - 1:
                self.beginInsertRows(QtCore.QModelIndex(), len(self._data), len(self._data))
                self._data.append([False] + [''] * (len(self._headers) - 1))
                self.endInsertRows()
            return True
        
        return False


    def flags(self, index: QtCore.QModelIndex) -> QtCore.Qt.ItemFlag:
        if not index.isValid():
            return QtCore.Qt.ItemFlag.ItemIsDropEnabled

        col = index.column()

        if col == 0:
            return QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsUserCheckable | QtCore.Qt.ItemFlag.ItemIsSelectable

        flags = QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable | QtCore.Qt.ItemFlag.ItemIsDragEnabled | QtCore.Qt.ItemFlag.ItemIsDropEnabled
        if self._editable:
            flags |= QtCore.Qt.ItemFlag.ItemIsEditable 
        
        return flags


    def relocateRow(self, row_source, row_target):
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
        stream.writeInt(indexes[0].row()) 
        data.setData("application/x-qabstractitemmodeldatalist", stream.device().data())
        return data

    def dropMimeData(self, data, action, row, column, parent):
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
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.InternalMove)
        self.setDragDropOverwriteMode(False)
        self.setAcceptDrops(True)
        self.setDragEnabled(True)
        self.setDropIndicatorShown(True)

    def dragEnterEvent(self, event):
        if event.source() is self:
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        event.acceptProposedAction()

    @pyqtSlot() 
    def unlockDragging(self):
        self.is_moving = False

    def dropEvent(self, event):
        if not self.model(): # Need this here to prevent access violations!
            event.ignore()
            return

        from_index = self.selectionModel().currentIndex().row()
        to_index = self.indexAt(event.position().toPoint()).row()

        if 0 <= from_index < self.model().rowCount() and 0 <= to_index < self.model().rowCount() and from_index != to_index:
            self.model().relocateRow(from_index, to_index)
            event.acceptProposedAction()
        else:
            event.ignore()


class Testing(QtWidgets.QMainWindow): #just in case for testing this by itself
    def __init__(self, editable=True):
        super().__init__()
        data = [
            ("1","A","Extra 1"),
            ("2","B","Extra 2"),
            ("3","C","Extra 3"),
            ("4","D","Extra 4"),
        ]
        headers = ["Numbers", "ABCD", "Extra"]

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
    window = Testing(editable=True) 
    sys.exit(app.exec())
