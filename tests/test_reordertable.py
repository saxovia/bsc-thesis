import pytest
from PyQt6.QtCore import Qt, QModelIndex
from src.reordertable import ReorderTableModel

@pytest.fixture
def sample_data():
    return [
        ("1", "A", "Extra 1"),
        ("2", "B", "Extra 2"),
        ("3", "C", "Extra 3")
    ]

@pytest.fixture
def model(sample_data):
    headers = ["Numbers", "Letters", "Extras"]
    return ReorderTableModel(sample_data, headers, editable=True, show_edit_column=True)

def test_row_and_column_count(model):
    assert model.rowCount() == 4
    assert model.columnCount() == 6

def test_get_table_data(model):
    table_data = model.get_table_data()
    assert len(table_data) == 3
    assert table_data[0][0] == "A"
    assert table_data[0][1] == "Extra 1"

def test_get_set_hidden_data(model):
    model.set_hidden_data(0, {"hidden_key": "changed_value"})
    assert model.get_hidden_data(0) == {"hidden_key": "changed_value"}

def test_flags_for_editable_cell(model):
    index = model.index(0, 1)
    flags = model.flags(index)
    assert Qt.ItemFlag.ItemIsEditable in flags

def test_editable_field_changes_value(model):
    index = model.index(0, 1)
    result = model.setData(index, "Changed", Qt.ItemDataRole.EditRole)
    assert result is True
    assert model.data(index, Qt.ItemDataRole.DisplayRole) == "Changed"

def test_multiply_selected_items(model):
    model._data[0][0]=True
    assert model.multiply_selected_items(3) is True
    assert model.rowCount() == 6

def test_remove_selected_items(model):
    model._data[1][0]=True
    model._data[2][0]=True
    assert model.remove_selected_items() is True
    assert model.rowCount() == 2

def test_set_data_on_last_row_adds_new_row(model):
    last_row_idx = model.rowCount() - 1
    index = model.index(last_row_idx, 1)
    model.setData(index, "New", Qt.ItemDataRole.EditRole)
    assert model.rowCount()==5

def test_invalid_index_handling(model):
    invalid_index = QModelIndex()
    assert model.setData(invalid_index, "X", Qt.ItemDataRole.EditRole) is False
    assert model.data(invalid_index, Qt.ItemDataRole.DisplayRole) is None

def test_toggle_checkbox(model):
    index = model.index(0, 0)
    original = model._data[0][0]
    model.setData(index, not original, Qt.ItemDataRole.EditRole)
    assert model._data[0][0] == (not original)