# tests/conftest.py
import pytest
from PyQt6 import QtWidgets
from src.mainwindow import MainWindow

@pytest.fixture
def app(qtbot):
    test_app = QtWidgets.QApplication.instance()
    if test_app is None:
        test_app = QtWidgets.QApplication([])
    main_win = MainWindow()
    qtbot.addWidget(main_win)
    return main_win

def test_mainwindow_initialization(app):
    assert app.windowTitle() == 'MainWindow'
    assert app.model_train_button.isEnabled()
    assert app.current_page == "Home"


def test_toggle_maximize_restore(app):
    app.showNormal()
    app.toggle_maximize_restore()
    assert app.isMaximized()
    assert app.is_maximized

    app.toggle_maximize_restore()
    assert not app.isMaximized()
    assert not app.is_maximized
def test_type_text_effect(app, qtbot):
    label = QtWidgets.QLabel()
    app.type_text_effect(label, "Test", interval=10)

    with qtbot.waitSignal(app.typing_timer.timeout, timeout=100, raising=False):
        pass

    assert label.text().startswith("T")


def test_multiply_rows_timeline(app, monkeypatch): #monkeypatch is used to mock the input dialog
    model = app.timelineTableModel
    app.reorder_table_view2.selectRow(0)

    monkeypatch.setattr(QtWidgets.QInputDialog, "getInt", lambda *a, **kw: (2, True))
    initial_count = model.rowCount()

    app.multiply_rows_timeline(model)

    assert model.rowCount() == initial_count + 2  # added 2 copies

def test_complete_reset(app):
    app.model_train_button.setEnabled(False)
    app.saved_label.setText("Saved!")
    app.complete_reset()

    assert app.model_train_button.isEnabled()
    assert app.saved_label.text() == ""

def test_overwrite_table_data(app):
    new_data = [["X", "Y", "Z"] + [""] * (app.pruningTableModel.columnCount() - 3)]
    app.overwrite_table_data(app.pruningTableModel, new_data)

    assert app.pruningTableModel._data[0][0] == "X"

def test_show_warning_calls_discard(monkeypatch, app):
    called = {"discarded": False}

    def fake_msgbox(*args, **kwargs):
        called["discarded"]=True
        return type("FakeBox", (), {"exec": lambda self: None})()

    monkeypatch.setattr("src.mainwindow.CustomMessageBox", fake_msgbox)
    app.show_warning(buttons=["discard", "cancel"])

    assert called["discarded"]
