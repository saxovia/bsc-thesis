# tests/conftest.py
import pytest
from PyQt6 import QtWidgets, QtCore
from src.mainwindow import MainWindow
import os
import sys

@pytest.fixture
def app(qtbot):
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    
    main_win = MainWindow()
    qtbot.addWidget(main_win)
    yield main_win
    
    if hasattr(main_win, 'model_training_handler') and hasattr(main_win.model_training_handler, 'trainer') and main_win.model_training_handler.trainer is not None:
        main_win.model_training_handler.trainer.stop()
    
    #cleanup
    for child in main_win.findChildren(QtCore.QObject):
        if hasattr(child, 'disconnect'):
            try:
                child.disconnect()
            except TypeError:
                pass
        if hasattr(child, 'deleteLater'):
            child.deleteLater()
    
    main_win.close()
    main_win.deleteLater()
    
    QtCore.QCoreApplication.processEvents()
    
    QtCore.QTimer.singleShot(100, app.quit)

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

    timer = app.typing_timer
    qtbot.addWidget(label)

    with qtbot.waitSignal(timer.timeout, timeout=100, raising=False):
        pass

    assert label.text().startswith("T")



def test_complete_reset(app, qtbot):
    app.model_train_button.setEnabled(False)
    app.saved_label.setText("Saved!")
    
    # Ensure any existing timers are cleaned up
    QtCore.QCoreApplication.processEvents()
    
    app.complete_reset()
    
    # Wait for any pending animations to complete
    def check_button():
        return app.model_train_button.isEnabled()
    
    qtbot.waitUntil(check_button, timeout=2000)
    assert app.model_train_button.isEnabled()
    assert app.saved_label.text() == ""


def test_show_warning_calls_discard(monkeypatch, app):
    called = {"discarded": False}

    def fake_msgbox(*args, **kwargs):
        called["discarded"] = True
        return type("FakeBox", (), {"exec": lambda self: None})()

    monkeypatch.setattr("src.mainwindow.CustomMessageBox", fake_msgbox)
    app.show_warning(buttons=["discard", "cancel"])

    assert called["discarded"]