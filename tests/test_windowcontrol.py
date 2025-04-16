import pytest
from PyQt6 import QtCore, QtWidgets, QtGui
from src.windowcontrol import WindowControl

@pytest.fixture
def test_window(qtbot):
    window = QtWidgets.QMainWindow()
    window.title_frame = QtWidgets.QFrame(window)
    window.is_maximized = False
    window.show()
    qtbot.addWidget(window)
    return window

@pytest.fixture
def window_control(test_window):
    return WindowControl(test_window)

def test_initial_state(window_control):
    assert window_control.mousePressed is False
    assert window_control.old_pos is None
    assert window_control.title_bar.hasMouseTracking() is True


def test_mouse_tracking_enabled(window_control):
    assert window_control.title_bar.hasMouseTracking() is True
    assert window_control.title_bar.testAttribute(QtCore.Qt.WidgetAttribute.WA_MouseTracking) is True


def test_ignore_non_titlebar_events(window_control, qtbot):
    other_widget = QtWidgets.QWidget(window_control.window)
    qtbot.mousePress(other_widget, QtCore.Qt.MouseButton.LeftButton)
    
    assert window_control.mousePressed is False
    assert window_control.old_pos is None