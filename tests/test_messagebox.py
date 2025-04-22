import pytest
from PyQt6 import QtCore, QtWidgets, QtGui
from unittest.mock import MagicMock
from src.windowcontrol import WindowControl


@pytest.fixture
def mock_window():
    window = MagicMock(spec=QtWidgets.QMainWindow)
    window.title_frame = MagicMock()
    window.x.return_value= 100
    window.y.return_value = 5
    window.geometry.return_value = QtCore.QRect(100, 5, 800, 600)
    
    return window


@pytest.fixture
def mock_screen():
    screen = MagicMock()
    screen.geometry.return_value = QtCore.QRect(0, 0, 1920, 1080)
    
    return screen


@pytest.fixture
def window_control(mock_window, mock_screen):
    QtWidgets.QApplication.primaryScreen = MagicMock(return_value=mock_screen)
    return WindowControl(mock_window)


def test_mouseReleaseEvent(window_control, mock_window, mock_screen):
    window_control.mousePressed = True
    event = MagicMock(spec=QtGui.QMouseEvent)
    event.button.return_value = QtCore.Qt.MouseButton.LeftButton
    window_control.old_pos = QtCore.QPoint(100, 100)
    window_control.mouseReleaseEvent(event)
    
    mock_window.showMaximized.assert_called_once()
