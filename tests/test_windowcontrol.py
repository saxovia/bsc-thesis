from unittest.mock import MagicMock
from PyQt6 import QtWidgets, QtCore, QtGui
import pytest
from src.windowcontrol import WindowControl


@pytest.fixture
def mock_window():
    window = MagicMock(spec=QtWidgets.QMainWindow)
    window.title_frame = MagicMock()
    window.geometry.return_value = QtCore.QRect(100, 100, 800, 600)
    
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

def test_mouseReleaseEvent(window_control, mock_window):
    window_control.mousePressed = True
    event = MagicMock(spec=QtGui.QMouseEvent)
    event.button.return_value = QtCore.Qt.MouseButton.LeftButton
    window_control.old_pos = QtCore.QPoint(100, 100)

    window_control.mouseReleaseEvent(event)
    assert window_control.mousePressed is False


def test_maximize_window_on_release_near_top_edge(window_control, mock_window, mock_screen):
    screen_geometry = mock_screen.geometry.return_value
    window_geometry = mock_window.geometry.return_value
    
    window_geometry.top = MagicMock(return_value=screen_geometry.top() + 5)

    event = MagicMock(spec=QtGui.QMouseEvent)
    event.button.return_value = QtCore.Qt.MouseButton.LeftButton

    window_control.mouseReleaseEvent(event)
    mock_window.showMaximized.assert_called_once()
    assert window_control.window.is_maximized is True


def test_no_maximize_when_released_far_from_top(window_control, mock_window, mock_screen):
    screen_geometry = mock_screen.geometry.return_value
    window_geometry = mock_window.geometry.return_value
    
    window_geometry.top = MagicMock(return_value=screen_geometry.top() + 50)
    event = MagicMock(spec=QtGui.QMouseEvent)
    event.button.return_value = QtCore.Qt.MouseButton.LeftButton
    
    window_control.mouseReleaseEvent(event)
    mock_window.showMaximized.assert_not_called()
