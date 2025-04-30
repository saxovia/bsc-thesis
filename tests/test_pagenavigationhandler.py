import pytest
from unittest.mock import MagicMock, patch
from src.pagenavigationhandler import PageNavigationHandler
import PyQt6.QtWidgets as QtWidgets
import pytest
import os

@pytest.fixture
def handler(mock_main_window):
    from src.pagenavigationhandler import PageNavigationHandler
    return PageNavigationHandler(mock_main_window)

@pytest.fixture
def mock_main_window():
    main_window = MagicMock()
    main_window.stackedWidget.setCurrentWidget = MagicMock()
    main_window.fade_in_up = MagicMock()
    main_window.settings_button.disconnect = MagicMock()
    main_window.settings_button.clicked.connect = MagicMock()
    main_window.restart_button.show = MagicMock()
    main_window.undo_button.show = MagicMock()
    main_window.undo_button.hide = MagicMock()
    main_window.undo_button.setEnabled = MagicMock()
    mock_layout = MagicMock(spec=QtWidgets.QVBoxLayout)
    main_window.scrollAreaWidgetContents_2.layout.return_value = mock_layout

    mock_scroll_area_widget = MagicMock(spec=QtWidgets.QWidget)
    main_window.scrollAreaWidgetContents_2 = mock_scroll_area_widget
    mock_scroll_area_widget.layout.return_value = mock_layout
    
    return main_window
    
    return main_window
@pytest.fixture
def mock_backend():
    mock_backend_qt6 = MagicMock()
    with patch('matplotlib.backends.backend_qt5', mock_backend_qt6):
        yield mock_backend_qt6
def mock_qcore_application():
    with patch("PyQt6.QtCore.QCoreApplication.exec"):
        yield

@patch("PyQt6.QtCore.QTimer.singleShot")
def test_fade_to_page(mock_singleShot, mock_main_window):
    handler = PageNavigationHandler(mock_main_window)
    new_page = MagicMock()

    handler.fade_to_page(new_page)

    called_lambda = mock_singleShot.call_args[0][1]
    with patch.object(handler, 'perform_fadeIn') as mock_perform_fadeIn:
        called_lambda()
        mock_perform_fadeIn.assert_called_once_with(new_page)

def test_perform_fade_in(mock_main_window):
    handler = PageNavigationHandler(mock_main_window)
    new_page = MagicMock()

    handler.perform_fadeIn(new_page)

    mock_main_window.stackedWidget.setCurrentWidget.assert_called_once_with(new_page)
    mock_main_window.fade_in_up.assert_called_once_with(new_page)

@patch("PyQt6.QtCore.QTimer.singleShot")
def test_show_home_page(mock_singleShot):
    mock_main_window = MagicMock()
    mock_main_window.stackedWidget = MagicMock()
    mock_main_window.home_page = MagicMock()
    mock_main_window.ui_handler = MagicMock()
    mock_main_window.restart_button = MagicMock()
    mock_main_window.undo_button = MagicMock()
    mock_main_window.home_text = MagicMock()
    

    mock_singleShot.side_effect = lambda delay, func: func()
    handler = PageNavigationHandler(mock_main_window)
    handler.show_home_page()
    
    
    assert mock_main_window.stackedWidget.setCurrentWidget.called, "setCurrentWidget was not called"
    mock_main_window.stackedWidget.setCurrentWidget.assert_called_once_with(mock_main_window.home_page)
    mock_main_window.ui_handler.type_text_effect.assert_called_once_with(mock_main_window.home_text, mock_main_window.home_text.text(), mock_main_window.home_page)
    mock_main_window.restart_button.hide.assert_called_once()
    mock_main_window.undo_button.show.assert_called_once()



def test_reset_settings_button(mock_main_window):
    handler = PageNavigationHandler(mock_main_window)

    handler.reset_settings_button()

    mock_main_window.settings_button.disconnect.assert_called_once()
    mock_main_window.settings_button.clicked.connect.assert_called_once()

def test_empty_results(handler):
    handler.main_window.previous_results = []
    handler.visualize_results()
    assert not any(handler.metrics.values())

