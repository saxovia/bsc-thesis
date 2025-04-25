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
    main_window.fadeInUp = MagicMock()
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

    handler.fadeToPage(new_page)

    called_lambda = mock_singleShot.call_args[0][1]
    with patch.object(handler, 'performFadeIn') as mock_performFadeIn:
        called_lambda()
        mock_performFadeIn.assert_called_once_with(new_page)

def test_perform_fade_in(mock_main_window):
    handler = PageNavigationHandler(mock_main_window)
    new_page = MagicMock()

    handler.performFadeIn(new_page)

    mock_main_window.stackedWidget.setCurrentWidget.assert_called_once_with(new_page)
    mock_main_window.fadeInUp.assert_called_once_with(new_page)

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
    handler.showHomePage()
    
    
    assert mock_main_window.stackedWidget.setCurrentWidget.called, "setCurrentWidget was not called"
    mock_main_window.stackedWidget.setCurrentWidget.assert_called_once_with(mock_main_window.home_page)
    mock_main_window.ui_handler.type_text_effect.assert_called_once_with(mock_main_window.home_text, mock_main_window.home_text.text(), mock_main_window.home_page)
    mock_main_window.restart_button.hide.assert_called_once()
    mock_main_window.undo_button.show.assert_called_once()



def test_reset_settings_button(mock_main_window):
    handler = PageNavigationHandler(mock_main_window)

    handler.resetSettingsButton()

    mock_main_window.settings_button.disconnect.assert_called_once()
    mock_main_window.settings_button.clicked.connect.assert_called_once()

def test_visualize_results_data_processing(handler):
    handler.main_window.previous_results = [{
        "model_type": "MLP",
        "graph_type": "BA",
        "training_metrics": {
            "final_train_loss": 0.5,
            "final_train_accuracy": 80.0,
            "final_val_loss": 0.6,
            "final_val_accuracy": 75.0,
            "model_metrics": {
                "global_sparsity": 0.2,
                "total_parameters": 10000,
                "prune_type": "IH"
            }
        },
        "graph_metrics": {
            "degree": {"node1": 0.5, "node2": 0.7},
            "eccentricity": {"node1": 0.1, "node2": 0.2},
            "closeness": {"node1": 0.6, "node2": 0.8},
            "betweenness": {"node1": 0.2, "node2": 0.3},
            "edge_betweenness": {"node1": 0.3, "node2": 0.4}
        }
    }]
    with patch('matplotlib.pyplot.figure'), \
         patch('src.pagenavigationhandler.PageNavigationHandler.display_graphs'):
        handler.visualize_results()
        assert 75.0 in handler.metrics["val_accuracy"]
        assert len(handler.metrics["val_accuracy"]) == 3

def test_empty_results(handler):
    handler.main_window.previous_results = []
    handler.visualize_results()
    assert not any(handler.metrics.values())


def test_save_graphs(mock_main_window, tmp_path):
    handler = PageNavigationHandler(mock_main_window)
    handler.metrics = {
        'model_type': ['MLP', 'MLP'],
        'graph_type': ['Full', 'Full'],
        'prune_type': ['IH', 'HH'],
        'total_parameters': [10000, 5000],
        'val_accuracy': [75.0, 80.0],
        'global_sparsity': [0.2, 0.5],
        'degree': [{'node1': 0.5}, {'node1': 0.3}],
        'eccentricity': [{'node1': 0.1}, {'node1': 0.2}],
        'closeness': [{'node1': 0.6}, {'node1': 0.4}],
        'betweenness': [{'node1': 0.2}, {'node1': 0.4}],
        'edge_betweenness': [{'node1': 0.3}, {'node1': 0.5}]
    }

    mock_main_window.load_default_graph_directory.return_value = str(tmp_path)
    mock_fig = MagicMock()
    mock_fig.savefig = MagicMock()

    with patch('matplotlib.pyplot.figure', return_value=mock_fig), \
         patch.object(handler, 'create_parameters_vs_accuracy_graph', return_value=mock_fig), \
         patch.object(handler, 'create_prune_metric_graph', return_value=mock_fig):
        handler.save_graphs()
        assert mock_fig.savefig.call_count == 6
        saved_paths = [call[0][0] for call in mock_fig.savefig.call_args_list]
        expected_filenames = [
            "parameters_vs_accuracy.png",
            "mean_eccentricity.png",
            "mean_degree.png",
            "mean_closeness.png",
            "mean_betweenness.png",
            "mean_edge_betweenness.png"
        ]

        for filename in expected_filenames:
            assert any(filename in path for path in saved_paths), f"{filename} not found in saved paths"
