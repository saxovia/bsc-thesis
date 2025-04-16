import sys
import pytest
from PyQt6 import QtCore, QtWidgets
from src.uianimations import UIAnimations


@pytest.fixture(scope="module", autouse=True)
def app():
    if not QtWidgets.QApplication.instance():
        app = QtWidgets.QApplication(sys.argv)
        yield app
        app.quit()
    else:
        app = QtWidgets.QApplication.instance()
    return app

@pytest.fixture
def widget():
    test_widget = QtWidgets.QWidget()
    test_widget.resize(100, 100)
    test_widget.show()
    return test_widget

def test_fadeInUp_position(widget, qtbot):
    initial_pos = widget.pos()
    widget.move(initial_pos + QtCore.QPoint(0, 30))
    UIAnimations.fadeInUp(widget)

    def check_position():
        assert widget.pos() == initial_pos

    qtbot.waitUntil(check_position, timeout=500)

def test_fadeInUp_opacity(widget, qtbot):
    effect = QtWidgets.QGraphicsOpacityEffect(widget)
    widget.setGraphicsEffect(effect)
    UIAnimations.fadeInUp(widget)

    def check_opacity():
        assert effect.opacity() == 1.0

    qtbot.waitUntil(check_opacity, timeout=500)