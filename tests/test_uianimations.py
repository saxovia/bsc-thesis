import sys
import pytest
from PyQt6 import QtCore, QtWidgets
from src.uianimations import UIAnimations


@pytest.fixture(scope="module", autouse=True)
def app(request):
    app_instance = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    
    def cleanup():
        if QtWidgets.QApplication.instance():
            QtWidgets.QApplication.instance().quit()
    
    request.addfinalizer(cleanup)
    return app_instance

@pytest.fixture
def widget():
    test_widget = QtWidgets.QWidget()
    test_widget.resize(100, 100)
    test_widget.show()
    return test_widget

def test_fadeInUp_position(widget, qtbot):
    initial_pos = widget.pos()
    UIAnimations.fadeInUp(widget)
    def check_position():
        return widget.pos() == initial_pos
    qtbot.waitUntil(check_position, timeout=1000)
    
def test_fadeInUp_opacity(widget, qtbot):
    effect = UIAnimations.fadeInUp(widget)
    qtbot.wait(500)
    assert abs(effect.opacity() - 1.0) < 0.01