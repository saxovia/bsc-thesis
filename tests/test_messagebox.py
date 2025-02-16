import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from PyQt6 import QtCore, QtWidgets
from src.messagebox import CustomMessageBox
import pytest
"""
@pytest.fixture(scope="module", autouse=True)
def app():
    if not QtWidgets.QApplication.instance():
        app = QtWidgets.QApplication(sys.argv)
        yield app
        app.quit()
    else:
        app = QtWidgets.QApplication.instance()
    return app


@pytest.fixture()
def messagebox(app):
    return CustomMessageBox('title', 'message', [('button_text', 1, 'callback')])

def test_init(messagebox : CustomMessageBox):
    assert messagebox.windowTitle() == 'title'
    assert messagebox.text() == 'message'
    assert messagebox.windowFlags() == QtCore.Qt.WindowType.FramelessWindowHint | QtCore.Qt.WindowType.Dialog"""


