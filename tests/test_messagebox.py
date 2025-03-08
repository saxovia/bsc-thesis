import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#this doesnt work as of now
"""
__________________________ ERROR collecting tests/test_messagebox.py __________________________ 
ImportError while importing test module 'C:\Users\Weebmachine\Desktop\bsc-thesis\tests\test_messagebox.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
F:\Programs\Anaconda\Lib\importlib\__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
tests\test_messagebox.py:7: in <module>
    from PyQt6 import QtCore, QtWidgets
E   ImportError: DLL load failed while importing QtCore: The specified procedure could not be found.
=================================== short test summary info =================================== 
ERROR tests/test_messagebox.py
!!!!!!!!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!
"""
from PyQt6 import QtCore, QtWidgets
from src.messagebox import CustomMessageBox
import pytest

def test_dummy():
    pass
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
    buttons = [
        ('Discard', QtWidgets.QMessageBox.ButtonRole.RejectRole, lambda: print('Discard clicked')),
        ('Save Changes', QtWidgets.QMessageBox.ButtonRole.AcceptRole, lambda: print('Save Changes clicked'))
    ]
    return CustomMessageBox('Test Title', 'Test Message', buttons)

def test_init(messagebox: CustomMessageBox):
    assert messagebox.windowTitle() == 'Test Title'
    assert messagebox.text() == 'Test Message'
    assert messagebox.windowFlags() == (QtCore.Qt.WindowType.FramelessWindowHint | QtCore.Qt.WindowType.Dialog)

def test_buttons(messagebox: CustomMessageBox):
    button_texts = [btn.text() for btn in messagebox.buttons()]
    assert 'Discard' in button_texts
    assert 'Save Changes' in button_texts

def test_button_callbacks(messagebox: CustomMessageBox, capsys):
    discard_button = messagebox.findChild(QtWidgets.QPushButton, 'discard')
    save_changes_button = messagebox.findChild(QtWidgets.QPushButton, 'save_changes')

    discard_button.click()
    captured = capsys.readouterr()
    assert 'Discard clicked' in captured.out

    save_changes_button.click()
    captured = capsys.readouterr()
    assert 'Save Changes clicked' in captured.out

def test_close_button(messagebox: CustomMessageBox):
    close_button = messagebox.findChild(QtWidgets.QPushButton, 'closeButton')
    assert close_button is not None
    assert close_button.icon().name() == "../resources/icons/close.png"
    assert close_button.styleSheet() == "background:transparent;"



@pytest.fixture()
def messagebox(app):
    buttons = [('OK', QtWidgets.QMessageBox.ButtonRole.AcceptRole, lambda: print('OK clicked')),
               ('Cancel', QtWidgets.QMessageBox.ButtonRole.RejectRole, lambda: print('Cancel clicked'))]
    return CustomMessageBox('Test Title', 'Test Message', buttons)

def test_messagebox_title(messagebox):
    assert messagebox.windowTitle() == 'Test Title'

def test_messagebox_message(messagebox):
    assert messagebox.text() == 'Test Message'

def test_messagebox_buttons(messagebox):
    buttons = messagebox.findChildren(QtWidgets.QPushButton)
    assert len(buttons) == 2
    assert buttons[0].text() == 'OK'
    assert buttons[1].text() == 'Cancel'

def test_messagebox_button_callbacks(messagebox):
    buttons = messagebox.findChildren(QtWidgets.QPushButton)
    for button in buttons:
        button.click()
        assert button in messagebox.button_map

def test_messagebox_close_button(messagebox):
    close_button = messagebox.closeButton
    assert close_button is not None
    assert close_button.icon().name() == "../resources/icons/close.png"
    close_button.click()
    assert not messagebox.isVisible()

def test_messagebox_styling(messagebox):
    stylesheet = messagebox.styleSheet()
    assert "QMessageBox QPushButton#discard:hover" in stylesheet
    assert "background-color: #c42b1c !important;" in stylesheet


