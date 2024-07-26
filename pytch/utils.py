import logging
import numpy as np

from PyQt6 import QtCore as qc
from PyQt6 import QtGui as qg
from PyQt6 import QtWidgets as qw
from PyQt6.QtWidgets import QFrame, QSizePolicy

logger = logging.getLogger("pytch.util")

eps = np.finfo(float).eps


class FloatQLineEdit(qw.QLineEdit):
    accepted_value = qc.pyqtSignal(float)

    def __init__(self, default=None, *args, **kwargs):
        qw.QLineEdit.__init__(self, *args, **kwargs)
        self.setValidator(qg.QDoubleValidator())
        self.setFocusPolicy(qc.Qt.FocusPolicy.ClickFocus | qc.Qt.FocusPolicy.TabFocus)
        self.returnPressed.connect(self.do_check)
        p = self.parent()
        if p:
            self.returnPressed.connect(p.setFocus)
        if default:
            self.setText(str(default))

    def do_check(self):
        text = self.text()
        val = float(text)
        self.accepted_value.emit(val)


class QHLine(QFrame):
    """
    a horizontal separation line
    """

    def __init__(self):
        super().__init__()
        self.setMinimumWidth(1)
        self.setFixedHeight(20)
        self.setFrameShape(QFrame.Shape.HLine)
        self.setFrameShadow(QFrame.Shadow.Sunken)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)


class QVLine(QFrame):
    """
    a vertical separation line
    """

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(1)
        self.setFixedWidth(20)
        self.setFrameShape(QFrame.Shape.VLine)
        self.setFrameShadow(QFrame.Shadow.Sunken)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)


def f2cent(f, standard_frequency):
    return 1200.0 * np.log2(np.abs(f) / standard_frequency + eps)


def cent2f(p, standard_frequency):
    return np.exp2(p / 1200.0) * standard_frequency


def consecutive(arr):
    return np.split(arr, np.where(np.diff(arr) != 1)[0] + 1)


def index_gradient_filter(x, y, max_gradient):
    """Get index where the abs gradient of x, y is < max_gradient."""
    return np.where(np.abs(np.diff(y) / np.diff(x)) < max_gradient)[0]
