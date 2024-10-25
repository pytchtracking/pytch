#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""GUI Utility Functions"""
import logging

from PyQt6 import QtCore as qc
from PyQt6 import QtGui as qg
from PyQt6 import QtWidgets as qw

import pyqtgraph as pg

logger = logging.getLogger("pytch.gui_utils")


class FloatQLineEdit(qw.QLineEdit):
    """A text field that accepts floating point numbers"""

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


class QHLine(qw.QFrame):
    """A horizontal separation line"""

    def __init__(self):
        super().__init__()
        self.setMinimumWidth(1)
        self.setFixedHeight(10)
        self.setFrameShape(qw.QFrame.Shape.HLine)
        self.setFrameShadow(qw.QFrame.Shadow.Sunken)
        self.setSizePolicy(
            qw.QSizePolicy.Policy.Preferred, qw.QSizePolicy.Policy.Minimum
        )


class QVLine(qw.QFrame):
    """A vertical separation line"""

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(1)
        self.setFixedWidth(20)
        self.setFrameShape(qw.QFrame.Shape.VLine)
        self.setFrameShadow(qw.QFrame.Shadow.Sunken)
        self.setSizePolicy(
            qw.QSizePolicy.Policy.Preferred, qw.QSizePolicy.Policy.Minimum
        )


def disable_interactivity(plot_item):
    plot_item.setMouseEnabled(x=False, y=False)  # Disable mouse panning & zooming
    plot_item.hideButtons()  # Disable corner auto-scale button
    plot_item.setMenuEnabled(False)  # Disable right-click context menu

    legend = plot_item.addLegend()  # This doesn't disable legend interaction
    # Override both methods responsible for mouse events
    legend.mouseDragEvent = lambda *args, **kwargs: None
    legend.hoverEvent = lambda *args, **kwargs: None
    # disable show / hide event of legend click
    legend.sampleType.mouseClickEvent = lambda *args, **kwargs: None
