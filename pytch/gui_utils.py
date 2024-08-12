#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""GUI Utility Functions"""
import logging

from PyQt6 import QtCore as qc
from PyQt6 import QtGui as qg
from PyQt6 import QtWidgets as qw

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


class BlitManager:
    """Manages blitting in matplotlib GUIs. Inspired by:
    https://matplotlib.org/stable/users/explain/animations/blitting.html
    """

    def __init__(self, canvas, animated_artists=()):
        self.canvas = canvas
        self._bg = None
        self._artists = []

        for a in animated_artists:
            self.add_artist(a)
        # grab the background on every draw
        self.cid = canvas.mpl_connect("draw_event", self.on_draw)

    def on_draw(self, event):
        """Callback to register with event."""
        cv = self.canvas
        if event is not None:
            if event.canvas != cv:
                raise RuntimeError
        self._bg = cv.copy_from_bbox(cv.figure.bbox)
        self._draw_animated()

    def add_artist(self, art):
        """Adds artist to the list of artsts to be animated"""
        if art.figure != self.canvas.figure:
            raise RuntimeError
        art.set_animated(True)
        self._artists.append(art)

    def _draw_animated(self):
        """Draw all the animated artists"""
        fig = self.canvas.figure
        for a in self._artists:
            fig.draw_artist(a)

    def update_background(self):
        """Update background"""
        self.canvas.draw()

    def update_artists(self):
        """Update the screen with animated artists"""
        cv = self.canvas
        fig = cv.figure
        # paranoia in case we missed the draw event,
        if self._bg is None:
            self.on_draw(None)
        else:
            # restore the background
            cv.restore_region(self._bg)
            # draw all the animated artists
            self._draw_animated()
            # update the GUI state
            cv.blit(fig.bbox)
        # let the GUI event loop process anything it has to do
        cv.flush_events()
