import logging

from PyQt5 import QtCore as qc
from PyQt5 import QtGui as qg
from PyQt5.QtWidgets import QWidget

from pytch.gui_util import _colors


logger = logging.getLogger(__name__)


keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
_semitones = [1, 3, 6, 8, 10]


class Key(QWidget):

    playKeyBoard = qc.pyqtSignal(int)

    def __init__(self, octave, semitone, *args, **kwargs):
        super(Key, self).__init__(*args, **kwargs)
        self.setContentsMargins(1, 1, 1, 1)
        self.octave = octave
        self.semitone = semitone
        self.pressed = False
        self.want_label = True
        self.setup()

    def setup(self):
        self.is_semitone = self.semitone in _semitones
        self.f = 2**((self.semitone + self.octave*12)/12) * 220
        self.name = keys[self.semitone]
        # self.static_label = qc.QStaticText(self.name)
        self.brush_pressed = qg.QBrush(qg.QColor(*_colors['aluminium4']))
        if self.is_semitone:
            self.brush = qg.QBrush(qc.Qt.black)
            self.pen = qg.QPen(qc.Qt.white)
        else:
            self.brush = qg.QBrush(qc.Qt.white)
            self.pen = qg.QPen(qc.Qt.black)

    def get_pen_brush(self):
        if self.pressed:
            return (self.pen, self.brush_pressed)
        return (self.pen, self.brush)

    def paintEvent(self, event):
        painter= qg.QPainter(self)
        pen, brush = self.get_pen_brush()
        rect = self.rect()
        painter.setPen(pen)
        painter.fillRect(rect, brush)
        painter.drawRect(rect)
        if self.want_label:
            x1, y1, x2, y2 = rect.getCoords()
            # painter.drawStaticText(self.static_label)
            painter.save()
            if self.is_semitone:
                painter.setPen(qg.QPen(qc.Qt.white))
            painter.drawText(x1 + (x2-x1)/2, y2, self.name)
            painter.restore()

    def mousePressEvent(self, mouse_ev):
        point = self.mapFromGlobal(mouse_ev.globalPos())

        if mouse_ev.button() == qc.Qt.LeftButton:
            self.pressed = True
            self.playKeyBoard.emit(self.f)
        else:
            QWidget.mousePressEvent(mouse_ev)
        self.repaint()

    def mouseReleaseEvent(self, mouse_ev):
        self.playKeyBoard.emit(0)
        self.pressed = False
        self.repaint()


class KeyBoard(QWidget):

    keyBoardKeyPressed = qc.pyqtSignal(int)

    def __init__(self, *args, **kwargs):
        super(KeyBoard, self).__init__(*args, **kwargs)

        self.setContentsMargins(1, 1, 1, 1)
        self.setMaximumHeight(200)
        self.setMinimumHeight(100)
        self.n_octaves = 3
        self.keys = []
        self.setup()

    def setup(self):
        rects = self.get_key_rects()
        for ir, r in enumerate(rects):
            noctave = int(ir/12)
            key = Key(octave=noctave, semitone=ir%12, parent=self)
            key.playKeyBoard.connect(self.keyBoardKeyPressed)
            self.keys.append(key)

    def get_key_rects(self):
        ''' Get rectangles for tone keys'''
        n = 14 * self.n_octaves
        deltax = self.width()/n
        y = self.height()
        y_semi = self.height() * 0.6
        top_lefts = []
        rects = []
        x = 0
        semitone = 0
        for i in range(n):
            if i%14 in [5, 13]:
                continue
            if semitone % 12 in _semitones:
                rect = qc.QRect(
                    qc.QPoint(i * deltax, 0),
                    qc.QPoint(i * deltax + deltax*2, y_semi)
                )
            else:
                rect = qc.QRect(
                    qc.QPoint(i * deltax, 0),
                    qc.QPoint(i * deltax + deltax*2, y)
                )
            rects.append(rect)
            semitone += 1
        return rects

    def resizeEvent(self, event):
        rects = self.get_key_rects()
        for r, k in zip(rects, self.keys):
            k.setGeometry(r)

    def connect_channel_views(self, channel_views_widget):
        for cv in channel_views_widget.channel_views:
            self.keyBoardKeyPressed.connect(cv.on_keyboard_key_pressed)
