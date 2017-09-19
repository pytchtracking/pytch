import logging
import math
import pyaudio
import numpy as num

from PyQt5 import QtCore as qc
from PyQt5 import QtGui as qg
from PyQt5 import QtWidgets as qw

from pytch.gui_util import _colors


logger = logging.getLogger(__name__)


keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
_semitones = [1, 3, 6, 8, 10]


def f2midi(f, standard_frequency=440.):
    ''' https://en.wikipedia.org/wiki/MIDI_tuning_standard '''
    return int(69 + 12*num.log2(f/standard_frequency))


class Synthy(qc.QObject):

    def __init__(self):
        qc.QObject.__init__(self)

        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paFloat32,
                                  channels=1, rate=44100, output=1)
        self.stream.start_stream()
        self.stop_playing = False

    def sine(self, frequency, length, rate):
        length = int(length * rate)
        factor = frequency * math.pi * 2. / rate
        return num.sin(num.arange(length) * factor)

    def play_tone(self, frequency=440, length=1., rate=44100):
        chunks = []
        y = num.zeros(int(length*rate))
        for i in range(3):
            y += (self.sine(frequency*(i+1), length, rate) / (i+1))

        chunks.append(y)
        chunk = num.concatenate(chunks) * 0.25
        self.stream.write(chunk.astype(num.float32).tostring())

    @qc.pyqtSlot()
    def stop(self):
        self.stop_playing = True

    @qc.pyqtSlot(float)
    def play(self, frequency):
        self.stop_playing = False
        while not self.stop_playing:
            self.play_tone(frequency)

    def close(self):
        self.stream.stop_stream()
        self.p.terminate()


class Key(qw.QWidget):

    playKeyBoard = qc.pyqtSignal(float)
    stopKeyBoard = qc.pyqtSignal()

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
        self.f = 2.**((self.semitone + self.octave*12.)/12.) * 130.81
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
        painter = qg.QPainter(self)
        pen, brush = self.get_pen_brush()
        rect = self.rect()
        painter.setPen(pen)
        painter.fillRect(rect, brush)
        painter.drawRect(rect)
        if self.want_label:
            x1, y1, x2, y2 = rect.getCoords()
            painter.save()
            if self.is_semitone:
                painter.setPen(qg.QPen(qc.Qt.white))
            painter.drawText(x1 + (x2-x1)/2, y2, self.name)
            painter.restore()

    @qc.pyqtSlot(qg.QMouseEvent)
    def mousePressEvent(self, mouse_ev):
        if mouse_ev.button() == qc.Qt.LeftButton:
            self.pressed = True
            self.playKeyBoard.emit(self.f)
        else:
            super(Key, self).mousePressEvent(mouse_ev)
        self.update()

    @qc.pyqtSlot(qg.QMouseEvent)
    def mouseReleaseEvent(self, mouse_ev):
        self.stopKeyBoard.emit()
        self.pressed = False
        self.update()

    @qc.pyqtSlot()
    def on_toggle_labels(self):
        self.want_label += 1
        self.want_label %= 2
        self.repaint()


class KeyBoard(qw.QWidget):

    keyBoardKeyReleased = qc.pyqtSignal()
    keyBoardKeyPressed = qc.pyqtSignal(float)
    toggle_tabels = qc.pyqtSignal()

    def __init__(self, *args, **kwargs):
        super(KeyBoard, self).__init__(*args, **kwargs)

        self.setContentsMargins(1, 1, 1, 1)
        self.setMaximumHeight(200)
        self.setMinimumHeight(100)
        self.n_octaves = 3

        self.synthy = Synthy()
        self.synthy_thread = qc.QThread()
        self.synthy.moveToThread(self.synthy_thread)
        self.synthy_thread.start()

        self.keys = []
        self.setup_right_click_menu()
        self.setup()

    def setup_right_click_menu(self):
        self.right_click_menu = qw.QMenu('Keyboard Settings', self)
        action = qw.QAction(str('Toggle labels'), self.right_click_menu)
        action.triggered.connect(self.toggle_tabels)
        self.right_click_menu.addAction(action)

    def setup(self):
        rects = self.get_key_rects()
        for ir, r in enumerate(rects):
            noctave = int(ir/12)
            key = Key(octave=noctave, semitone=ir % 12, parent=self)
            key.playKeyBoard.connect(self.keyBoardKeyPressed)
            key.playKeyBoard.connect(self.synthy.play)
            key.stopKeyBoard.connect(self.stop_synthy)
            self.toggle_tabels.connect(key.on_toggle_labels)
            self.keys.append(key)

    def stop_synthy(self):
        self.synthy.stop()
        self.keyBoardKeyPressed.emit(0)

    def get_key_rects(self):
        ''' Get rectangles for tone keys'''
        n = 14 * self.n_octaves
        deltax = self.width()/n
        y = self.height()
        y_semi = self.height() * 0.6
        rects = []
        semitone = 0
        for i in range(n):
            if i % 14 in [5, 13]:
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
        ''' required to move all :py:class:`pytch.keyboard.Key` instances to
        their approproate locations'''
        rects = self.get_key_rects()
        for r, k in zip(rects, self.keys):
            k.setGeometry(r)

    def connect_channel_views(self, channel_views_widget):
        ''' Connect Keyboard's signals to channel views.

        :param channel_views_widget: instance of
            :py:class:`pytch.gui.ChannelViewsWidget
        '''
        for cv in channel_views_widget.views:
            self.keyBoardKeyPressed.connect(cv.on_keyboard_key_pressed)

    def mousePressEvent(self, mouse_ev):
        if mouse_ev.button() == qc.Qt.RightButton:
            self.right_click_menu.exec_(mouse_ev.pos())
        else:
            super(KeyBoard, self).mousePressEvent(mouse_ev)
