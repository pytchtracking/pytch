import sys
import numpy as num

from PyQt5 import QtCore as qc
from PyQt5 import QtGui as qg
from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QLabel
from PyQt5.QtWidgets import QMainWindow, QVBoxLayout

import time
import logging

from two_channel_tuner import LiveFFTWidget, Worker

logger = logging.getLogger(__name__)


class GaugeWidget(QWidget):
    def __init__(self, *args, **kwargs):
        '''
        '''
        QWidget.__init__(self, *args, **kwargs)
        hbox_fixedGain = QVBoxLayout()
        fixedGain = QLabel('Gauge')
        hbox_fixedGain.addWidget(fixedGain)
        self.setLayout(hbox_fixedGain)
        self.clip = None
        self.rectf = qc.QRectF(10., 10., 100., 100.)
        self.color = qg.QColor(0, 0, 0)
        self.clip_color = qg.QColor(255, 0, 0)
        self._val = 0

    def set_clip(self, clip_value):
        ''' Set a clip value'''
        self.clip = clip_value

    def paintEvent(self, e):
        ''' This is executed when self.repaint() is called'''
        painter = qg.QPainter(self)
        if self._val<self.clip and self.clip:
            color = self.color
        else:
            color = self.clip_color
        pen = qg.QPen(color, 20, qc.Qt.SolidLine)
        painter.setPen(pen)
        painter.drawArc(self.rectf, 2880., self._val)
        #painter.drawPie(self.rectf, 2880., self._val)

    def update_value(self, val):
        '''
        Call this method to update the arc
        '''
        if self.clip:
            self._val = min(math.log(val)/math.log(self.clip)*2880., 2880)  # 2880=16*180 (half circle)
        else:
            self._val = math.log(val) * 100
        self.repaint()

    def sizeHint(self):
        return qc.QSize(200, 200)


class MainWindow(QMainWindow):
    ''' Top level Window. The entry point of the gui.'''
    def __init__(self, *args, **kwargs):
        QMainWindow.__init__(self, *args, **kwargs)

        self.setCentralWidget(MainWidget(self))
        self.show()

    def keyPressEvent(self, key_event):
        ''' React on keyboard keys when they are pressed.'''
        key_text = key_event.text()
        if key_text == 'q':
            self.close()

        elif key_text == 'f':
            print 'maximize'
            self.showMaximized()
        QWidget.keyPressEvent(self, key_event)


class MainWidget(QWidget):
    ''' Top level widget covering the central widget in the MainWindow.'''

    signalStatus = qc.pyqtSignal()

    def __init__(self, *args, **kwargs):
        tstart = time.time()
        QWidget.__init__(self, *args, **kwargs)

        top_layout = QVBoxLayout()
        self.setLayout(top_layout)
        label = QLabel('PLAYGROUND')
        top_layout.addWidget(label)

        self.trace1 = TraceWidget(parent=self)
        top_layout.addWidget(self.trace1)

        self.trace2 = TraceWidget(parent=self)
        top_layout.addWidget(self.trace2)

        self.worker = Worker()

        self.make_connections()

        self.worker.start()
        self.start_drawing()

    def make_connections(self):
        self.worker.signalReady.connect(self.refreshwidgets)

        #self.connect(self.worker, pyqtSignal('ready'), self.refreshwidgets)

    def start_drawing(self):

        self.timer = qc.QTimer()
        self.timer.timeout.connect(self.refreshwidgets)
        self.timer.start(100)

    def refreshwidgets(self):
        self.trace1.draw_trace(num.abs(self.worker.freq_vect1),
                               num.abs(self.worker.fft_frame1))
        #self.trace2.draw_trace(self.worker.freq_vect2, self.worker.fft_frame2)
        self.repaint()



class TraceWidget(QWidget):
    ''' A TraceWidget displays data (x, y coordinates). '''

    signalStatus = qc.pyqtSignal()

    def __init__(self, *args, **kwargs):
        QWidget.__init__(self, *args, **kwargs)
        layout = QHBoxLayout()
        label = QLabel('Trace 1')
        layout.addWidget(label)
        self.setLayout(layout)
        self.setContentsMargins(1, 1, 1, 1)

        self.color = qg.QColor(4, 1, 255)
        self._xvisible = num.zeros(1)
        self._yvisible = num.zeros(1)

    def paintEvent(self, e):
        ''' This is executed e.g. when self.repaint() is called. Draws the
        underlying data and scales the content to fit into the widget.'''
        painter = qg.QPainter(self)
        pen = qg.QPen(self.color, 0.001, qc.Qt.SolidLine)
        painter.setPen(pen)
        self._xvisible /= self._xvisible.max()
        self._yvisible = num.log(self._yvisible)
        self._yvisible /= self._yvisible.max()


        qpoints = make_QPolygonF(self._xvisible, self._yvisible)

        height = self.height()
        width = self.width()

        stransform = qg.QTransform()
        stransform.scale(width/self._xvisible.max(), height/self._yvisible.max())

        #ttransform = qg.QTransform()
        #ttransform.translate(0., self.geometry().center().y())

        #transform = stransform * ttransform
        painter.setTransform(stransform)
        painter.drawPolyline(qpoints)

        #pen = qg.QPen(self.color, 0.1, qc.Qt.SolidLine)
        #painter.setPen(pen)
        #painter.drawPoints(qpoints)

    def draw_trace(self, xdata, ydata):
        '''
        Call this method to update the arc
        '''
        self._yvisible = ydata
        self._xvisible = xdata

    def sizeHint(self):
        return qc.QSize(100, 100)


def make_QPolygonF(xdata, ydata):
    '''Create a :py:class:`qg.QPolygonF` instance from xdata and ydata, both
    numpy arrays.'''
    assert len(xdata) == len(ydata)
    qpoints = qg.QPolygonF(len(ydata))
    vptr = qpoints.data()
    vptr.setsize(len(ydata)*8*2)
    aa = num.ndarray(
        shape=(len(ydata), 2),
        dtype=num.float64,
        buffer=buffer(vptr))
    aa.setflags(write=True)
    aa[:, 0] = xdata
    aa[:, 1] = ydata
    return qpoints


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
