import sys
import numpy as num

from PyQt5 import QtCore as qc
from PyQt5 import QtGui as qg
from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QLabel
from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QComboBox, QGridLayout

import time
import logging

from two_channel_tuner import Worker, getaudiodevices

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

        self.setCentralWidget(MainWidget(parent=self))
        self.show()

    def keypressevent(self, key_event):
        ''' react on keyboard keys when they are pressed.

        blocked by menu'''
        key_text = key_event.text()
        if key_text == 'q':
            self.close()

        elif key_text == 'f':
            self.showMaximized()
        QMainWindow.keyPressEvent(self, key_event)


class MenuWidget(QWidget):
    ''' todo: blocks keypressevents!'''
    def __init__(self, *args, **kwargs):
        QWidget.__init__(self, *args, **kwargs)
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.select_input = QComboBox()
        devices = getaudiodevices()
        curr = 0
        for idevice, device in enumerate(devices):
            self.select_input.addItem('%s: %s' % (idevice, device))
            if 'default' in device:
                curr = idevice

        self.select_input.setCurrentIndex(idevice)

        layout.addWidget(self.select_input)


class CanvasWidget(QWidget):
    def __init__(self, *args, **kwargs):
        QWidget.__init__(self, *args, **kwargs)
        self.layout = QGridLayout()
        self.setLayout(self.layout)


class MainWidget(QWidget):
    ''' top level widget covering the central widget in the MainWindow.'''

    signalstatus = qc.pyqtSignal()

    def __init__(self, *args, **kwargs):
        tstart = time.time()
        QWidget.__init__(self, *args, **kwargs)

        top_layout = QVBoxLayout()
        self.setLayout(top_layout)

        #autogainCheckBox = QCheckBox(checked=True)
        #top_layout.addWidget(autoGain)
        #hbox_gain.addWidget(autoGainCheckBox)

        # reference to checkbox
        #self.autoGainCheckBox = autoGainCheckBox

        #hbox_fixedGain = QHBoxLayout()
        #fixedgain = QLabel('Fixed gain level')
        #self.fixedGainSlider = QSlider(QtCore.Qt.Horizontal)
        #hbox_fixedGain.addWidget(fixedGain)
        #hbox_fixedGain.addWidget(self.fixedGainSlider)
        menu = MenuWidget(parent=self)
        top_layout.addWidget(menu)

        canvas = CanvasWidget(parent=self)
        self.spectrum1 = PlotLogWidget(parent=canvas)
        canvas.layout.addWidget(self.spectrum1, 1, 0)

        self.spectrum2 = PlotLogWidget(parent=canvas)
        canvas.layout.addWidget(self.spectrum2, 2, 0)

        self.trace1 = PlotWidget(parent=canvas)
        canvas.layout.addWidget(self.trace1, 1, 1)

        self.trace2 = PlotWidget(parent=canvas)
        canvas.layout.addWidget(self.trace2, 2, 1)

        self.pitch1 = PlotPointsWidget(parent=canvas)
        canvas.layout.addWidget(self.pitch1, 3, 0, 1, 2)

        self.pitch2 = PlotPointsWidget(parent=canvas)
        canvas.layout.addWidget(self.pitch2, 3, 0, 1, 2)

        top_layout.addWidget(canvas)

        self.worker = Worker()
        self.worker.set_device_no(menu.select_input.currentIndex())
        menu.select_input.activated.connect(self.set_input)

        self.make_connections()

        self.worker.start()
        self.start_drawing()

    def set_input(self, i):
        self.worker.stop()
        self.worker.set_device_no(i)
        self.worker.start()

    def make_connections(self):
        self.worker.signalReady.connect(self.refreshwidgets)
        #self.connect(self.worker, pyqtSignal('ready'), self.refreshwidgets)

    def start_drawing(self):

        self.timer = qc.QTimer()
        self.timer.timeout.connect(self.refreshwidgets)
        self.timer.start(50)

    def refreshwidgets(self):
        w = self.worker
        self.spectrum1.draw_trace(w.freq_vect1,
                           num.abs(w.fft_frame1))
        self.spectrum2.draw_trace(w.freq_vect2,
                           num.abs(w.fft_frame2))

        n = num.shape(w.current_frame1)[0]
        xt = num.linspace(0, self.spectrum1.width(), n)
        y1 = num.asarray(w.current_frame1, dtype=num.float32)
        y2 = num.asarray(w.current_frame2, dtype=num.float32)
        self.trace1.draw_trace(xt, y1)
        self.trace2.draw_trace(xt, y2)

        y1 = num.asarray(w.current_frame1, dtype=num.float32)
        y2 = num.asarray(w.current_frame2, dtype=num.float32)
        #print w.pitchlog1
        #print w.pitchlog_vect1
        self.pitch1.draw_trace(
            w.pitchlog1, num.abs(w.pitchlog_vect1-w.pitchlog_vect2))
        self.pitch2.draw_trace(
            w.pitchlog1, num.abs(w.pitchlog_vect2-w.pitchlog_vect1))

        self.repaint()


#class canvaswidget(QWidget):
#    ''' contains the data viewers'''

class PlotWidget(QWidget):
    ''' a plotwidget displays data (x, y coordinates). '''

    signalstatus = qc.pyqtSignal()

    def __init__(self, *args, **kwargs):
        QWidget.__init__(self, *args, **kwargs)
        layout = QHBoxLayout()
        label = QLabel('Trace 1')
        layout.addWidget(label)
        self.setLayout(layout)
        self.setContentsMargins(1, 1, 1, 1)

        self.color = qg.QColor(4, 1, 255)
        self._xvisible = num.random.random(1)
        self._yvisible = num.random.random(1)
        self.yscale = 1E-4

    def draw_trace(self, xdata, ydata):
        self._yvisible = ydata
        self._xvisible = xdata

    def paintEvent(self, e):
        ''' this is executed e.g. when self.repaint() is called. Draws the
        underlying data and scales the content to fit into the widget.'''
        painter = qg.QPainter(self)
        pen = qg.QPen(self.color, 1, qc.Qt.SolidLine)
        painter.setPen(pen)

        xdata = self._xvisible
        ydata = self._yvisible
        xdata /= xdata[-1]
        ydata *= self.yscale

        ydata = (ydata + 0.5) * self.height()
        qpoints = make_QPolygon(xdata*self.width(), ydata)
        painter.drawPolyline(qpoints)

    def sizehint(self):
        return qc.QSize(100, 100)

class PlotPointsWidget(PlotWidget):
    ''' delta pitch widget'''
    def __init__(self, *args, **kwargs):
        PlotWidget.__init__(self, *args, **kwargs)

    def paintEvent(self, e):
        painter = qg.QPainter(self)
        pen = qg.QPen(self.color, 1, qc.Qt.SolidLine)
        painter.setPen(pen)

        xdata = self._xvisible
        ydata = self._yvisible
        print xdata, ydata
        xdata /= xdata[-1]
        ydata *= self.yscale

        ydata = (ydata + 0.5) * self.height()
        qpoints = make_QPolygon(xdata*self.width(), ydata)
        #qpoints = make_QPolygonF(xdata, ydata)
        painter.drawPoints(qpoints)


class PlotLogWidget(PlotWidget):

    def __init__(self, *args, **kwargs):
        PlotWidget.__init__(self, *args, **kwargs)
        self.yscale = 1./15

    def paintEvent(self, e):
        ''' This is executed e.g. when self.repaint() is called. Draws the
        underlying data and scales the content to fit into the widget.'''
        painter = qg.QPainter(self)
        pen = qg.QPen(self.color, 1, qc.Qt.SolidLine)
        painter.setPen(pen)
        xdata = self._xvisible
        ydata = self._yvisible

        xdata = num.log(xdata)
        xdata /= xdata[-1]

        ydata = num.log(ydata)

        ydata *= self.height() * self.yscale
        xdata *= self.width()
        qpoints = make_QPolygon(xdata[1:], ydata[1:])

        painter.drawPolyline(qpoints)


def make_QPolygon(xdata, ydata):
    '''Create a :py:class:`qg.QPolygonF` instance from xdata and ydata, both
    numpy arrays.'''
    assert len(xdata) == len(ydata)

    points = []

    for i in xrange(len(xdata)):
        points.append(qc.QPoint(xdata[i], ydata[i]))

    return qg.QPolygon(points)


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
