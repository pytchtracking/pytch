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

    def keyPressEvent(self, key_event):
        ''' React on keyboard keys when they are pressed.

        Blocked by menu'''
        key_text = key_event.text()
        if key_text == 'q':
            self.close()

        elif key_text == 'f':
            self.showMaximized()
        QMainWindow.keyPressEvent(self, key_event)


class MenuWidget(QWidget):
    ''' TODO: Blocks keypressevents!'''
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
    ''' Top level widget covering the central widget in the MainWindow.'''

    signalStatus = qc.pyqtSignal()

    def __init__(self, *args, **kwargs):
        tstart = time.time()
        QWidget.__init__(self, *args, **kwargs)

        top_layout = QVBoxLayout()
        self.setLayout(top_layout)

        #autoGainCheckBox = QCheckBox(checked=True)
        #top_layout.addWidget(autoGain)
        #hbox_gain.addWidget(autoGainCheckBox)

        # reference to checkbox
        #self.autoGainCheckBox = autoGainCheckBox

        #hbox_fixedGain = QHBoxLayout()
        #fixedGain = QLabel('Fixed gain level')
        #self.fixedGainSlider = QSlider(QtCore.Qt.Horizontal)
        #hbox_fixedGain.addWidget(fixedGain)
        #hbox_fixedGain.addWidget(self.fixedGainSlider)
        menu = MenuWidget(parent=self)
        top_layout.addWidget(menu)

        canvas = CanvasWidget(parent=self)
        self.trace1 = PlotLogWidget(parent=canvas)
        canvas.layout.addWidget(self.trace1, 1, 0)

        self.trace2 = PlotLogWidget(parent=canvas)
        canvas.layout.addWidget(self.trace2, 2, 0)

        self.trace3 = PlotWidget(parent=canvas)
        canvas.layout.addWidget(self.trace3, 1, 1)

        self.trace4 = PlotWidget(parent=canvas)
        canvas.layout.addWidget(self.trace4, 2, 1)

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
        self.trace1.draw_trace(self.worker.freq_vect1,
                           num.abs(self.worker.fft_frame1))
        self.trace2.draw_trace(self.worker.freq_vect2,
                           num.abs(self.worker.fft_frame2))
        n = num.shape(self.worker.current_frame1)[0]
        xt = num.linspace(0, self.trace3.width(), n)
        y1 = num.asarray(self.worker.current_frame1, dtype=num.float32)
        y2 = num.asarray(self.worker.current_frame2, dtype=num.float32)
        self.trace3.draw_trace(xt, y1)
        self.trace4.draw_trace(xt, y2)
        self.repaint()


#class CanvasWidget(QWidget):
#    ''' Contains the data viewers'''

class PlotWidget(QWidget):
    ''' A PlotWidget displays data (x, y coordinates). '''

    signalStatus = qc.pyqtSignal()

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
        ''' This is executed e.g. when self.repaint() is called. Draws the
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
        #qpoints = make_QPolygonF(xdata*self.width(), ydata)

        #scale = 1E-3
        #stransform = qg.QTransform()
        #stransform.scale(width, height)

        #ttransform = qg.QTransform()
        #ttransform.translate(0., self.geometry().center().y())

        #transform = stransform * ttransform
        #painter.setTransform(ttransform)
        painter.drawPolyline(qpoints)

    def sizeHint(self):
        return qc.QSize(100, 100)


class PlotLogWidget(PlotWidget):

    def __init__(self, *args, **kwargs):
        PlotWidget.__init__(self, *args, **kwargs)
        self.scale = 1./15

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

        ydata *= self.height() * self.scale
        xdata *= self.width()
        #qpoints = make_QPolygonF(xdata, ydata)
        qpoints = make_QPolygon(xdata, ydata)

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
