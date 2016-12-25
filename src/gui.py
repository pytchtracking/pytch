import sys
import numpy as num
import math

from PyQt5 import QtCore as qc
from PyQt5 import QtGui as qg
from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QLabel, QMenu
from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QComboBox, QGridLayout
from PyQt5.QtWidgets import QAction, QSlider, QPushButton

import time
import logging

from pytch.two_channel_tuner import Worker
from pytch.data import getaudiodevices, sampling_rate_options
from pytch.core import Core

logger = logging.getLogger(__name__)


if sys.version_info < (3, 0):
    _buffer = buffer
else:
    _buffer = memoryview


class Projection(object):
    def __init__(self):
        self.xr = 0., 1.
        self.ur = 0., 1.

    def set_in_range(self, xmin, xmax):
        if xmax == xmin:
            xmax = xmin + 1.

        self.xr = xmin, xmax

    def get_in_range(self):
        return self.xr

    def set_out_range(self, umin, umax):
        if umax == umin:
            umax = umin + 1.

        self.ur = umin, umax

    def get_out_range(self):
        return self.ur

    def __call__(self, x):
        umin, umax = self.ur
        xmin, xmax = self.xr
        return umin + (x-xmin)*((umax-umin)/(xmax-xmin))

    def clipped(self, x):
        umin, umax = self.ur
        xmin, xmax = self.xr
        return min(umax, max(umin, umin + (x-xmin)*((umax-umin)/(xmax-xmin))))

    def rev(self, u):
        umin, umax = self.ur
        xmin, xmax = self.xr
        return xmin + (u-umin)*((xmax-xmin)/(umax-umin))

    def copy(self):
        return copy.copy(self)


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
        #self.repaint()

    def sizeHint(self):
        return qc.QSize(200, 200)


class MainWindow(QMainWindow):
    ''' Top level Window. The entry point of the gui.'''
    def __init__(self, *args, **kwargs):
        QMainWindow.__init__(self, *args, **kwargs)
        self.show()

    def sizeHint(self):
        return qc.QSize(800, 600)

    def keyPressEvent(self, key_event):
        ''' react on keyboard keys when they are pressed.'''
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

        self.play_button = QPushButton('Play')
        layout.addWidget(self.play_button)

        self.pause_button = QPushButton('Pause')
        layout.addWidget(self.pause_button)

        layout.addWidget(QLabel('Select Input Device'))
        self.select_input = QComboBox()
        layout.addWidget(self.select_input)
        self.set_input_devices()

        layout.addWidget(QLabel('Sampling rate'))
        self.select_sampling_rate = QComboBox()
        layout.addWidget(self.select_sampling_rate)
        self.set_input_sampling_rates()

        layout.addWidget(QLabel('NFFT'))
        self.nfft_slider = self.get_nfft_slider()
        layout.addWidget(self.nfft_slider)

    def get_nfft_slider(self):
        ''' Return a QSlider for modifying FFT width'''
        nfft_slider = QSlider(qc.Qt.Horizontal)
        nfft_slider.setRange(256, 256*8)
        nfft_slider.setValue(512)
        nfft_slider.setTickInterval(256)
        return nfft_slider

    def set_input_devices(self):
        ''' Query device list and set the drop down menu'''
        devices = getaudiodevices()
        curr = 0
        for idevice, device in enumerate(devices):
            self.select_input.addItem('%s: %s' % (idevice, device))
            if 'default' in device:
                curr = idevice

        self.select_input.setCurrentIndex(idevice)

    def set_input_sampling_rates(self):
        ''' Set input sampling rates in drop down menu'''
        print('TEST')
        for sr in sampling_rate_options(self.select_input.currentIndex()):
            print(sr)
            self.select_sampling_rate.addItem(str(sr))


class CanvasWidget(QWidget):
    def __init__(self, *args, **kwargs):
        QWidget.__init__(self, *args, **kwargs)
        self.layout = QGridLayout()
        self.setLayout(self.layout)


class MainWidget(QWidget):
    ''' top level widget covering the central widget in the MainWindow.'''

    processingFinished = qc.pyqtSignal()

    def __init__(self, *args, **kwargs):
        tstart = time.time()
        QWidget.__init__(self, *args, **kwargs)

        self.setMouseTracking(True)
        top_layout = QVBoxLayout()
        self.setLayout(top_layout)

        self.menu = MenuWidget(parent=self)
        top_layout.addWidget(self.menu)

        self.core = Core()
        self.core.device_no = self.menu.select_input.currentIndex()
        self.core.worker.processingFinished = self.processingFinished
        self.menu.select_input.activated.connect(self.core.set_device_no)

        canvas = CanvasWidget(parent=self)
        canvas.dx = 1./self.core.data_input.sampling_rate

        self.spectrum1 = PlotLogWidget(parent=canvas)
        canvas.layout.addWidget(self.spectrum1, 1, 0)

        self.spectrum2 = PlotLogWidget(parent=canvas)
        canvas.layout.addWidget(self.spectrum2, 2, 0)

        self.trace1 = PlotWidget(parent=canvas)
        canvas.layout.addWidget(self.trace1, 1, 1)

        self.trace2 = PlotWidget(parent=canvas)
        canvas.layout.addWidget(self.trace2, 2, 1)

        #self.pitch1 = PlotPointsWidget(parent=canvas)
        self.pitch1 = PlotWidget(parent=canvas)
        canvas.layout.addWidget(self.pitch1, 3, 0, 1, 2)

        #self.pitch2 = PlotPointsWidget(parent=canvas)
        self.pitch2 = PlotWidget(parent=canvas)
        canvas.layout.addWidget(self.pitch2, 3, 0, 1, 2)

        top_layout.addWidget(canvas)

        self.make_connections()

        self.core.start()

        #self.worker.start_new_stream()

    def make_connections(self):
        #self.core.signalReady.connect(self.refreshwidgets)
        #self.processingFinished.connect(self.refreshwidgets)
        self.refresh_timer = qc.QTimer()
        self.refresh_timer.timeout.connect(self.refreshwidgets)
        self.refresh_timer.start(105)
        self.menu.nfft_slider.valueChanged.connect(self.core.worker.set_fft_length)
        self.menu.pause_button.clicked.connect(self.core.data_input.stop)
        self.menu.play_button.clicked.connect(self.core.data_input.start)
        logger.debug('connections made')

    def refreshwidgets(self):
        tstart = time.time()
        print('start', tstart)
        w = self.core.worker
        #self.spectrum1.plot(w.freqs, num.abs(w.ffts[0]))
        #self.spectrum2.plot(w.freqs, num.abs(w.ffts[1]))

        #ju#n = num.shape(w.current_frame1)[0]
        #xt = num.linspace(0, self.spectrum1.width(), n)
        #y1 = num.asarray(w.frames[0], dtype=num.float32)
        #y2 = num.asarray(w.frames[1], dtype=num.float32)
        #self.trace1.plot(xt, y1)
        #self.trace2.plot(xt, y2)

        ##############
        self.trace1.plot(*w.frames[0].latest_frame(2))
        self.trace2.plot(*w.frames[1].latest_frame(2))
        ##############
        #self.trace2.plot(w.frames[1].xdata, w.frames[1].ydata)

        #y1 = num.asarray(w.current_frame1, dtype=num.float32)
        #y2 = num.asarray(w.current_frame2, dtype=num.float32)
        #self.pitch1.plot(
        #    w.pitchlog1, num.abs(w.pitchlog_vect1-w.pitchlog_vect2))
        #self.pitch2.plot(
        #    w.pitchlog1, num.abs(w.pitchlog_vect2-w.pitchlog_vect1))
        #self.pitch1.plot(
        #    #w.pitchlogs[0].xdata, num.log(num.abs(w.pitchlogs[0].ydata)))
        #    w.pitchlogs[0].xdata, num.abs(w.pitchlogs[0].ydata))
        #self.pitch2.plot(
            #w.pitchlog1, num.log(num.abs(w.pitchlog_vect2)))
        self.repaint()
        tstop = time.time()
        logger.debug('refreshing widgets took %s seconds' % (tstop-tstart))
        t1 = time.time()
        print('tdraw', t1-time.time())

    def closeEvent(self, ev):
        '''Called when application is closed.'''
        logger.debug('closing')
        self.core.data_input.terminate()
        QWidget.closeEvent(self, ev)


def normalized_to01(d):
    ''' normalize data vector *d* between 0 and 1'''
    dmin = num.min(d)
    dmax = num.max(d)
    return (d-dmin)/(dmax-dmin)


class PlotWidget(QWidget):
    ''' a plotwidget displays data (x, y coordinates). '''

    def __init__(self, buffer=None, *args, **kwargs):
        QWidget.__init__(self, *args, **kwargs)
        layout = QHBoxLayout()
        label = QLabel('Trace 1')
        layout.addWidget(label)
        self.setLayout(layout)
        self.setContentsMargins(1, 1, 1, 1)

        #self.set_pen_color(41, 1, 255)
        self.set_pen_color(qc.Qt.black)

        self.track_start = None
        self.yscale_mode = None
        self.xscale_mode = None

        self.yscale = 1E-4

        self.right_click_menu = QMenu(self)

        select_scale = QAction('asdf', self.right_click_menu)
        select_scale.triggered.connect(self.set_yscale_mode)
        self.data_rect = self.rect()
        self.pen_width = 10.
        self.addAction(select_scale)
        self._xvisible = num.empty(0)
        self._yvisible = num.empty(0)
        self.initialized = False

    def test_refresh(self, npoints=100, ntimes=100):
        tstart = time.time()
        for i in range(ntimes):
            self.plot(num.random.random(npoints))
            self.repaint()
        tstop = time.time()
        print(tstop-tstart)

    def keyPressEvent(self, key_event):
        ''' react on keyboard keys when they are pressed.'''
        key_text = key_event.text()
        if key_text == 'q':
            self.close()

        elif key_text == 'f':
            self.showMaximized()

        QMainWindow.keyPressEvent(self, key_event)

    def set_pen_color(self, *rgb):
        self.color = qg.QColor(*rgb)

    def set_yscale_mode(self, mode):
        self.yscale_mode = mode

    def set_xscale_mode(self, mode):
        self.xscale_mode = mode

    def plot(self, xdata=None, ydata=None):
        ''' plot data

        :param *args:  ydata | xdata, ydata
        '''
        xmin, xmax = num.min(xdata), num.max(xdata)
        ymin, ymax = num.min(ydata), num.max(ydata)
        self.data_rect.setCoords(xmin, ymin, xmax, ymax)
        self.set_line_width()
        self._xvisible = xdata
        self._yvisible = ydata

    def set_xlim(self, xmin, xmax):
        ''' Set x data range '''
        self.data_rect.setLeft(xmin)
        self.data_rect.setRight(xmax)

    def set_ylim(self, ymin, ymax):
        ''' Set x data range '''
        # swap ?
        #self.data_rect.setBottom(ymax)
        #self.data_rect.setTop(ymin)
        self.data_rect.setBottom(ymin)
        self.data_rect.setTop(ymax)

    def set_line_width(self, width=3.):
        ''' Set width of the pen in pixels.'''
        self.pen_width = width / self.width()

    def paintEvent(self, e):
        ''' this is executed e.g. when self.repaint() is called. Draws the
        underlying data and scales the content to fit into the widget.'''
        draw_trace_num(e)

    def draw_trace_num(self, e):
        painter = qg.QPainter(self)
        pen = qg.QPen(self.color, self.pen_width, qc.Qt.SolidLine)
        painter.setPen(pen)
        x = self._xvisible
        y = self._yvisible
        
        qpoints = make_QPolygonF(normalized_to01(self._xvisible)*self.width(),
                                 normalized_to01(self._yvisible)*self.height())

        #painter.setRenderHint(qg.QPainter.Antialiasing)
        #painter.fillRect(self.data_rect, qg.QBrush(qc.Qt.white))

        painter.drawPolyline(qpoints)

    def draw_trace_qt(self, e):
        painter = qg.QPainter(self)
        painter.setWindow(self.data_rect)
        painter.setPen(qg.QPen(self.color, self.pen_width, qc.Qt.SolidLine))
        qpoints = make_QPolygonF(self._xvisible, self._yvisible)

        #painter.setRenderHint(qg.QPainter.Antialiasing)
        #painter.fillRect(self.data_rect, qg.QBrush(qc.Qt.white))

        painter.drawPolyline(qpoints)

    def mousePressEvent(self, mouse_ev):
        self.test_refresh()
        point = self.mapFromGlobal(mouse_ev.globalPos())
        self.track_start = (point.x(), point.y())
        #if mouse_ev.button() == qc.Qt.RightButton:
        #    self.right_click_menu.exec_(qg.QCursor.pos())
        #elif mouse_ev.button() == qc.Qt.LeftButton:
        #    self.parent().worker.stop()

    def mouseReleaseEvent(self, mouse_event):
        self.track_start = None

    def mouseMoveEvent(self, mouse_ev):
        point = self.mapFromGlobal(mouse_ev.globalPos())
        if self.track_start:
            x0, y0 = self.track_start
            dy = (point.y() - y0) / float(self.height())
            self.istart += dy*self.width()
            self.istop -= dy*self.width()

        self.update_indices()
        #self.repaint()


class PlotBuffer(PlotWidget):
    ''' plot the content of a Buffer instance.

    This class represents an interface between a Buffer and a PlotWidget.'''
    def __init__(self, *args, **kwargs):
        PlotWidget.__init__(self, *args, **kwargs)


class PlotPointsWidget(PlotWidget):
    ''' delta pitch widget'''
    def __init__(self, *args, **kwargs):
        PlotWidget.__init__(self, *args, **kwargs)
        self.yscale = 1E-1

    def paintEvent_asdf(self, e):
        painter = qg.QPainter(self)
        pen = qg.QPen(self.color, 4, qc.Qt.SolidLine)
        painter.setPen(pen)

        xdata = num.asarray(self._xvisible, dtype=num.float)
        ydata = self._yvisible
        xdata /= xdata[-1]
        ydata *= self.yscale

        #ydata = (ydata + 0.5) * self.height() * self.yscale
        ydata = ydata * self.height()
        qpoints = make_QPolygonF(xdata*self.width(), ydata)
        painter.drawPoints(qpoints)


class PlotLogWidget(PlotWidget):

    def __init__(self, *args, **kwargs):
        PlotWidget.__init__(self, *args, **kwargs)
        self.yscale = 1./15

    def paintEvent_ASDf(self, e):
        ''' This is executed e.g. when self.repaint() is called. Draws the
        underlying data and scales the content to fit into the widget.'''
        painter = qg.QPainter(self)
        pen = qg.QPen(self.color, 1, qc.Qt.SolidLine)
        painter.setPen(pen)
        xdata = self._xvisible#[self.istart:self.istop]
        ydata = self._yvisible#[self.istart:self.istop]
        xdata = normalized_to01(xdata)
        ##xdata = num.log(xdata)
        #xdata /= xdata[self.istop]
        #xdata = self._xvisible
        ydata = num.log(self._yvisible)
        #ydata = self._yvisible
        #ydata = normalized_to01(ydata)

        ydata *= self.height() * self.yscale
        xdata *= float(self.width())
        qpoints = make_QPolygonF(xdata[1:], ydata[1:])

        painter.drawPolyline(qpoints)


def make_QPolygonF(xdata, ydata):
    '''Create a :py:class:`qg.QPolygonF` instance from xdata and ydata, both
    numpy arrays.'''
    assert len(xdata) == len(ydata)

    nydata = len(ydata)
    qpoints = qg.QPolygonF(nydata)
    vptr = qpoints.data()
    vptr.setsize(int(nydata*8*2))
    aa = num.ndarray(
        shape=(nydata, 2),
        dtype=num.float64,
        buffer=_buffer(vptr))
    aa.setflags(write=True)
    aa[:, 0] = xdata
    aa[:, 1] = ydata

    return qpoints


def mean_decimation(d, ndecimate):
    ''' Decimate signal by factor (int) *ndecimate* using averaging.'''
    pad_size = math.ceil(float(d.size)/ndecimate)*ndecimate - d.size
    d = num.append(d, num.zeros(pad_size)*num.nan)
    return num.nanmean(d.reshape(-1, ndecimate), axis=1)


def from_command_line(close_after=None):
    ''' Start the GUI from comamand line'''
    app = QApplication(sys.argv)
    window = MainWindow()
    main_widget = MainWidget()
    window.setCentralWidget(main_widget)

    if close_after:
        close_timer = qc.QTimer()
        close_timer.timeout.connect(window.close)
        close_timer.start(close_after)

    sys.exit(app.exec_())


if __name__ == '__main__':
    from_command_line()
