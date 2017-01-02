import sys
import numpy as num
import scipy.signal as signal
import math
import time
import logging
import copy

if False:
    from PyQt4 import QtCore as qc
    from PyQt4 import QtGui as qg
    from PyQt4.QtGui import QApplication, QWidget, QHBoxLayout, QLabel, QMenu
    from PyQt4.QtGui import QMainWindow, QVBoxLayout, QComboBox, QGridLayout
    from PyQt4.QtGui import QAction, QSlider, QPushButton, QDockWidget
else:
    from PyQt5 import QtCore as qc
    from PyQt5 import QtGui as qg
    from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QLabel, QMenu
    from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QComboBox, QGridLayout
    from PyQt5.QtWidgets import QAction, QSlider, QPushButton, QDockWidget

from pytch.two_channel_tuner import cross_spectrum
from pytch.data import getaudiodevices, sampling_rate_options
from pytch.core import Core
from pytch.gui_util import AutoScaler


logger = logging.getLogger(__name__)
use_pyqtgraph = False

if sys.version_info < (3, 0):
    _buffer = buffer
else:
    _buffer = memoryview


colors = {
    'black': (0, 0, 0),
    'white': (255, 255, 255),
    'red': (255, 0, 0),
    'green': (0, 255, 0),
    'blue': (0, 0, 255),
}

channel_to_color = [ 'blue', 'red', 'green', 'black']


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

    #def sizeHint(self):
    #    return qc.QSize(200, 200)


class MainWindow(QMainWindow):
    ''' Top level Window. The entry point of the gui.'''
    def __init__(self, *args, **kwargs):
        QMainWindow.__init__(self, *args, **kwargs)
        self.show()

    def keyPressEvent(self, key_event):
        ''' react on keyboard keys when they are pressed.'''
        key_text = key_event.text()
        if key_text == 'q':
            self.close()

        elif key_text == 'f':
            self.showMaximized()

        QMainWindow.keyPressEvent(self, key_event)

    def sizeHint(self):
        return qc.QSize(900, 600)

class MenuWidget(QWidget):
    ''' todo: blocks keypressevents!'''
    def __init__(self, *args, **kwargs):
        QWidget.__init__(self, *args, **kwargs)
        layout = QGridLayout()
        self.setLayout(layout)

        self.play_button = QPushButton('Play')
        layout.addWidget(self.play_button, 0, 0)

        self.pause_button = QPushButton('Pause')
        layout.addWidget(self.pause_button, 0, 1)

        layout.addWidget(QLabel('Select Input Device'), 1, 0)
        self.select_input = QComboBox()
        layout.addWidget(self.select_input, 1, 1)
        self.set_input_devices(self.select_input)

        layout.addWidget(QLabel('Sampling rate'), 2, 0)
        self.select_sampling_rate = QComboBox()
        layout.addWidget(self.select_sampling_rate, 2, 1)
        self.set_input_sampling_rates()

        layout.addWidget(QLabel('NFFT'), 3, 0)
        self.nfft_slider = self.get_nfft_box()
        layout.addWidget(self.nfft_slider, 3, 1)

    def get_nfft_box(self):
        ''' Return a QSlider for modifying FFT width'''
        b = QComboBox()

        for fft_factor in [2, 4, 8]:
            b.addItem('%s' % (512*fft_factor))

        b.setCurrentIndex(0)
        return b

    def set_input_devices(self, box):
        ''' Query device list and set the drop down menu'''
        devices = getaudiodevices()
        curr = 0
        for idevice, device in enumerate(devices):
            box.addItem('%s: %s' % (idevice, device))
            if 'default' in device:
                curr = idevice

        box.setCurrentIndex(idevice)

    def set_input_sampling_rates(self):
        ''' Set input sampling rates in drop down menu'''
        print('TEST')
        for sr in sampling_rate_options(self.select_input.currentIndex()):
            print(sr)
            #self.select_sampling_rate.addItem(str(sr))
            self.select_sampling_rate.addItem('ASDFASDF')


class CanvasWidget(QWidget):
    ''' Contains all visualizing elements'''
    def __init__(self, *args, **kwargs):
        QWidget.__init__(self, *args, **kwargs)
        self.layout = QGridLayout()
        self.setLayout(self.layout)

    def sizeHint(self):
        return qc.QSize(1200, 1200)


tfollow = 3.

class MainWidget(QWidget):
    ''' top level widget covering the central widget in the MainWindow.'''

    processingFinished = qc.pyqtSignal()

    def __init__(self, *args, **kwargs):
        #tstart = time.time()
        QWidget.__init__(self, *args, **kwargs)

        self.setMouseTracking(True)
        top_layout = QHBoxLayout()
        self.setLayout(top_layout)

        self.menu = MenuWidget()

        dock_widget_area = QDockWidget()
        dock_widget_area.setWidget(self.menu)
        top_layout.addWidget(self.menu)

        self.core = Core()
        self.core.device_no = self.menu.select_input.currentIndex()
        self.core.worker.processingFinished = self.processingFinished
        self.menu.select_input.activated.connect(self.core.set_device_no)

        canvas = CanvasWidget(parent=self)
        canvas.dx = 1./self.core.data_input.sampling_rate

        self.spectrum1 = PlotWidget(parent=canvas)
        self.spectrum2 = PlotWidget(parent=canvas)
        canvas.layout.addWidget(self.spectrum1, 1, 0)
        canvas.layout.addWidget(self.spectrum2, 2, 0)

        if use_pyqtgraph:
            import pyqtgraph as pg

            trace1_widget = pg.PlotWidget()
            self.trace1_qtgraph = trace1_widget.getPlotItem()
            canvas.layout.addWidget(trace1_widget, 1, 1)

            trace2_widget= pg.PlotWidget()
            self.trace2_qtgraph = trace2_widget.getPlotItem()
            canvas.layout.addWidget(trace2_widget, 2, 1)
        else:
            self.trace1 = PlotWidget(parent=canvas)
            self.trace2 = PlotWidget(parent=canvas)
            self.trace1.tfollow = 4.
            self.trace2.tfollow = 4.
            canvas.layout.addWidget(self.trace1, 1, 1)
            canvas.layout.addWidget(self.trace2, 2, 1)

        self.pitch = PlotWidget(parent=canvas)
        canvas.layout.addWidget(self.pitch, 3, 0, 1, 2)

        top_layout.addWidget(canvas)

        self.cross_spectrum = PlotWidget(parent=canvas)
        canvas.layout.addWidget(self.cross_spectrum, 4, 0, 1, 2)

        self.make_connections()
        self.core.start()

        #self.worker.start_new_stream()

    def make_connections(self):
        self.refresh_timer = qc.QTimer()
        self.refresh_timer.timeout.connect(self.refreshwidgets)
        self.refresh_timer.start(50)
        self.menu.nfft_slider.activated.connect(self.core.worker.set_fft_length)
        self.menu.pause_button.clicked.connect(self.core.data_input.stop)
        self.menu.play_button.clicked.connect(self.core.data_input.start)
        logger.debug('connections made')

    def refreshwidgets(self):
        #tstart = time.time()
        w = self.core.worker
        self.spectrum1.plotlog(w.freqs, num.abs(w.ffts[0]),
                            color=channel_to_color[0])
        self.spectrum2.plotlog(w.freqs, num.abs(w.ffts[1]),
                            color=channel_to_color[1])
        self.spectrum1.set_ylim(num.log(10.), num.log(100000.))
        self.spectrum2.set_ylim(num.log(10.), num.log(100000.))

        if use_pyqtgraph:
            self.trace1_qtgraph.clear()
            self.trace2_qtgraph.clear()
            self.trace1_qtgraph.plot(*w.frames[0].latest_frame(tfollow))
            self.trace2_qtgraph.plot(*w.frames[1].latest_frame(tfollow))
        else:
            self.trace1.plot(*w.frames[0].latest_frame(self.trace1.tfollow),
                             ndecimate=20, color=channel_to_color[0])
            self.trace2.plot(*w.frames[1].latest_frame(self.trace2.tfollow),
                             ndecimate=20, color=channel_to_color[1])
            self.trace1.set_ylim(-3000., 3000.)
            self.trace2.set_ylim(-3000., 3000.)

        self.pitch.plot(
            *w.pitchlogs[0].latest_frame(tfollow*2),
            symbol='o',
            color=channel_to_color[0])
        self.pitch.plot(
            *w.pitchlogs[1].latest_frame(tfollow*2),
            symbol='o',
            color=channel_to_color[1])

        self.cross_spectrum.plotlog(w.freqs, cross_spectrum(w.ffts[0], w.ffts[1])[0])
        #self.cross_spectrum.plot(w.freqs, cross_spectrum(w.ffts[0], w.ffts[1])[0])
        #self.cross_spectrum.set_ylim(0, 1E7)
        self.cross_spectrum.set_ylim(0.0001, num.log(1E8))

        self.repaint()
        #tstop = time.time()
        #logger.debug('refreshing widgets took %s seconds' % (tstop-tstart))

    def closeEvent(self, ev):
        '''Called when application is closed.'''
        logger.debug('closing')
        self.core.data_input.terminate()
        QWidget.closeEvent(self, ev)


def normalized_to01(d):
    ''' normalize data vector *d* between 0 and 1'''
    if len(d) == 0:
        return d
    dmin = num.min(d)
    dmax = num.max(d)
    return (d-dmin)/(dmax-dmin)



class PlotWidget(QWidget):
    ''' a plotwidget displays data (x, y coordinates). '''

    def __init__(self, buffer=None, *args, **kwargs):
        QWidget.__init__(self, *args, **kwargs)
        layout = QHBoxLayout()
        self.setLayout(layout)
        self.setContentsMargins(1, 1, 1, 1)

        self.set_pen_color('black')

        self.track_start = None

        self.yscale = 1.
        self.tfollow = 0

        self.ymin = None
        self.ymax = None
        self.xmin = None
        self.xmax = None

        self._ymin = 0.
        self._ymax = 1.
        self._xmin = 0.
        self._xmax = 1.

        self.left = 0.1
        self.right = 1.
        self.top = 1.
        self.bottom = 0.1

        self.right_click_menu = QMenu(self)

        self.yticks = None
        self.xticks = None

        self.yscaler = AutoScaler(
            no_exp_interval=(-3, 2), approx_ticks=7,
            snap=True
        )

        self.xscaler = AutoScaler(
            no_exp_interval=(-3, 2), approx_ticks=7,
            snap=True
        )

        select_scale = QAction('asdf', self.right_click_menu)
        self.pen_width = .0
        self.addAction(select_scale)
        self._xvisible = num.empty(0)
        self._yvisible = num.empty(0)
        self.yproj = Projection()
        self.xproj = Projection()

    def setup_annotation_boxes(self):
        ''' left and bottom boxes containing labels, dashes, marks, etc.'''
        w, h = self.wh
        l = self.left
        r = self.right
        t = self.top
        b = self.bottom

        tl = qc.QPoint((1.-b) * h , l * w)
        br = qc.QPoint(h, w * r)
        size = qc.QSize(w * (1. - (l + (1.-r))),
                        h * (1. - ((1.-t)+b)))
        #tl = qc.QPoint(0., 0)
        #br = qc.QPoint( 100., 100.)

        #self.x_annotation_rect = qc.QRect(w * (l+(r-l)/2.),
        #                                  h * (b+(t-b)/2.),
        #                                  w * l,
        #                                  h * b)
        #self.x_annotation_rect = qc.QRect(tl, br)
        self.x_annotation_rect = qc.QRect(tl, size)

    @property
    def wh(self):
        return self.width(), self.height()

    def test_refresh(self, npoints=100, ntimes=100):
        tstart = time.time()
        for i in range(ntimes):
            self.plot(num.random.random(npoints))
            self.repaint()
        tstop = time.time()

    def keyPressEvent(self, key_event):
        ''' react on keyboard keys when they are pressed.'''
        key_text = key_event.text()
        if key_text == 'q':
            self.close()

        elif key_text == 'f':
            self.showMaximized()

        QMainWindow.keyPressEvent(self, key_event)

    def set_pen_color(self, color):
        '''
        :param color: color name as string
        '''
        self.color = qg.QColor(*colors[color])

    def plot(self, xdata=None, ydata=None, ndecimate=0, envelope=False,\
             symbol='--', color='black'):
        ''' plot data

        :param *args:  ydata | xdata, ydata
        '''
        self.symbol = symbol
        self.set_pen_color(color)
        if ydata is None:
            return

        if num.size(ydata) == 0:
            print('no data in array')
            return

        if xdata is None:
            xdata = num.arange(len(ydata))

        self.set_data(xdata, ydata, ndecimate)

    def set_data(self, xdata=None, ydata=None, ndecimate=0):

        if ndecimate:
            self._xvisible = mean_decimation(xdata, ndecimate)
            self._yvisible = mean_decimation(ydata, ndecimate)
        else:
            self._xvisible = xdata
            self._yvisible = ydata

        if False: # envelope
            y = num.copy(self._yvisible)
            if len(y)>0:
                # from pyrocko.trace import hilbert
                #self._yvisible = num.sqrt(y**2 + hilbert(y)**2)
                self._yvisible = num.sqrt(y**2 + signal.hilbert(y)**2)

        self.update_datalims()

    def plotlog(self, xdata=None, ydata=None, ndecimate=0, envelope=False, **style_kwargs):
        self.plot(xdata, num.log(ydata), ndecimate, envelope, **style_kwargs)

    def update_datalims(self):

        if not self.ymin:
            self._ymin = num.min(self._yvisible)
        else:
            self._ymin = self.ymin

        if not self.ymax:
            self._ymax = num.max(self._yvisible)
        else:
            self._ymax = self.ymax

        if self.tfollow:
            self._xmin = num.max((num.max(self._xvisible) - self.tfollow, 0))
            self._xmax = num.max((num.max(self._xvisible), self.tfollow))
        else:
            if not self.xmin:
                self._xmin = num.min(self._xvisible)

            if not self.xmax:
                self._xmax = num.max(self._xvisible)

        w, h = self.wh
        self.xproj.set_in_range(self._xmin, self._xmax)
        self.xproj.set_out_range(w * self.left, w * self.right)

        self.yproj.set_in_range(self._ymin, self._ymax)
        self.yproj.set_out_range(
            h*self.top,
            h*self.bottom,)#)
             #h*self.top)

    def set_xlim(self, xmin, xmax):
        ''' Set x data range. If unset scale to min|max of ydata range '''
        self.xmin = xmin
        self.xmax = xmax

    def set_ylim(self, ymin, ymax):
        ''' Set x data range. If unset scales to min|max of ydata range'''
        self.ymin = ymin
        self.ymax = ymax

    def paintEvent(self, e):
        ''' this is executed e.g. when self.repaint() is called. Draws the
        underlying data and scales the content to fit into the widget.'''

        if len(self._xvisible) == 0:
            return

        self.draw_trace(e)

    def draw_trace(self, e):
        painter = qg.QPainter(self)
        qpoints = make_QPolygonF(self.xproj(self._xvisible),
                                 self.yproj(self._yvisible))

        painter.setRenderHint(qg.QPainter.Antialiasing)

        xmin, xmax = self.xproj.get_out_range()
        ymin, ymax = self.yproj.get_out_range()
        painter.fillRect(self.rect(), qg.QBrush(qc.Qt.white))

        if self.symbol == '--':
            painter.save()
            pen = qg.QPen(self.color, self.pen_width, qc.Qt.SolidLine)
            painter.setPen(pen)
            painter.drawPolyline(qpoints)
            painter.restore()

        elif self.symbol == 'o':
            pen = qg.QPen(self.color, 10, qc.Qt.SolidLine)
            painter.save()
            painter.setPen(pen)
            painter.drawPoints(qpoints)
            painter.restore()

        self.draw_axes(painter)
        #self.draw_labels(painter)
        self.draw_y_ticks(painter)
        self.draw_x_ticks(painter)

    def draw_axes(self, painter):
        ''' draw x and y axis'''
        w, h = self.wh
        points = [qc.QPoint(w*self.left, h*(1.-self.top)),
                  qc.QPoint(w*self.left, h*(1.-self.bottom)),
                  qc.QPoint(w*self.right, h*(1.-self.bottom)),]
        painter.drawPoints(qg.QPolygon(points))

    def draw_x_ticks(self, painter):
        w, h = self.wh
        ymin, ymax, yinc = self.yscaler.make_scale(
            (num.min(self._xvisible), num.max(self._xvisible))
        )
        ticks = num.arange(ymin, ymax, yinc)
        ticks_proj = self.xproj(ticks)
        tick_anchor = self.bottom*h
        lines = [qc.QLineF(xval, tick_anchor* 0.8, xval, tick_anchor) for xval in ticks]
        painter.drawLines(lines)
        for i, xval in enumerate(ticks):
            painter.drawText(qc.QPointF(ticks_proj[i], tick_anchor), str(xval))

    def draw_y_ticks(self, painter):
        w, h = self.wh
        ymin, ymax, yinc = self.yscaler.make_scale(
            (self._ymin, self._ymax)
        )
        ticks = num.arange(ymin, ymax, yinc)
        ticks_proj = self.yproj(ticks)
        lines = [qc.QLineF(w * self.left * 0.8, yval, w*self.left, yval) for yval in ticks_proj]
        painter.drawLines(lines)
        for i, yval in enumerate(ticks):
            painter.drawText(qc.QPointF(0, ticks_proj[i]), str(yval))

    def draw_labels(self, painter):
        self.setup_annotation_boxes()
        painter.drawText(self.x_annotation_rect, qc.Qt.AlignCenter, 'Time')

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

    #def mouseMoveEvent(self, mouse_ev):
    #    point = self.mapFromGlobal(mouse_ev.globalPos())
    #    if self.track_start:
    #        x0, y0 = self.track_start
    #        dy = (point.y() - y0) / float(self.height())
    #        self.istart += dy*self.width()
    #        self.istop -= dy*self.width()

    #    self.repaint()


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

    def paintEvent(self, e):
        if len(self._xvisible) == 0:
            return

        self.xproj.set_in_range(num.min(self._xvisible), num.max(self._xvisible))
        self.xproj.set_out_range(self.width(), 0.)

        self.yproj.set_in_range(0., 10.)
        self.yproj.set_out_range(self.height(), 0.)
        self.draw_log_trace(e)


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
    pad_size = int(math.ceil(float(d.size)/ndecimate)*ndecimate - d.size)
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
