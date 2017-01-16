import sys
import numpy as num
import scipy.interpolate as interpolate
import math
import logging
import copy

from pytch.two_channel_tuner import cross_spectrum
from pytch.data import getaudiodevices, sampling_rate_options
from pytch.core import Core
from pytch.gui_util import AutoScaler
from pytch.util import Profiler    # noqa

if False:
    from PyQt4 import QtCore as qc
    from PyQt4 import QtGui as qg
    from PyQt4.QtGui import QApplication, QWidget, QHBoxLayout, QLabel, QMenu
    from PyQt4.QtGui import QMainWindow, QVBoxLayout, QComboBox, QGridLayout
    from PyQt4.QtGui import QAction, QSlider, QPushButton, QDockWidget, QFrame
else:
    from PyQt5 import QtCore as qc
    from PyQt5 import QtGui as qg
    from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QLabel
    from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QComboBox
    from PyQt5.QtWidgets import QAction, QSlider, QPushButton, QDockWidget
    from PyQt5.QtWidgets import QCheckBox, QSizePolicy, QFrame, QMenu
    from PyQt5.QtWidgets import QGridLayout, QSpacerItem

logger = logging.getLogger(__name__)
use_pyqtgraph = False

if sys.version_info < (3, 0):
    _buffer = buffer
else:
    _buffer = memoryview

_colors = {
    'butter1':     (252, 233,  79),
    'butter2':     (237, 212,   0),
    'butter3':     (196, 160,   0),
    'chameleon1':  (138, 226,  52),
    'chameleon2':  (115, 210,  22),
    'chameleon3':  (78,  154,   6),
    'orange1':     (252, 175,  62),
    'orange2':     (245, 121,   0),
    'orange3':     (206,  92,   0),
    'skyblue1':    (114, 159, 207),
    'skyblue2':    (52,  101, 164),
    'skyblue3':    (32,   74, 135),
    'plum1':       (173, 127, 168),
    'plum2':       (117,  80, 123),
    'plum3':       (92,  53, 102),
    'chocolate1':  (233, 185, 110),
    'chocolate2':  (193, 125,  17),
    'chocolate3':  (143,  89,   2),
    'scarletred1': (239,  41,  41),
    'scarletred2': (204,   0,   0),
    'scarletred3': (164,   0,   0),
    'aluminium1':  (238, 238, 236),
    'aluminium2':  (211, 215, 207),
    'aluminium3':  (186, 189, 182),
    'aluminium4':  (136, 138, 133),
    'aluminium5':  (85,   87,  83),
    'aluminium6':  (46,   52,  54),
    'black': (0, 0, 0),
    'grey': (10, 10, 10),
    'white': (255, 255, 255),
    'red': (255, 0, 0),
    'green': (0, 255, 0),
    'blue': (0, 0, 255),
    'transparent': (0, 0, 0, 0),
}


_pen_styles = {
    'solid': qc.Qt.SolidLine,
    'dashed': qc.Qt.DashLine,
    'dashdot': qc.Qt.DashDotLine,
    'dotted': qc.Qt.DotLine,
    '-': qc.Qt.SolidLine,
    '--': qc.Qt.DashLine,
    '-.': qc.Qt.DashDotLine,
    ':': qc.Qt.DotLine,
    'o': qc.Qt.SolidLine,
}


channel_to_color = ['blue', 'red', 'green', 'black']


class InterpolatedColormap:
    ''' Continuously interpolating colormap '''
    def __init__(self, name=''):
        self.name = name
        self.colors = num.array([
                _colors['red'], _colors['green'], _colors['blue']
            ])

        self.values = num.linspace(0, 255., len(self.colors))
        self.r_interp = interpolate.interp1d(self.values, self.colors.T[0])
        self.g_interp = interpolate.interp1d(self.values, self.colors.T[1])
        self.b_interp = interpolate.interp1d(self.values, self.colors.T[2])
        self.proj = Projection()
        self.proj.set_out_range(0, 255.)

    def update(self):
        pass

    def _map(self, val):
        ''' Interpolate RGB colormap for *val*
        val can be a 1D array.

        Values which are out of range are clipped.
        '''
        val = self.proj.clipped(val)
        return self.r_interp(val), self.g_interp(val), self.b_interp(val)

    def map(self, val):
        return self._map(val)

    def map_to_QColor(self, val):
        return qg.QColor(*self.map(val))

    def set_vlim(self, vmin, vmax):
        self.proj.set_in_range(vmin, vmax)
        self.update()

    def get_incremented_values(self, n=40):
        ''' has to be implemented by every subclass. Needed for plotting.'''
        mi, ma = self.proj.get_in_range()
        return num.linspace(mi, ma, n)

    def get_visualization(self, callback=None):
        '''get dict of values and colors for visualization.

        :param callback: method to retrieve colors from value range.
                        default: *map*
        '''
        vals = self.get_incremented_values()

        if callback:
            colors = [callback(v) for v in vals]
        else:
            colors = [self.map(v) for v in vals]

        return vals, colors

    def __call__(self, val):
        return self._map(val)


class Colormap(InterpolatedColormap):
    ''' Like Colormap but with discrete resolution and precalculated.
    Can return tabulated QColors. Faster than Colormap'''

    def __init__(self, name='', n=20):
        InterpolatedColormap.__init__(self, name=name)
        self.n = n
        self.update()

    def update(self):
        vals = self.get_incremented_values()
        self.colors_QPen = []
        self.colors_QColor = []
        self.colors_rgb = []
        for v in vals:
            rgb = self._map(v)
            self.colors_rgb.append(rgb)
            c = qg.QColor(*rgb)
            self.colors_QColor.append(c)
            self.colors_QPen.append(qg.QPen(c))

    def get_incremented_values(self):
        return num.linspace(*self.proj.xr, num=self.n+1)

    def get_index(self, val):
        return int(self.proj.clipped(val)/self.proj.ur[1] * self.n)

    def map(self, val):
        return self.colors_rgb[self.get_index(val)]

    def map_to_QColor(self, val):
        return self.colors_QColor[self.get_index(val)]

    def map_to_QPen(self, val):
        i = self.get_index(val)
        return self.colors_QPen[i]


class ColormapWidget(QWidget):
    def __init__(self, colormap, *args, **kwargs):
        QWidget.__init__(self, *args, **kwargs)
        self.colormap = colormap
        self.yproj = Projection()
        size_policy = QSizePolicy()
        size_policy.setHorizontalPolicy(QSizePolicy.Maximum)
        self.setSizePolicy(size_policy)

        self.update()

    def update(self):
        _, rgb = self.colormap.get_visualization()
        self.vals, self.colors = self.colormap.get_visualization(
            callback=self.colormap.map_to_QColor)
        self.yproj.set_in_range(num.min(self.vals), num.max(self.vals))

    def paintEvent(self, e):
        rect = self.rect()
        self.yproj.set_out_range(rect.top(), rect.bottom())

        yvals = self.yproj(self.vals)
        painter = qg.QPainter(self)
        for i in range(len(self.vals)-1):
            patch = qc.QRect(qc.QPoint(rect.left(), yvals[i]),
                             qc.QPoint(rect.right(), yvals[i+1]))
            painter.save()
            painter.fillRect(patch, qg.QBrush(self.colors[i]))
            painter.restore()

    def sizeHint(self):
        return qc.QSize(60, 100)


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
        if self._val < self.clip and self.clip:
            color = self.color
        else:
            color = self.clip_color
        pen = qg.QPen(color, 20, qc.Qt.SolidLine)
        painter.setPen(pen)
        painter.drawArc(self.rectf, 2880., self._val)
        # painter.drawPie(self.rectf, 2880., self._val)

    def update_value(self, val):
        '''
        Call this method to update the arc
        '''
        if self.clip:
            # 2880=16*180 (half circle)
            self._val = min(math.log(val)/math.log(self.clip)*2880., 2880)
        else:
            self._val = math.log(val) * 100


class MenuWidget(QFrame):
    ''' Contains all widget of left-side panel menu'''
    def __init__(self, *args, **kwargs):
        QFrame.__init__(self, *args, **kwargs)
        layout = QGridLayout()
        self.setLayout(layout)

        self.play_button = QPushButton('Play')
        layout.addWidget(self.play_button, 0, 0)

        self.pause_button = QPushButton('Pause')
        layout.addWidget(self.pause_button, 0, 1)

        layout.addWidget(QLabel('Select Input Device'), 1, 0)
        self.select_input = QComboBox()
        layout.addWidget(self.select_input, 1, 1)

        # layout.addWidget(QLabel('Sampling rate'), 2, 0)
        # self.select_sampling_rate = QComboBox()
        # layout.addWidget(self.select_sampling_rate, 2, 1)

        layout.addWidget(QLabel('NFFT'), 3, 0)
        self.nfft_choice = self.get_nfft_box()
        layout.addWidget(self.nfft_choice, 3, 1)

        layout.addWidget(QLabel('Noise Threshold'), 4, 0)
        self.noise_thresh_slider = QSlider()
        self.noise_thresh_slider.setRange(0, 10000)
        self.noise_thresh_slider.setValue(1000)
        self.noise_thresh_slider.setOrientation(qc.Qt.Horizontal)
        layout.addWidget(self.noise_thresh_slider, 4, 1)

        self.spectral_smoothing = QCheckBox('Spectral smoothing')
        self.spectral_smoothing.setCheckable(True)
        layout.addWidget(self.spectral_smoothing, 5, 0)

        layout.addWidget(QLabel('Select Algorithm'), 6, 0)
        self.select_algorithm = QComboBox()
        layout.addWidget(self.select_algorithm, 6, 1)

        layout.addItem(QSpacerItem(40, 20), 7, 1, qc.Qt.AlignTop)

        self.setFrameStyle(QFrame.Sunken)
        self.setLineWidth(1)
        self.setFrameShape(QFrame.Box)
        self.setMinimumWidth(300)
        self.setSizePolicy(QSizePolicy.Maximum,
                           QSizePolicy.Maximum)
        self.setup_palette()

    def make_connections(self):
        core = self.core
        worker = core.worker
        core.device_no = self.select_input.currentIndex()
        self.select_input.activated.connect(core.set_device_no)

        self.noise_thresh_slider.valueChanged.connect(worker.set_pmin)
        self.nfft_choice.activated.connect(self.set_worker_nfft)
        self.pause_button.clicked.connect(core.data_input.stop)
        self.play_button.clicked.connect(core.data_input.start)

        self.spectral_smoothing.stateChanged.connect(
            worker.set_spectral_smoothing)
        self.spectral_smoothing.setChecked(worker.spectral_smoothing)
        self.set_algorithms(worker.pitch_algorithms, default='yin')
        self.select_algorithm.activated.connect(worker.set_pitch_algorithm)
        logger.debug('connections made')

        self.set_worker_nfft(3)
        self.set_input_devices()
        # self.set_input_sampling_rates()

    def setup_palette(self):
        pal = self.palette()
        pal.setColor(qg.QPalette.Background, qg.QColor(*_colors['aluminium3']))
        self.setPalette(pal)
        self.setAutoFillBackground(True)

    def set_worker_nfft(self, index):
        self.core.worker.set_fft_length(self.nfft_options[index])

    def get_nfft_box(self):
        ''' Return a QSlider for modifying FFT width'''
        b = QComboBox()
        self.nfft_options = [f*512 for f in [1, 2, 4, 8]]

        for fft_factor in self.nfft_options:
            b.addItem('%s' % fft_factor)

        b.setCurrentIndex(3)
        return b

    def set_algorithms(self, algorithms, default=None):
        ''' Query device list and set the drop down menu'''
        for alg in algorithms:
            self.select_algorithm.addItem('%s' % alg)

        if default:
            self.select_algorithm.setCurrentIndex(algorithms.index(default))

    def set_input_devices(self):
        ''' Query device list and set the drop down menu'''
        devices = getaudiodevices()
        curr = len(devices)-1
        for idevice, device in enumerate(devices):
            self.select_input.addItem('%s: %s' % (idevice, device))
            if 'default' in device:
                curr = idevice

        self.select_input.setCurrentIndex(curr)

    def set_input_sampling_rates(self):
        ''' Set input sampling rates in drop down menu'''
        p = self.core.worker.provider.p
        opts = list(sampling_rate_options(5, p))
        for sr in opts:
            # self.select_sampling_rate.addItem(str(sr))
            self.select_sampling_rate.addItem('ASDFASDF')

        self.select_sampling_rate.setCurrentIndex(2)

    def sizeHint(self):
        return qc.QSize(200, 200)


class ContentWidget(QWidget):
    ''' Contains all visualizing elements'''
    def __init__(self, *args, **kwargs):
        QWidget.__init__(self, *args, **kwargs)
        self.layout = QGridLayout()
        self.setLayout(self.layout)

        self.setSizePolicy(QSizePolicy.Minimum,
                           QSizePolicy.Minimum)

    def sizeHint(self):
        return qc.QSize(800, 700)

tfollow = 1.
fmax = 2000.


class MainWindow(QMainWindow):
    ''' Top level Window. The entry point of the gui.'''
    def __init__(self, *args, **kwargs):
        QMainWindow.__init__(self, *args, **kwargs)
        self.main_widget = MainWidget()
        self.setCentralWidget(self.main_widget)

        controls_dock_widget = QDockWidget()
        controls_dock_widget.setWidget(self.main_widget.menu)
        self.addDockWidget(qc.Qt.LeftDockWidgetArea, controls_dock_widget)

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


class MainWidget(QWidget):
    ''' top level widget covering the central widget in the MainWindow.'''

    processingFinished = qc.pyqtSignal()

    def __init__(self, *args, **kwargs):
        QWidget.__init__(self, *args, **kwargs)

        self.setMouseTracking(True)
        top_layout = QHBoxLayout()
        self.setLayout(top_layout)

        self.core = Core()
        self.core.worker.processingFinished = self.processingFinished

        self.menu = MenuWidget()
        self.menu.core = self.core

        canvas = ContentWidget(parent=self)
        canvas.dx = 1./self.core.data_input.sampling_rate

        self.spectrum1 = PlotWidget(parent=canvas)
        self.spectrum2 = PlotWidget(parent=canvas)
        self.spectrum1.set_ylim(num.log(10.), num.log(100000.))
        self.spectrum2.set_ylim(num.log(10.), num.log(100000.))

        self.spectrum1.set_xlim(0., fmax)
        self.spectrum2.set_xlim(0., fmax)
        canvas.layout.addWidget(self.spectrum1, 1, 0)
        canvas.layout.addWidget(self.spectrum2, 2, 0)

        #if use_pyqtgraph:
        #    import pyqtgraph as pg

        #    trace1_widget = pg.PlotWidget()
        #    self.trace1_qtgraph = trace1_widget.getPlotItem()
        #    canvas.layout.addWidget(trace1_widget, 1, 1)

        #    trace2_widget = pg.PlotWidget()
        #    self.trace2_qtgraph = trace2_widget.getPlotItem()
        #    canvas.layout.addWidget(trace2_widget, 2, 1)
        self.trace1 = PlotWidget(parent=canvas)
        self.trace2 = PlotWidget(parent=canvas)
        self.trace1.tfollow = tfollow
        self.trace2.tfollow = tfollow
        canvas.layout.addWidget(self.trace1, 1, 1)
        canvas.layout.addWidget(self.trace2, 2, 1)
        self.trace1.set_ylim(-3000., 3000.)
        self.trace2.set_ylim(-3000., 3000.)

        self.pitch0 = PlotWidget(parent=canvas)
        self.pitch0.setAttribute(qc.Qt.WA_NoSystemBackground)
        canvas.layout.addWidget(self.pitch0, 3, 1)

        self.pitch1 = PlotWidget(parent=canvas)
        self.pitch1.set_background_color('transparent')
        canvas.layout.addWidget(self.pitch1, 3, 1)

        self.pitch0.show_grid = True
        self.pitch1.show_grid = True
        self.pitch0.set_ylim(-1000., 2500.)
        self.pitch1.set_ylim(-1000., 2500.)

        top_layout.addWidget(canvas)

        self.cross_spectrum = PlotWidget(parent=canvas)
        self.cross_spectrum.set_ylim(0.0001, 25)
        self.cross_spectrum.set_xlim(0., fmax)
        canvas.layout.addWidget(self.cross_spectrum, 3, 0)

        self.pitch_diff = PlotPitchWidget(parent=canvas)
        self.pitch_diff.tfollow = tfollow*200
        self.pitch_diff.set_ylim(-2000, 2000.)

        cmap = self.pitch_diff.colormap
        cmap.set_vlim(0, 2000.)
        pitch_diff_colormap = ColormapWidget(cmap)

        pitch_and_cmap = QWidget()
        pitch_and_cmap_layout = QHBoxLayout()
        pitch_and_cmap.setLayout(pitch_and_cmap_layout)
        pitch_and_cmap_layout.addWidget(self.pitch_diff)
        pitch_and_cmap_layout.addWidget(pitch_diff_colormap)
        canvas.layout.addWidget(pitch_and_cmap, 4, 0, 1, 2)

        self.menu.make_connections()
        self.core.start()

        self.refresh_timer = qc.QTimer()
        self.refresh_timer.timeout.connect(self.refreshwidgets)
        self.refresh_timer.start(100)

    def refreshwidgets(self):
        w = self.core.worker
        self.spectrum1.plotlog(
            w.freqs, num.abs(w.ffts[0]), color=channel_to_color[0])
        self.spectrum2.plotlog(
            w.freqs, num.abs(w.ffts[1]), color=channel_to_color[1])

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

        pitch_0 = w.pitchlogs[0].latest_frame(tfollow*200.)
        pitch_1 = w.pitchlogs[1].latest_frame(tfollow*200.)
        self.pitch0.plot(
            *pitch_0, style='o', line_width=6, color=channel_to_color[0],
            ignore_nan=True)

        self.pitch1.plot(
            *pitch_1, style='o', line_width=6, color=channel_to_color[1],
            ignore_nan=True)

        self.cross_spectrum.plotlog(
            w.freqs, cross_spectrum(w.ffts[0], w.ffts[1])[0])
        self.pitch_diff.fill_between(
            pitch_0[0], pitch_0[1], pitch_1[0], pitch_1[1])

        self.repaint()

    def closeEvent(self, ev):
        '''Called when application is closed.'''
        logger.debug('closing')
        self.core.data_input.terminate()
        QWidget.closeEvent(self, ev)


def normalized_to01(d):
    ''' normalize data vector *d* between 0 and 1'''
    dmin = num.min(d)
    return (d-dmin)/(num.max(d)-dmin)


class PlotWidget(QWidget):
    ''' a plotwidget displays data (x, y coordinates). '''

    def __init__(self, buffer=None, *args, **kwargs):
        QWidget.__init__(self, *args, **kwargs)
        layout = QVBoxLayout()
        self.setLayout(layout)
        self.setContentsMargins(1, 1, 1, 1)

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

        self.left = 0.15
        self.right = 1.
        self.bottom = 0.1
        self.top = 0.1

        self.right_click_menu = QMenu(self)

        self.yticks = None
        self.xticks = None
        self.xzoom = 0.

        self.__show_grid = False
        self.set_background_color('white')
        self.yscaler = AutoScaler(
            no_exp_interval=(-3, 2), approx_ticks=7,
            snap=True
        )

        self.xscaler = AutoScaler(
            no_exp_interval=(-3, 2), approx_ticks=7,
            snap=True
        )
        self.draw_fill = False
        self.draw_points = False
        select_scale = QAction('asdf', self.right_click_menu)
        self.set_pen()
        self.set_brush()
        self.addAction(select_scale)
        self._xvisible = num.empty(0)
        self._yvisible = num.empty(0)
        self.yproj = Projection()
        self.xproj = Projection()
        self.colormap = Colormap()

    def setup_annotation_boxes(self):
        ''' left and top boxes containing labels, dashes, marks, etc.'''
        w, h = self.wh
        l = self.left
        r = self.right
        t = self.bottom
        b = self.top

        tl = qc.QPoint((1.-b) * h, l * w)
        size = qc.QSize(w * (1. - (l + (1.-r))),
                        h * (1. - ((1.-t)+b)))
        self.x_annotation_rect = qc.QRect(tl, size)

    @property
    def wh(self):
        return self.width(), self.height()

    def test_refresh(self, npoints=100, ntimes=100):
        for i in range(ntimes):
            self.plot(num.random.random(npoints))
            self.repaint()

    def keyPressEvent(self, key_event):
        ''' react on keyboard keys when they are pressed.'''
        key_text = key_event.text()
        if key_text == 'q':
            self.close()

        elif key_text == 'f':
            self.showMaximized()

        QWidget.keyPressEvent(self, key_event)

    def set_brush(self, color='black'):
        self.brush = qg.QBrush(qg.QColor(*_colors[color]))

    def set_pen(self, color='black', line_width=1, pen_style='solid'):
        '''
        :param color: color name as string
        '''
        if pen_style == 'o':
            self.draw_points = True

        self.pen = qg.QPen(qg.QColor(*_colors[color]),
                           line_width, _pen_styles[pen_style])

    @property
    def show_grid(self):
        return self.__show_grid

    @show_grid.setter
    def show_grid(self, show):
        self.__show_grid = show

    def plot(self, xdata=None, ydata=None, ndecimate=0, envelope=False,
             style='solid', color='black', line_width=1, ignore_nan=False):
        ''' plot data

        :param *args:  ydata | xdata, ydata
        :param ignore_nan: skip values which are nan
        '''
        self.set_pen(color, line_width, style)
        if ydata is None:
            return

        if num.size(ydata) == 0:
            print('no data in array')
            return

        if xdata is None:
            xdata = num.arange(len(ydata))

        if ignore_nan:
            ydata = num.ma.masked_invalid(ydata)
            xdata = xdata[~ydata.mask]
            ydata = ydata[~ydata.mask]

        self.set_data(xdata, ydata, ndecimate)

    def fill_between(self, xdata, ydata1, ydata2, *args, **kwargs):
        x = num.hstack((xdata, xdata[::-1]))
        y = num.hstack((ydata1, ydata2[::-1]))
        self.draw_fill = True
        self.set_data(x, y)

    def set_data(self, xdata=None, ydata=None, ndecimate=0):

        if ndecimate:
            self._xvisible = mean_decimation(xdata, ndecimate)
            self._yvisible = mean_decimation(ydata, ndecimate)
        else:
            self._xvisible = xdata
            self._yvisible = ydata

        self.colormap.set_vlim(num.min(self._yvisible),
                               num.max(self._yvisible))
        self.update_datalims()

    def plotlog(self, xdata=None, ydata=None, ndecimate=0, envelope=False,
                **style_kwargs):
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
            else:
                self._xmin = self.xmin

            if not self.xmax:
                self._xmax = num.max(self._xvisible)
            else:
                self._xmax = self.xmax

        w, h = self.wh

        mi, ma = self.xproj.get_out_range()
        drange = ma-mi
        xzoom = self.xzoom * drange/2.
        self.xproj.set_in_range(self._xmin - xzoom, self._xmax + xzoom)
        self.xproj.set_out_range(w * self.left, w * self.right)

        self.yproj.set_in_range(self._ymin, self._ymax)
        self.yproj.set_out_range(
            h*(1-self.bottom),
            h*self.top,)

    def set_background_color(self, color):
        '''
        :param color: color as string
        '''
        self.background_color = qg.QColor(*_colors[color])

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

        qpoints = make_QPolygonF(self.xproj(self._xvisible),
                                 self.yproj(self._yvisible))

        painter = qg.QPainter(self)
        painter.save()
        #painter.setRenderHint(qg.QPainter.Antialiasing)
        painter.fillRect(self.rect(), qg.QBrush(self.background_color))
        painter.setPen(self.pen)

        if not self.draw_fill and not self.draw_points:
            painter.drawPolyline(qpoints)

        elif self.draw_fill and not self.draw_points:
            painter.drawPolygon(qpoints)
            qpath = qg.QPainterPath()
            qpath.addPolygon(qpoints)
            painter.fillPath(qpath, self.brush)

        elif self.draw_points:
            painter.drawPoints(qpoints)

        else:
            raise Exception('dont know what to draw')

        painter.restore()
        self.draw_deco(painter)

    def draw_deco(self, painter):
        self.draw_axes(painter)
        # self.draw_labels(painter)
        self.draw_y_ticks(painter)
        self.draw_x_ticks(painter)

        if self.show_grid:
            self.draw_grid_lines(painter)

    def draw_grid_lines(self, painter):
        ''' draw semi transparent grid lines'''
        w, h = self.wh
        ymin, ymax, yinc = self.yscaler.make_scale(
            (self._ymin, self._ymax)
        )
        ticks = num.arange(ymin, ymax, yinc)
        ticks_proj = self.yproj(ticks)

        lines = [qc.QLineF(w * self.left * 0.8, yval, w, yval)
                 for yval in ticks_proj]

        painter.save()
        painter.drawLines(lines)
        painter.restore()

    def draw_axes(self, painter):
        ''' draw x and y axis'''
        w, h = self.wh
        points = [qc.QPoint(w*self.left, h*(1.-self.bottom)),
                  qc.QPoint(w*self.left, h*(1.-self.top)),
                  qc.QPoint(w*self.right, h*(1.-self.top))]
        painter.drawPoints(qg.QPolygon(points))

    def draw_x_ticks(self, painter):
        w, h = self.wh
        xmin, xmax, xinc = self.xscaler.make_scale((self._xmin, self._xmax))
        ticks = num.arange(xmin, xmax, xinc)
        ticks_proj = self.xproj(ticks)
        tick_anchor = self.top*h
        lines = [qc.QLineF(xval, tick_anchor * 0.8, xval, tick_anchor)
                 for xval in ticks_proj]

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
        lines = [qc.QLineF(w * self.left * 0.8, yval, w*self.left, yval)
                 for yval in ticks_proj]
        painter.drawLines(lines)
        for i, yval in enumerate(ticks):
            painter.drawText(qc.QPointF(0, ticks_proj[i]), str(yval))

    def draw_labels(self, painter):
        self.setup_annotation_boxes()
        painter.drawText(self.x_annotation_rect, qc.Qt.AlignCenter, 'Time')

    def mousePressEvent(self, mouse_ev):
        self.test_refresh()
        point = self.mapFromGlobal(mouse_ev.globalPos())
        if mouse_ev.button() == qc.Qt.RightButton:
            self.right_click_menu.exec_(qg.QCursor.pos())
        elif mouse_ev.button() == qc.Qt.LeftButton:
            self.track_start = (point.x(), point.y())
            self.last_y = point.y()
            self._zoom_track = 0.

    def mouseReleaseEvent(self, mouse_event):
        self.track_start = None
        self._zoom_track = 0.

    def mouseMoveEvent(self, mouse_ev):
        point = self.mapFromGlobal(mouse_ev.globalPos())
        x0, y0 = self.track_start
        if self.track_start:
            self.dy = (y0 - point.y()) / float(self.height())
            self._zoom_track = self.dy

            self.last_y = point.y()

        self.xzoom = self._zoom_track


class PlotPitchWidget(PlotWidget):
    def __init__(self, *args, **kwargs):
        PlotWidget.__init__(self, *args, **kwargs)

    def fill_between(self, xdata1, ydata1, xdata2, ydata2, *args, **kwargs):
        '''
        plot only data points which are in both x arrays

        :param xdata1, xdata2: xdata arrays
        :param ydata1, ydata2: ydata arrays
        :param colors: either single color or rgb array
                of length(intersect(xdata1, xdata2))
        '''
        indxdata1 = num.in1d(xdata1, xdata2)
        indxdata2 = num.in1d(xdata2, xdata1)

        # this is usually done by *set_data*:
        self._xvisible = num.vstack((xdata1[indxdata1], xdata2[indxdata2]))
        self._yvisible = num.vstack((ydata1[indxdata1], ydata2[indxdata2]))

        self.update_datalims()

    def paintEvent(self, e):
        ''' this is executed e.g. when self.repaint() is called. Draws the
        underlying data and scales the content to fit into the widget.'''

        if len(self._xvisible) == 0:
            return
        # p = Profiler()
        # p.mark('start')

        lines = []
        pens = []
        dy = num.abs(self._yvisible[0] - self._yvisible[1])

        # SHOULD BE DONE OUTSIDE THIS SCOPE AND FIXED!
        x = self.xproj(self._xvisible)
        y = self.yproj(self._yvisible)
        # p.mark('start setup lines')

        for i in range(len(self._xvisible[0])):
            lines.append(qc.QLineF(x[0][i], y[0][i], x[1][i], y[1][i]))
            pens.append(self.colormap.map_to_QPen(dy[i]))
        
        # p.mark('start finished setup lines')

        painter = qg.QPainter(self)
        #painter.setRenderHint(qg.QPainter.Antialiasing)
        painter.fillRect(self.rect(), qg.QBrush(self.background_color))
        # p.mark('start draw lines')
        for iline, line in enumerate(lines):
            painter.save()
            painter.setPen(pens[iline])
            painter.drawLine(line)
            painter.restore()
        # p.mark('finished draw lines')
        self.draw_deco(painter)
        # print p


class PlotBuffer(PlotWidget):
    ''' plot the content of a Buffer instance.

    This class represents an interface between a Buffer and a PlotWidget.'''
    def __init__(self, *args, **kwargs):
        PlotWidget.__init__(self, *args, **kwargs)


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

    if close_after:
        close_timer = qc.QTimer()
        close_timer.timeout.connect(window.close)
        close_timer.start(close_after)

    sys.exit(app.exec_())


if __name__ == '__main__':
    from_command_line()
