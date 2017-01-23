import sys
import numpy as num
import scipy.interpolate as interpolate
import math
import logging

from pytch.two_channel_tuner import cross_spectrum, Worker

from pytch.data import MicrophoneRecorder, getaudiodevices, sampling_rate_options
from pytch.gui_util import AutoScaler, Projection, mean_decimation
from pytch.gui_util import make_QPolygonF
from pytch.util import Profiler, smooth    # noqa

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
    from PyQt5.QtWidgets import QGridLayout, QSpacerItem, QDialog, QLineEdit

logger = logging.getLogger(__name__)
use_pyqtgraph = False
tfollow = 3.
fmax = 2000.


_color_names = [
    'butter1', 'butter2', 'butter3',
    'chameleon1', 'chameleon2', 'chameleon3',
    'orange1', 'orange2', 'orange3',
    'skyblue1', 'skyblue2', 'skyblue3',
    'plum1', 'plum2', 'plum3',
    'chocolate1', 'chocolate2', 'chocolate3',
    'scarletred1', 'scarletred2', 'scarletred3',
    'aluminium1', 'aluminium2', 'aluminium3',
    'aluminium4', 'aluminium5', 'aluminium6',
    'black', 'grey', 'white',
    'red', 'green', 'blue',
    'transparent',
]

_color_values = [
    (252, 233,  79), (237, 212,   0), (196, 160,   0),
    (138, 226,  52), (115, 210,  22), (78,  154,   6),
    (252, 175,  62), (245, 121,   0), (206,  92,   0),
    (114, 159, 207), (52,  101, 164), (32,   74, 135),
    (173, 127, 168), (117,  80, 123), (92,  53, 102),
    (233, 185, 110), (193, 125,  17), (143,  89,   2),
    (239,  41,  41), (204,   0,   0), (164,   0,   0),
    (238, 238, 236), (211, 215, 207), (186, 189, 182),
    (136, 138, 133), (85,   87,  83), (46,   52,  54),
    (0, 0, 0), (10, 10, 10), (255, 255, 255),
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (0, 0, 0, 0),
]

_colors = dict(
    zip(_color_names, _color_values))

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
        return qc.QSize(100, 500)


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


class LineEditWithLabel(QWidget):
    def __init__(self, label, default=None, *args, **kwargs):
        QWidget.__init__(self, *args, **kwargs)
        layout = QHBoxLayout()
        layout.addWidget(QLabel(label))
        self.setLayout(layout)

        self.edit = QLineEdit()
        layout.addWidget(self.edit)

        if default:
            self.edit.setText(str(default))

    @property
    def value(self):
        return self.edit.text()


class DeviceMenu(QDialog):

    def __init__(self, set_input_callback, *args, **kwargs):
        QDialog.__init__(self, *args, **kwargs)
        self.setModal(True)

        self.set_input_callback = set_input_callback

        layout = QVBoxLayout()
        self.setLayout(layout)

        layout.addWidget(QLabel('Select Input Device'))
        self.select_input = QComboBox()
        layout.addWidget(self.select_input)

        self.select_input.clear()
        devices = getaudiodevices()
        curr = len(devices)-1
        for idevice, device in enumerate(devices):
            self.select_input.addItem('%s: %s' % (idevice, device))
            if 'default' in device:
                curr = idevice

        self.select_input.setCurrentIndex(curr)

        #self.edit_sampling_rate = QLineEdit()
        self.edit_sampling_rate = LineEditWithLabel('Sampling rate', default=44100)
        layout.addWidget(self.edit_sampling_rate)

        #self.edit_nchannels = QLineEdit()
        self.edit_nchannels = LineEditWithLabel('Number of Channels', default=2)
        layout.addWidget(self.edit_nchannels)

        buttons = QWidget()
        buttons_layout = QHBoxLayout()
        buttons.setLayout(buttons_layout)

        button_cancel = QPushButton('Cancel')
        button_cancel.clicked.connect(self.close)

        buttons_layout.addWidget(button_cancel)

        button_ok = QPushButton('OK')
        button_ok.clicked.connect(self.on_ok_clicked)
        buttons_layout.addWidget(button_ok)

        layout.addWidget(buttons)
        button_ok.setFocus()

    def on_ok_clicked(self):
        self.set_input_callback(
            MicrophoneRecorder(
                chunksize=512,
                device_no=self.select_input.currentIndex(),
                sampling_rate=int(self.edit_sampling_rate.value),
                nchannels=int(self.edit_nchannels.value))
        )

        self.hide()


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

        self.input_button = QPushButton('Set Input')
        layout.addWidget(self.input_button, 0, 0)

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

        #self.spectral_smoothing = QCheckBox('Spectral smoothing')
        #self.spectral_smoothing.setCheckable(True)
        #layout.addWidget(self.spectral_smoothing, 5, 0)

        layout.addWidget(QLabel('Select Algorithm'), 6, 0)
        self.select_algorithm = QComboBox()
        layout.addWidget(self.select_algorithm, 6, 1)

        layout.addWidget(QLabel('Show traces'), 7, 0)
        self.box_show_traces = QCheckBox()
        layout.addWidget(self.box_show_traces, 7, 1)

        layout.addItem(QSpacerItem(40, 20), 8, 1, qc.Qt.AlignTop)

        self.setFrameStyle(QFrame.Sunken)
        self.setLineWidth(1)
        self.setFrameShape(QFrame.Box)
        self.setMinimumWidth(300)
        self.setSizePolicy(QSizePolicy.Maximum,
                           QSizePolicy.Maximum)
        self.setup_palette()

    def setup_palette(self):
        pal = self.palette()
        pal.setColor(qg.QPalette.Background, qg.QColor(*_colors['aluminium3']))
        self.setPalette(pal)
        self.setAutoFillBackground(True)

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

    def connect_pitch_view(self, pitch_view):
        self.noise_thresh_slider.valueChanged.connect(
            pitch_view.set_noise_threshold)

    def connect_channel_views(self, channel_views):
        self.box_show_traces.stateChanged.connect(channel_views.show_trace_widgets)

    def sizeHint(self):
        return qc.QSize(200, 200)


class ChannelViews(QWidget):
    '''
    Display all ChannelView objects in a QVBoxLayout
    '''
    def __init__(self, channel_views):
        QWidget.__init__(self)
        self.channel_views = channel_views
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        for c_view in self.channel_views:
            self.layout.addWidget(c_view)

    def show_trace_widgets(self, show):
        for c_view in self.channel_views:
            c_view.show_trace_widget(show)

    def add_channel_view(self, channel_view):
        '''
        :param channel_view: ChannelView widget instance
        '''
        self.layout.addWidget(channel_view)

    def draw(self):
        for c_view in self.channel_views:
            c_view.draw()


class ChannelView(QWidget):
    def __init__(self, channel, color='red', *args, **kwargs):
        '''
        Visual representation of a Channel instance.

        :param channel: pytch.data.Channel instance
        '''
        QWidget.__init__(self, *args, **kwargs)
        self.channel = channel

        self.color = color

        layout = QHBoxLayout()
        self.setLayout(layout)

        self.trace_widget = PlotWidget(parent=self)
        layout.addWidget(self.trace_widget)
        self.trace_widget.set_ylim(-1000., 1000.)

        self.spectrum = PlotWidget()
        self.spectrum.set_xlim(0, 2000)
        self.spectrum.set_ylim(0, 20)
        layout.addWidget(self.spectrum)

    def draw(self):
        c = self.channel
        self.trace_widget.plot(*c.latest_frame(
            tfollow), ndecimate=20, color=self.color)

        self.spectrum.plotlog(c.freqs, num.abs(c.fft), color=self.color,
                              ignore_nan=True)

    def show_trace_widget(self, show=True):
        self.trace_widget.setVisible(show)


class PitchWidget(QWidget):
    def __init__(self, channel_views, *args, **kwargs):
        QWidget.__init__(self, *args, **kwargs)
        self.channel_views = channel_views
        layout = QGridLayout()
        self.setLayout(layout)
        self.figure = PlotWidget()
        self.figure.set_ylim(-1000., 1000)
        layout.addWidget(self.figure)
        self.noise_threshold = -99999

    def set_noise_threshold(self, threshold):
        '''
        self.channel_views_widget.
        '''
        self.threshold = threshold

    def draw(self):
        for cv in self.channel_views:
            x, y = cv.channel.pitch.latest_frame(10)
            index = num.where(y>=self.noise_threshold)
            self.figure.plot(x[index], y[index], style='o', line_width=4, color=cv.color)


class ContentWidget(QWidget):
    ''' Contains all visualizing elements'''
    def __init__(self, *args, **kwargs):
        QWidget.__init__(self, *args, **kwargs)
        self.layout = QGridLayout()
        self.setLayout(self.layout)

        self.setSizePolicy(QSizePolicy.Minimum,
                           QSizePolicy.Minimum)

    def sizeHint(self):
        return qc.QSize(1200, 1200)


class MainWidget(QWidget):
    ''' top level widget covering the central widget in the MainWindow.'''

    dataReady = qc.pyqtSignal()

    def __init__(self, *args, **kwargs):
        QWidget.__init__(self, *args, **kwargs)

        self.setMouseTracking(True)
        self.top_layout = QHBoxLayout()
        self.setLayout(self.top_layout)

        self.refresh_timer = qc.QTimer()
        self.refresh_timer.timeout.connect(self.refreshwidgets)
        self.menu = MenuWidget()

        self.input_dialog = DeviceMenu(
            set_input_callback=self.set_input, parent=self)

        self.data_input = None

        self.make_connections()

    def make_connections(self):
        menu = self.menu
        #worker = self.worker
        #core.device_no = menu.select_input.currentIndex()
        #menu.select_input.activated.connect(self.data_input.set_device_no)

        menu.nfft_choice.activated.connect(self.set_fftsize)
        menu.input_button.clicked.connect(self.set_input_dialog)

        #menu.pause_button.clicked.connect(core.data_input.stop)
        #menu.play_button.clicked.connect(core.data_input.start)

        #menu.spectral_smoothing.stateChanged.connect(
        #    worker.set_spectral_smoothing)
        #menu.spectral_smoothing.setChecked(worker.spectral_smoothing)
        #menu.set_algorithms(worker.pitch_algorithms, default='yin')
        #menu.select_algorithm.activated.connect(worker.set_pitch_algorithm)

        self.set_fftsize(3)
        self.set_input_dialog()

    def cleanup(self):
        ''' clear all widgets. '''
        if self.data_input:
            self.data_input.stop()
            self.data_input.terminate()

        while self.top_layout.count():
            item = self.top_layout.takeAt(0)
            item.widget().deleteLater()

    def set_input_dialog(self):
        ''' Query device list and set the drop down menu'''
        self.refresh_timer.stop()
        self.input_dialog.show()
        self.input_dialog.raise_()
        self.input_dialog.activateWindow()

    def set_input(self, input):

        self.cleanup()

        self.data_input = input
        self.data_input.start_new_stream()

        self.worker = Worker(
            self.data_input.channels,
            buffer_length=10*60.)

        channel_views = []
        for ichannel, channel in enumerate(self.data_input.channels):
            channel_views.append(ChannelView(channel, color=_color_names[3*ichannel]))
            #channel_views.append(ChannelView(channel, color='red'))

        self.channel_views_widget = ChannelViews(channel_views)
        self.top_layout.addWidget(self.channel_views_widget)

        self.pitch_view_widget = PitchWidget(channel_views)
        self.top_layout.addWidget(self.pitch_view_widget)
        self.menu.connect_pitch_view(self.pitch_view_widget)
        self.menu.connect_channel_views(self.channel_views_widget)
        self.refresh_timer.start(50)

    def refreshwidgets(self):
        self.data_input.flush()
        self.worker.process()
        self.channel_views_widget.draw()
        self.pitch_view_widget.draw()
        self.repaint()

    def set_fftsize(self, size):
        #for channel in self.data_input.channels:
        #    channel.fftsize = size
        pass

    def closeEvent(self, ev):
        '''Called when application is closed.'''
        logger.debug('closing')
        self.core.data_input.terminate()
        self.cleanup()
        QWidget.closeEvent(self, ev)


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
        #p = Profiler()
        if ndecimate != 0.:
            #p.start()
            self._xvisible = mean_decimation(xdata, ndecimate)
            self._yvisible = mean_decimation(ydata, ndecimate)
            #p.mark('finisehd mean')
            #self._yvisible = smooth(ydata, window_len=ndecimate*2)[::ndecimate]
            #index = num.arange(0, len(self._xvisible), ndecimate)
            #self._yvisible = self._yvisible[index]
            #self._xvisible = xdata[index]
            #if self._yvisible.size == 0:
            #    return
            #p.mark('stop smooth')
            #print(p)
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

        if self.ymin is None:
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
        painter.setRenderHint(qg.QPainter.Antialiasing)
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
        painter.setRenderHint(qg.QPainter.Antialiasing)
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
