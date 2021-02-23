import logging
import sys
import numpy as num
import os
from abc import abstractmethod

from pytch.processing import Worker

from .gui_util import add_action_group
from .gui_util import make_QPolygonF, _color_names, _colors # noqa
from .util import consecutive, f2cent, index_gradient_filter, relative_keys
from .plot import GLAxis, Axis, GaugeWidget, MikadoWidget, FixGrid
from .keyboard import KeyBoard
from .menu import DeviceMenu, ProcessingMenu
from .channel_mixer import ChannelMixer
from .config import get_config

from PyQt5 import QtCore as qc
from PyQt5 import QtGui as qg
from PyQt5 import QtWidgets as qw
from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtWidgets import QAction, QPushButton, QDockWidget
from PyQt5.QtWidgets import QMenu, QActionGroup, QFileDialog
from PyQt5.QtChart import QChart, QChartView, QValueAxis, QLogValueAxis, QLineSeries


logger = logging.getLogger('pytch.gui')
tfollow = 3.
fmax = 2000.
colormaps = ['viridis', 'wb', 'bw']


class SignalDispatcherWidget(qw.QWidget):
    def __init__(self, *args, **kwargs):
        qw.QWidget.__init__(self, *args, **kwargs)

        self.setLayout(qw.QHBoxLayout())
        self.setContentsMargins(-10, -10, -10, -10)

    def show_spectrum_widget(self, show):
        self.spectrum_widget.setVisible(show)

    def show_spectrogram_widget(self, show):
        self.spectrogram_widget.setVisible(show)

    @abstractmethod
    def rotate_spectrogram_widget(self, rotate):
        pass

    def show_trace_widget(self, show):
        self.trace_widget.setVisible(show)

    @qc.pyqtSlot()
    def on_confidence_threshold_changed(self, *args):
        pass

    @qc.pyqtSlot()
    def on_keyboard_key_pressed(self, *args):
        pass

    @qc.pyqtSlot()
    def on_spectrum_type_select(self, *args):
        pass

    @qc.pyqtSlot(float)
    def on_standard_frequency_changed(self, *args):
        pass

    @qc.pyqtSlot(float)
    def on_pitch_shift_changed(self, f):
        pass

    @qc.pyqtSlot()
    def on_draw(self):
        self.spectrum_widget.on_draw()


class ChannelView(SignalDispatcherWidget):
    def __init__(self, channel, color='red', *args, **kwargs):
        '''
        Visual representation of a Channel instance.

        This is a per-channel container. It contains the trace-view,
        spectrogram-view and the sepctrum-view of a single channel.

        :param channel: pytch.data.Channel instance
        '''
        SignalDispatcherWidget.__init__(self, *args, **kwargs)
        self.channel = channel
        self.color = color

        self.confidence_threshold = 0.9
        self.freq_keyboard = 0

        self.trace_widget = GLAxis()
        self.trace_widget.grids = []
        self.trace_widget.yticks = False
        self.trace_widget.set_ylim(-1000., 1000.)
        self.trace_widget.left = 0.

        self.spectrogram_widget = SpectrogramWidget(channel=channel)

        self.spectrum_widget = SpectrumWidget(parent=self)
#        self.plot_spectrum = self.spectrum_widget.plotlog

        self.fft_smooth_factor = 4

        layout = self.layout()
        layout.addWidget(self.trace_widget)
        layout.addWidget(self.spectrum_widget)
        layout.addWidget(self.spectrogram_widget)

        self.right_click_menu = QMenu('RC', self)
        self.channel_color_menu = QMenu('Channel Color', self.right_click_menu)

        self.color_choices = add_action_group(
            _colors, self.channel_color_menu, self.on_color_select)

        self.right_click_menu.addMenu(self.channel_color_menu)
        self.fft_smooth_factor_menu = QMenu(
            'FFT smooth factor', self.right_click_menu)
        smooth_action_group = QActionGroup(self.fft_smooth_factor_menu)
        smooth_action_group.setExclusive(True)
        self.smooth_choices = []
        for factor in range(5):
            factor += 1
            fft_smooth_action = QAction(
                str(factor),
                self.fft_smooth_factor_menu)

            fft_smooth_action.triggered.connect(self.on_fft_smooth_select)
            fft_smooth_action.setCheckable(True)
            if factor == self.fft_smooth_factor:
                fft_smooth_action.setChecked(True)
            self.smooth_choices.append(fft_smooth_action)
            smooth_action_group.addAction(fft_smooth_action)
            self.fft_smooth_factor_menu.addAction(fft_smooth_action)

        self.right_click_menu.addMenu(self.fft_smooth_factor_menu)

        self.spectrum_type_menu = QMenu(
            'lin/log', self.right_click_menu)
        plot_action_group = QActionGroup(self.spectrum_type_menu)
        plot_action_group.setExclusive(True)

        self.spectrogram_refresh_timer = qc.QTimer()
        self.spectrogram_refresh_timer.timeout.connect(
            self.spectrogram_widget.update_spectrogram)
        self.spectrogram_refresh_timer.start(300)

    @qc.pyqtSlot(float)
    def on_keyboard_key_pressed(self, f):
        self.freq_keyboard = f

    @qc.pyqtSlot()
    def on_draw(self):
        self.trace_widget.clear()
#        self.spectrum_widget.clear()
        c = self.channel
        d = c.fft.latest_frame_data(self.fft_smooth_factor)
        self.trace_widget.plot(*c.latest_frame(
            tfollow), ndecimate=25, color=self.color, line_width=1)
#        self.plot_spectrum(
#            c.freqs, num.mean(d, axis=0), ndecimate=2,
#            color=self.color, ignore_nan=True)

#        self.spectrum_widget.set_xlim(0, 2000)
        self.spectrum_widget.plot_spectrum(c.freqs, num.mean(d, axis=0))

        confidence = c.pitch_confidence.latest_frame_data(1)
        if confidence > self.confidence_threshold:
            x = c.undo_pitch_proxy(c.get_latest_pitch())
#            self.spectrum_widget.axvline(x)

        if self.freq_keyboard:
            self.spectrum_widget.axvline(
                self.freq_keyboard, color='aluminium4', style='dashed',
                line_width=4)

    @qc.pyqtSlot(int)
    def on_confidence_threshold_changed(self, threshold):
        '''
        self.channel_views_widget.
        '''
        self.confidence_threshold = threshold/10.
        logger.debug('update confidence threshold: %i' %
            self.confidence_threshold)

    @qc.pyqtSlot(float)
    def on_standard_frequency_changed(self, f):
        self.channel.standard_frequency = f

    @qc.pyqtSlot(float)
    def on_pitch_shift_changed(self, shift):
        self.channel.pitch_shift = shift

    def rotate_spectrogram_widget(self, rotate=True):
        layout = self.layout()
        self.spectrogram_refresh_timer.stop()
        self.spectrogram_refresh_timer.timeout.disconnect(self.spectrogram_widget.update_spectrogram)
        visible = self.spectrogram_widget.isVisible()
        self.show_spectrogram_widget(False)
        layout.removeWidget(self.spectrogram_widget)
        del self.spectrogram_widget
        if rotate:
            self.spectrogram_widget = SpectrogramWidgetRotated(channel=self.channel)
        else:
            self.spectrogram_widget = SpectrogramWidget(channel=self.channel)
        layout.addWidget(self.spectrogram_widget)
        self.show_spectrogram_widget(visible)
        self.spectrogram_refresh_timer.timeout.connect(self.spectrogram_widget.update_spectrogram)
        self.spectrogram_refresh_timer.start(300)

    @qc.pyqtSlot(qg.QMouseEvent)
    def mousePressEvent(self, mouse_ev):
        if mouse_ev.button() == qc.Qt.RightButton:
            self.right_click_menu.exec_(qg.QCursor.pos())

    @qc.pyqtSlot(str)
    def on_spectrum_type_select(self, arg):
        '''
        Slot to update the spectrum type
        '''
        self.spectrum_widget.set_spectral_type(arg)
        # TODO: support 'pitch' type

    def on_fft_smooth_select(self):
        for c in self.smooth_choices:
            if c.isChecked():
                self.fft_smooth_factor = int(c.text())
                break

    @qc.pyqtSlot(bool)
    def on_color_select(self, triggered):
        for c in self.color_choices:
            if c.isChecked():
                self.color = c.text()
                break


class ProductView(SignalDispatcherWidget):
    def __init__(self, channels, *args, **kwargs):
        SignalDispatcherWidget.__init__(self, *args, **kwargs)

        self.spectrum_widget = ProductSpectrum(self, channels=channels)
        self.spectrogram_widget = ProductSpectrogram(channels=channels)

        self.channels = channels
        self.trace_widget = GLAxis()
        self.trace_widget.setContentsMargins(-10, -10, -10, -10)
        self.trace_widget.grids = []
        self.trace_widget.xticks = False
        self.trace_widget.yticks = False
        layout = self.layout()
        layout.addWidget(self.trace_widget)
        layout.addWidget(self.spectrum_widget)
        layout.addWidget(self.spectrogram_widget)
        self.confidence_threshold = 1

    def rotate_spectrogram_widget(self, rotate=True):
        layout = self.layout()
        visible = self.spectrogram_widget.isVisible()
        self.show_spectrogram_widget(False)
        layout.removeWidget(self.spectrogram_widget)
        del self.spectrogram_widget
        
        if rotate:
            self.spectrogram_widget = ProductSpectrogramRotated(channels=self.channels)
        else:
            self.spectrogram_widget = ProductSpectrogram(channels=self.channels)
        layout.addWidget(self.spectrogram_widget)
        self.show_spectrogram_widget(visible)

    @qc.pyqtSlot()
    def on_draw(self):
        self.spectrum_widget.on_draw()

    @qc.pyqtSlot(qg.QMouseEvent)
    def mousePressEvent(self, mouse_ev):
        if mouse_ev.button() == qc.Qt.RightButton:
            self.spectrum_widget.setVisible(
                not self.spectrum_widget.isVisible())


class ChannelViews(qw.QWidget):
    '''Creates and contains the channel widgets.'''
    def __init__(self, channels):
        qw.QWidget.__init__(self)
        self.views = []
        for ichannel, channel in enumerate(channels):
            self.views.append(
                ChannelView(channel, color=_color_names[3*ichannel])
            )

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        channels = [cv.channel for cv in self.views]
        self.views.append(ProductView(channels=channels))

        for c_view in self.views:
            self.layout.addWidget(c_view)

        self.show_trace_widgets(True)
        self.show_spectrum_widgets(True)
        self.show_spectrogram_widgets(False)
        self.setSizePolicy(qw.QSizePolicy.Minimum, qw.QSizePolicy.Minimum)

    def show_trace_widgets(self, show):
        for view in self.views:
            view.show_trace_widget(show)

    def show_spectrum_widgets(self, show):
        for view in self.views:
            view.show_spectrum_widget(show)

    def show_spectrogram_widgets(self, show):
        for view in self.views:
            view.show_spectrogram_widget(show)

    def rotate_spectrogram_widgets(self, rotate):
        for view in self.views:
            view.rotate_spectrogram_widget(rotate)

    def show_product_widgets(self, show):
        self.views[-1].setVisible(show)

    def set_in_range(self, val_range):
        for c_view in self.views:
            c_view.trace_widget.set_ylim(-val_range, val_range)

    @qc.pyqtSlot(float)
    def on_standard_frequency_changed(self, f):
        for cv in self.views:
            cv.on_standard_frequency_changed(f)

    @qc.pyqtSlot(float)
    def on_pitch_shift_changed(self, f):
        for view in self.views:
            view.on_pitch_shift_changed(f)

    @qc.pyqtSlot()
    def on_draw(self):
        for view in self.views:
            view.on_draw()

    def sizeHint(self):
        return qc.QSize(400, 200)

    def __iter__(self):
        for view in self.views:
            yield view


class SpectrogramWidget(Axis):
    def __init__(self, channel, *args, **kwargs):
        Axis.__init__(self, *args, **kwargs)
        self.ny, self.nx = 100, 300
        self.channel = channel
        fake = num.ones((self.nx, self.ny))
        self.image = self.colormesh(z=fake)
        self.xticks = False
        self.ytick_formatter = '%i'

        self.right_click_menu = QMenu('RC', self)
        self.color_choices = add_action_group(
            colormaps, self.right_click_menu, self.on_color_select)

    @qc.pyqtSlot()
    def update_spectrogram(self):
        c = self.channel

        try:
            y = c.freqs[: self.nx]
            x = c.xdata[-self.ny:]
            d = c.fft.latest_frame_data(self.ny)
            self.image.set_data(num.flipud(d[:, :self.nx].transpose()))
            self.update_datalims(x, y)
        except ValueError as e:
            logger.debug(e)
            return

        self.update()

    @qc.pyqtSlot()
    def on_color_select(self):
        for c in self.color_choices:
            if c.isChecked():
                self.image.set_colortable(c.text())
                break

    @qc.pyqtSlot(qg.QMouseEvent)
    def mousePressEvent(self, mouse_ev):
        if mouse_ev.button() == qc.Qt.RightButton:
            self.right_click_menu.exec_(qg.QCursor.pos())

    def __del__(self):
        print('Spectrogram deleted')

class SpectrogramWidgetRotated(SpectrogramWidget):
    def __init__(self, channel, *args, **kwargs):
        Axis.__init__(self, *args, **kwargs)
        self.ny, self.nx = 300, 100
        self.channel = channel
        fake = num.ones((self.nx, self.ny))
        self.image = self.colormesh(z=fake)
        self.yticks = False
        self.xtick_formatter = '%i'

        self.right_click_menu = QMenu('RC', self)
        self.color_choices = add_action_group(
            colormaps, self.right_click_menu, self.on_color_select)

    @qc.pyqtSlot()
    def update_spectrogram(self):
        c = self.channel

        try:
            x = c.freqs[: self.ny]
            y = c.xdata[-self.nx:]
            d = c.fft.latest_frame_data(self.nx)
            self.image.set_data(d[:, :self.ny])
            self.update_datalims(x, y)
        except ValueError as e:
            logger.debug(e)
            return

        self.update()

#class SpectrumWidget(GLAxis):
#    def __init__(self, *args, **kwargs):
#        GLAxis.__init__(self, *args, **kwargs)
#        self.set_xlim(0, 2000)
#        self.set_ylim(0, 20)
#        self.left = 0.
#        self.yticks = False
#        self.grids = [FixGrid(delta=100., horizontal=False)]
#        self.xtick_formatter = '%i'
#        # TODO: migrate functionanlity from ChannelView

class SpectrumWidget(QChartView):
    def __init__(self, parent):
        QChartView.__init__(self)

        # Creating QChart
        self.chart = QChart()
        self.chart.setAnimationOptions(QChart.NoAnimation)
        self.chart.legend().hide()

        # Adding Chart to view
        self.setChart(self.chart)

        # Setting X-axis (frequency)
        self.axis_x = QValueAxis()
        self.axis_x.setLabelFormat('%d')
        self.axis_x.setTitleText('Frequency')
        self.axis_x.setMax(880)
        self.chart.addAxis(self.axis_x, qc.Qt.AlignBottom)

        self.y_max = 100000
        self.setup_y_axis('log')

        self.setRenderHint(qg.QPainter.Antialiasing)

        self.series = QLineSeries()
        self.chart.addSeries(self.series)
        self.series.attachAxis(self.axis_x)
        self.series.attachAxis(self.axis_y)

    def setup_y_axis(self, type):
        # Setting Y-axis (gain)
        self.current_type = type
        if type == 'log':
            self.axis_y = QLogValueAxis()
        elif type == 'linear':
            self.axis_y = QValueAxis()
        self.axis_y.setTitleText('Gain')
        self.axis_y.setLabelsVisible(False)
        self.axis_y.setMax(self.y_max)
        self.chart.addAxis(self.axis_y, qc.Qt.AlignLeft)

    def plot_spectrum(self, x_data, y_data):
        plot_points = qg.QPolygonF()
        y_data[y_data <= 0] = 1
        for i, x in enumerate(x_data):
            plot_points << qc.QPointF(x, y_data[i])
        data_y_max = num.amax(y_data)
        if data_y_max > self.y_max:
            self.y_max = data_y_max
            self.axis_y.setMax(data_y_max)
        self.series.replace(plot_points)

    def set_spectral_type(self, type):
        if self.current_type != type:
            self.series.detachAxis(self.axis_y)
            self.chart.removeAxis(self.axis_y)
            self.setup_y_axis(type)
            self.series.attachAxis(self.axis_y)

class CheckBoxSelect(qw.QWidget):
    check_box_toggled = qc.pyqtSignal(int)

    def __init__(self, value, parent):
        qw.QWidget.__init__(self, parent=parent)
        self.value = value
        self.check_box = QPushButton(str(self.value), parent=self)
        self.action = qw.QWidgetAction(self)
        self.action.setDefaultWidget(self.check_box)
        self.check_box.clicked.connect(self.on_state_changed)

    @qc.pyqtSlot()
    def on_state_changed(self):
        self.check_box_toggled.emit(self.value)


def set_tick_choices(menu, default=20):
    group = QActionGroup(menu)
    group.setExclusive(True)
    for tick_increment in [10, 20, 50, 100]:
        action = QAction(str(tick_increment), menu)
        action.setCheckable(True)
        if tick_increment == default:
            action.setChecked(True)
        group.addAction(action)
        menu.addAction(action)


class OverView(qw.QWidget):
    highlighted_pitches = []

    def __init__(self, *args, **kwargs):
        qw.QWidget.__init__(self, *args, **kwargs)

        layout = qw.QGridLayout()
        self.setLayout(layout)
        self.ax = Axis()
        self.ax.ytick_formatter = '%i'
        self.ax.xlabels = False
        self.ax.set_ylim(-1500., 1500)
        self.ax.set_grids(100.)
        layout.addWidget(self.ax)

        self.right_click_menu = QMenu('Tick Settings', self)
        self.right_click_menu.triggered.connect(
            self.ax.on_tick_increment_select)
        set_tick_choices(self.right_click_menu, default=100)
        action = QAction('Minor ticks', self.right_click_menu)
        action.setCheckable(True)
        action.setChecked(True)
        self.right_click_menu.addAction(action)
        self.attach_highlight_pitch_menu()

    def draw_highlighted(self, x):
        for high_pitch, label in self.highlighted_pitches:
            self.ax.axhline(high_pitch, line_width=2)
            self.ax.text(x=x, y=high_pitch, text=label)

    def attach_highlight_pitch_menu(self):
        pmenu = QMenu('Highlight pitches', self)
        pmenu.addSeparator()
        pmenu.addAction(qw.QWidgetAction(pmenu))
        for k in sorted(list(relative_keys.keys())):
            action = qw.QWidgetAction(pmenu)
            check_box_widget = CheckBoxSelect(k, pmenu)
            check_box_widget.check_box_toggled.connect(
                self.on_check_box_widget_toggled)
            pmenu.addAction(check_box_widget.action)

        self.right_click_menu.addMenu(pmenu)

    @qc.pyqtSlot(qg.QMouseEvent)
    def mousePressEvent(self, mouse_ev):
        if mouse_ev.button() == qc.Qt.RightButton:
            self.right_click_menu.exec_(qg.QCursor.pos())
        else:
            qw.QWidget.mousePressEvent(self, mouse_ev)

    @qc.pyqtSlot(int)
    def on_check_box_widget_toggled(self, value):
        label = relative_keys[value]
        if (value, label) in self.highlighted_pitches:
            self.highlighted_pitches.remove((value, label))
            self.highlighted_pitches.remove((-1*value, label))
        else:
            self.highlighted_pitches.append((value, label))
            self.highlighted_pitches.append((-1*value, label))


class PitchWidget(OverView):
    ''' Pitches of each trace as discrete samples.'''

    low_pitch_changed = qc.pyqtSignal(num.ndarray)

    def __init__(self, channel_views, *args, **kwargs):
        OverView.__init__(self, *args, **kwargs)
        self.channel_views = channel_views

        save_as_action = QAction('Save pitches', self.right_click_menu)
        save_as_action.triggered.connect(self.on_save_as)
        self.current_low_pitch = num.zeros(len(channel_views))
        self.current_low_pitch[:] = num.nan
        self.right_click_menu.addAction(save_as_action)
        self.track_start = None
        self.tfollow = 3.
        self.setContentsMargins(-10, -10, -10, -10)

    @qc.pyqtSlot()
    def on_draw(self):
        self.ax.clear()
        for i, cv in enumerate(self.channel_views):
            x, y = cv.channel.pitch.latest_frame(
                self.tfollow, clip_min=True)
            index = num.where(cv.channel.pitch_confidence.latest_frame_data(
                len(x))>=cv.confidence_threshold)[0]

            # TODO: attach filter 2000 to slider
            index_grad = index_gradient_filter(x, y, 2000)
            index = num.intersect1d(index, index_grad)
            indices_grouped = consecutive(index)
            for group in indices_grouped:
                if len(group) == 0:
                    continue
                self.ax.plot(
                    x[group], y[group], color=cv.color, line_width=4)

            xstart = num.min(x)
            self.ax.set_xlim(xstart, xstart+self.tfollow)
        try:
            self.current_low_pitch[i] = y[indices_grouped[-1][-1]]
        except IndexError as e:
            pass

        self.low_pitch_changed.emit(self.current_low_pitch)
        self.draw_highlighted(xstart+self.tfollow)
        self.ax.update()

    @qc.pyqtSlot()
    def on_save_as(self):
        _fn = QFileDialog().getSaveFileName(self, 'Save as text file', '.', '')[0]
        if _fn:
            if not os.path.exists(_fn):
                os.makedirs(_fn)
            for i, cv in enumerate(self.channel_views):
                fn = os.path.join(_fn, 'channel%s.txt' %i)
                x, y = cv.channel.pitch.xdata, cv.channel.pitch.ydata
                index = num.where(cv.channel.pitch_confidence.latest_frame_data(
                    len(x))>=cv.confidence_threshold)
                num.savetxt(fn, num.vstack((x[index], y[index])).T)


class ImageWorker(qc.QObject):

    processingFinished = qc.pyqtSignal()
    on_scaling_changed = qc.pyqtSignal(float)
    start = qc.pyqtSignal(str)

    def __init__(self, channels, nx, ny):
        super(ImageWorker, self).__init__()
        self.channels = channels
        self.scaling = 4.
        self.data = None
        self.x = None
        self.y = None
        self.nx = nx
        self.ny = ny
        self.start.connect(self.run)

    @qc.pyqtSlot(float)
    def on_scaling_changed(self, newscale):
        self.scaling = newscale

    @qc.pyqtSlot(str)
    def run(self, message):
        self.spectrogram_refresh_timer = qc.QTimer()
        self.spectrogram_refresh_timer.timeout.connect(
            self.process)
        self.spectrogram_refresh_timer.start(200)

    @qc.pyqtSlot()
    def process(self):
        z = num.asarray(self.channels[0].fft.latest_frame_data(self.ny), dtype=num.float)
        for c in self.channels:
            z *= num.asarray(c.fft.latest_frame_data(self.ny), dtype=num.float)

        self.x = c.xdata[-self.ny:]
        self.y = c.freqs[: self.nx]
        self.data = num.ma.log(num.flipud(z[:, :self.nx].transpose()))**self.scaling
        self.processingFinished.emit()


class ImageWorkerRotated(ImageWorker):
    def process(self):
        z = num.asarray(self.channels[0].fft.latest_frame_data(self.nx), dtype=num.float)
        for c in self.channels:
            z *= num.asarray(c.fft.latest_frame_data(self.nx), dtype=num.float)

        self.y = c.xdata[-self.nx:]
        self.x = c.freqs[: self.ny]
        self.data = num.ma.log(z[:, :self.ny])**self.scaling
        self.processingFinished.emit()


class ProductSpectrogram(SpectrogramWidget):

    scalingChanged = qc.pyqtSignal(float)

    def __init__(self, channels, *args, **kwargs):
        SpectrogramWidget.__init__(self, None, *args, **kwargs)

        self.channels = channels
        self.setContentsMargins(-10, -10, -10, -10)

        self.init_gain_slider()
        self.init_image_worker(False)

    def init_gain_slider(self):
        slider = qw.QSlider()
        slider.valueChanged.connect(self.on_scaling_changed)
        slider.setOrientation(qc.Qt.Horizontal)
        slider.setMinimum(25)
        slider.setMaximum(55)
        slider.setPageStep(1)
        slider.setSliderPosition(40)
        widget_slider = qw.QWidgetAction(self.right_click_menu)
        widget_slider.setDefaultWidget(slider)
        self.right_click_menu.addSeparator()
        self.right_click_menu.addAction('gain:')
        self.right_click_menu.addAction(widget_slider)
        self.right_click_menu.addSeparator()

    def init_image_worker(self, rotate=False):
        self.thread = qc.QThread()
        if rotate:
            self.image_worker = ImageWorkerRotated(self.channels, self.nx, self.ny)
        else:
            self.image_worker = ImageWorker(self.channels, self.nx, self.ny)
        self.image_worker.moveToThread(self.thread)
        self.image_worker.processingFinished.connect(self.update_spectrogram)
        self.image_worker.start.emit('Start Thread')
        self.scalingChanged.connect(self.image_worker.on_scaling_changed)
        self.thread.start()

    @qc.pyqtSlot()
    def update_spectrogram(self):
        try:
            self.update_datalims(self.image_worker.x, self.image_worker.y)
            self.image.set_data(self.image_worker.data)
        except ValueError as e:
            pass

        self.image.update()
        self.update()

    @qc.pyqtSlot(int)
    def on_scaling_changed(self, value):
        self.scalingChanged.emit(value/10.)

    @qc.pyqtSlot(bool)
    def on_color_select(self, triggered):
        for c in self.color_choices:
            if c.isChecked():
                self.image.set_colortable(c.text())
                break

    @qc.pyqtSlot(qg.QMouseEvent)
    def mousePressEvent(self, mouse_ev):
        if mouse_ev.button() == qc.Qt.RightButton:
            self.right_click_menu.exec_(qg.QCursor.pos())

    def __del__(self):
        self.thread.quit()
        self.thread.wait()


class ProductSpectrogramRotated(ProductSpectrogram):
    def __init__(self, channels, *args, **kwargs):
        SpectrogramWidgetRotated.__init__(self, None, *args, **kwargs)

        self.channels = channels
        self.setContentsMargins(-10, -10, -10, -10)

        self.init_gain_slider()
        self.init_image_worker(True)


class ProductSpectrum(SpectrumWidget): #GLAxis):
    def __init__(self, parent, channels):
        SpectrumWidget.__init__(self, parent)
        self.channels = channels
        self.grids = [FixGrid(delta=100., horizontal=False)]
        self.xtick_formatter = '%i'
        self.ylabels = False
        self.setVisible(True)
        self.setContentsMargins(-10, -10, -10, -10)
        sp_retain = self.sizePolicy()
        sp_retain.setRetainSizeWhenHidden(True)
        self.setSizePolicy(sp_retain)

    @qc.pyqtSlot()
    def on_draw(self):
#        self.clear()
        ydata = num.asarray(
            self.channels[0].fft.latest_frame_data(3), dtype=num.float)

        for c in self.channels[1:]:
            ydata *= num.asarray(c.fft.latest_frame_data(3), dtype=num.float)

#        self.plotlog(self.channels[0].freqs, num.mean(ydata, axis=0), ndecimate=2)


class DifferentialPitchWidget(OverView):
    ''' Diffs as line'''
    def __init__(self, channel_views, *args, **kwargs):
        OverView.__init__(self, *args, **kwargs)
        self.setContentsMargins(-10, -10, -10, -10)
        self.channel_views = channel_views
        self.derivative_filter = 2000    # pitch/seconds

    @qc.pyqtSlot(int)
    def on_derivative_filter_changed(self, max_derivative):
        self.derivative_filter = max_derivative

    @qc.pyqtSlot()
    def on_draw(self):
        self.ax.clear()
        for i1, cv1 in enumerate(self.channel_views):
            x1, y1 = cv1.channel.pitch.latest_frame(tfollow, clip_min=True)
            xstart = num.min(x1)
            index1 = cv1.channel.latest_confident_indices(len(x1), cv1.confidence_threshold)
            index1_grad = index_gradient_filter(x1, y1, self.derivative_filter)
            index1 = num.intersect1d(index1, index1_grad)
            for i2, cv2 in enumerate(self.channel_views):
                if i1 >= i2:
                    continue
                x2, y2 = cv2.channel.pitch.latest_frame(tfollow, clip_min=True)
                index2_grad = index_gradient_filter(x2, y2, self.derivative_filter)
                index2 = cv2.channel.latest_confident_indices(
                    len(x2),
                    cv2.confidence_threshold)

                index2 = num.intersect1d(index2, index2_grad)
                indices = num.intersect1d(index1, index2)
                indices_grouped = consecutive(indices)

                for group in indices_grouped:
                    if len(group) == 0:
                        continue

                    y = y1[group] - y2[group]
                    x = x1[group]
                    self.ax.plot(
                        x, y, style='solid', line_width=4, color=cv1.color,
                        antialiasing=False)
                    self.ax.plot(
                        x, y, style=':', line_width=4, color=cv2.color,
                        antialiasing=False)

        self.ax.set_xlim(xstart, xstart+tfollow)
        self.draw_highlighted(xstart)
        self.ax.update()


class PitchLevelDifferenceViews(qw.QWidget):
    ''' The Gauge widget collection'''
    def __init__(self, channel_views, *args, **kwargs):
        qw.QWidget.__init__(self, *args, **kwargs)
        self.channel_views = channel_views
        layout = qw.QGridLayout()
        self.setLayout(layout)
        self.widgets = []
        self.right_click_menu = QMenu('Tick Settings', self)
        self.right_click_menu.triggered.connect(
            self.on_tick_increment_select)

        set_tick_choices(self.right_click_menu)

        # TODO add slider
        self.naverage = 7
        ylim = (-1500, 1500.)
        for i1, cv1 in enumerate(self.channel_views):
            for i2, cv2 in enumerate(self.channel_views):
                if i1>=i2:
                    continue
                w = GaugeWidget(gl=True)
                w.set_ylim(*ylim)
                w.set_title('Channels: %s | %s' % (i1+1, i2+1))
                self.widgets.append((cv1, cv2, w))
                layout.addWidget(w, i1, i2)

    @qc.pyqtSlot(QAction)
    def on_tick_increment_select(self, action):
        for cv1, cv2, widget in self.widgets:
            widget.xtick_increment = int(action.text())

    @qc.pyqtSlot()
    def on_draw(self):
        for cv1, cv2, w in self.widgets:
            confidence1 = num.where(
                cv1.channel.pitch_confidence.latest_frame_data(
                    self.naverage)>cv1.confidence_threshold)
            confidence2 = num.where(
                cv2.channel.pitch_confidence.latest_frame_data(
                    self.naverage)>cv2.confidence_threshold)
            confidence = num.intersect1d(confidence1, confidence2)
            if len(confidence)>1:
                d1 = cv1.channel.pitch.latest_frame_data(self.naverage)[confidence]
                d2 = cv2.channel.pitch.latest_frame_data(self.naverage)[confidence]
                w.set_data(num.median(d1-d2))
            else:
                w.set_data(None)
            w.update()

    @qc.pyqtSlot(qg.QMouseEvent)
    def mousePressEvent(self, mouse_ev):
        if mouse_ev.button() == qc.Qt.RightButton:
            self.right_click_menu.exec_(qg.QCursor.pos())
        else:
            qw.QWidget.mousePressEvent(self, mouse_ev)


class PitchLevelMikadoViews(qw.QWidget):
    def __init__(self, channel_views, *args, **kwargs):
        qw.QWidget.__init__(self, *args, **kwargs)
        self.channel_views = channel_views
        layout = qw.QGridLayout()
        self.setLayout(layout)
        self.widgets = []

        for i1, cv1 in enumerate(self.channel_views):
            for i2, cv2 in enumerate(self.channel_views):
                if i1>=i2:
                    continue
                w = MikadoWidget()
                w.set_ylim(-1500, 1500)
                w.set_title('Channels: %s %s' % (i1, i2))
                w.tfollow = 60.
                self.widgets.append((cv1, cv2, w))
                layout.addWidget(w, i1, i2)

    @qc.pyqtSlot()
    def on_draw(self):
        for cv1, cv2, w in self.widgets:
            x1, y1 = cv1.channel.pitch.latest_frame(w.tfollow)
            x2, y2 = cv2.channel.pitch.latest_frame(w.tfollow)
            w.fill_between(x1, y1, x2, y2)
            w.update()


class RightTabs(qw.QTabWidget):
    def __init__(self, *args, **kwargs):
        qw.QTabWidget.__init__(self, *args, **kwargs)
        self.setSizePolicy(
            qw.QSizePolicy.MinimumExpanding,
            qw.QSizePolicy.MinimumExpanding)

        self.setAutoFillBackground(True)

        pal = self.palette()
        pal.setColor(qg.QPalette.Background, qg.QColor(*_colors['white']))
        self.setPalette(pal)

    def sizeHint(self):
        return qc.QSize(300, 200)


class MainWidget(qw.QWidget):
    ''' top level widget (central widget in the MainWindow.'''
    signal_widgets_clear = qc.pyqtSignal()
    signal_widgets_draw = qc.pyqtSignal()

    def __init__(self, *args, **kwargs):
        qw.QWidget.__init__(self, *args, **kwargs)
        self.tabbed_pitch_widget = RightTabs(parent=self)

        pal = self.palette()
        self.setAutoFillBackground(True)
        pal.setColor(qg.QPalette.Background, qg.QColor(*_colors['white']))
        self.setPalette(pal)

        self.setMouseTracking(True)
        self.top_layout = qw.QGridLayout()
        self.setLayout(self.top_layout)

        self.refresh_timer = qc.QTimer()
        self.refresh_timer.timeout.connect(self.refresh_widgets)
        self.menu = ProcessingMenu()
        self.input_dialog = DeviceMenu()

        self.input_dialog.set_input_callback = self.set_input
        self.data_input = None

        # self.channel_mixer = ChannelMixer()

        qc.QTimer().singleShot(0, self.set_input_dialog)

    def make_connections(self):
        menu = self.menu
        menu.input_button.clicked.connect(self.set_input_dialog)

        menu.pause_button.clicked.connect(self.data_input.stop)
        menu.pause_button.clicked.connect(self.refresh_timer.stop)

        menu.save_as_button.clicked.connect(self.on_save_as)

        menu.play_button.clicked.connect(self.data_input.start)
        menu.play_button.clicked.connect(self.refresh_timer.start)

        menu.select_algorithm.currentTextChanged.connect(
            self.on_algorithm_select)

    @qc.pyqtSlot()
    def on_save_as(self):
        '''Write traces to wav files'''
        _fn = QFileDialog().getSaveFileName(self, 'Save as', '.', '')[0]
        if _fn:
            for i, tr in enumerate(self.channel_views_widget.views):
                if not os.path.exists(_fn):
                    os.makedirs(_fn)
                fn = os.path.join(_fn, 'channel%s' %i)
                tr.channel.save_as(fn, fmt='wav')

    @qc.pyqtSlot(str)
    def on_algorithm_select(self, arg):
        '''change pitch algorithm'''
        for c in self.data_input.channels:
            c.pitch_algorithm = arg

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

    def reset(self):
        dinput = self.data_input

        self.worker = Worker(dinput.channels)

        self.channel_views_widget = ChannelViews(dinput.channels)
        channel_views = self.channel_views_widget.views[:-1]
        for cv in channel_views:
            self.menu.connect_to_confidence_threshold(cv)
        self.signal_widgets_draw.connect(self.channel_views_widget.on_draw)

        self.top_layout.addWidget(self.channel_views_widget, 1, 0, 1, 1)

        # self.top_layout.addWidget(self.channel_mixer, 1, 0, 1, 1)
        self.keyboard = KeyBoard(self)
        self.keyboard.setVisible(False)
        self.keyboard.connect_channel_views(self.channel_views_widget)
        self.top_layout.addWidget(self.keyboard, 0, 0, 1, -1)

        pitch_view = PitchWidget(channel_views)
        pitch_view.low_pitch_changed.connect(
            self.menu.on_adapt_standard_frequency)

        pitch_view_all_diff = DifferentialPitchWidget(channel_views)
        pitch_diff_view = PitchLevelDifferenceViews(channel_views)
        # self.pitch_diff_view_colorized = PitchLevelMikadoViews(channel_views)

        # remove old tabs from pitch view
        self.tabbed_pitch_widget.clear()

        self.tabbed_pitch_widget.addTab(pitch_view, 'Pitches')
        self.tabbed_pitch_widget.addTab(pitch_view_all_diff, 'Differential')
        self.tabbed_pitch_widget.addTab(pitch_diff_view, 'Current')
        # self.tabbed_pitch_widget.addTab(self.pitch_diff_view_colorized, 'Mikado')

        self.menu.derivative_filter_slider.valueChanged.connect(
            pitch_view_all_diff.on_derivative_filter_changed)
        self.menu.connect_channel_views(self.channel_views_widget)

        self.signal_widgets_draw.connect(pitch_view.on_draw)
        self.signal_widgets_draw.connect(pitch_view_all_diff.on_draw)
        self.signal_widgets_draw.connect(pitch_diff_view.on_draw)

        t_wait_buffer = max(dinput.fftsizes)/dinput.sampling_rate*1500.
        qc.QTimer().singleShot(t_wait_buffer, self.start_refresh_timer)

    def start_refresh_timer(self):
        self.refresh_timer.start(58)

    def set_input(self, input):
        self.cleanup()
        self.data_input = input
        # self.channel_mixer.set_channels(self.data_input.channels)
        self.data_input.start_new_stream()
        self.make_connections()

        self.reset()

    @qc.pyqtSlot()
    def refresh_widgets(self):
        '''This is the main refresh loop.'''
        self.worker.process()
        self.signal_widgets_clear.emit()
        self.signal_widgets_draw.emit()

    def closeEvent(self, ev):
        '''Called when application is closed.'''
        logger.info('closing')
        self.data_input.terminate()
        self.cleanup()
        qw.QWidget.closeEvent(self, ev)

    def toggle_keyboard(self):
        self.keyboard.setVisible(not self.keyboard.isVisible())


class AdjustableMainWindow(qw.QMainWindow):

    def sizeHint(self):
        return qc.QSize(1200, 500)

    @qc.pyqtSlot(qg.QKeyEvent)
    def keyPressEvent(self, key_event):
        ''' react on keyboard keys when they are pressed.'''
        key_text = key_event.text()
        if key_text == 'q':
            self.close()
        elif key_text == 'f':
            self.showMaximized()
        super().keyPressEvent(key_event)


class MainWindow(AdjustableMainWindow):
    ''' Top level Window. The entry point of the gui.'''
    def __init__(self):
        super(MainWindow, self).__init__()
        self.main_widget = MainWidget()
        self.main_widget.setFocusPolicy(qc.Qt.StrongFocus)

        self.setCentralWidget(self.main_widget)

        controls_dock_widget = QDockWidget()
        controls_dock_widget.setWidget(self.main_widget.menu)

        views_dock_widget = QDockWidget()
        views_dock_widget.setWidget(self.main_widget.tabbed_pitch_widget)

        # channel_mixer_dock_widget = QDockWidget()
        # channel_mixer_dock_widget.setWidget(self.main_widget.channel_mixer)

        self.addDockWidget(qc.Qt.LeftDockWidgetArea, controls_dock_widget)
        self.addDockWidget(qc.Qt.RightDockWidgetArea, views_dock_widget)
        # self.addDockWidget(qc.Qt.BottomDockWidgetArea, channel_mixer_dock_widget)

        config = get_config()
        if config.start_maximized:
            self.showMaximized()

        self.show()

    @qc.pyqtSlot(qg.QKeyEvent)
    def keyPressEvent(self, key_event):
        ''' react on keyboard keys when they are pressed.'''
        if key_event.text() == 'k':
            self.main_widget.toggle_keyboard()

        super().keyPressEvent(key_event)


def from_command_line(close_after=None, check_opengl=False,
        disable_opengl=False):
    ''' Start the GUI from command line'''
    if check_opengl:
        try:
            from PyQt5.QtWidgets import QOpenGLWidget  # noqa
            logger.info('opengl supported')
        except ImportError as e:
            logger.warning(str(e) + ' - opengl not supported')
        finally:
            sys.exit()

    app = qw.QApplication(sys.argv)

    win = MainWindow()
    if close_after:
        close_timer = qc.QTimer()
        close_timer.timeout.connect(app.quit)
        close_timer.start(close_after * 1000.)

    app.exec_()


if __name__ == '__main__':
    from_command_line()
