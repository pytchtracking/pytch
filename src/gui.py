import logging
import sys
import numpy as num

from pytch.two_channel_tuner import Worker

from pytch.data import MicrophoneRecorder, getaudiodevices, sampling_rate_options
from pytch.gui_util import AutoScaler, Projection, mean_decimation
from pytch.gui_util import make_QPolygonF, _color_names, _colors # noqa
from pytch.util import Profiler, smooth
from pytch.plot import PlotWidget, GaugeWidget

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
    from PyQt5.QtWidgets import QDialogButtonBox, QTabWidget


logger = logging.getLogger(__name__)
tfollow = 3.
fmax = 2000.


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


class DeviceMenuSetting:
    device_index = 0
    accept = True
    show_traces = True

    def set_menu(self, m):
        if isinstance(m, MenuWidget):
            m.box_show_traces.setChecked(self.show_traces)


class DeviceMenu(QDialog):
    ''' Pop up menu at program start devining basic settings'''

    def __init__(self, set_input_callback=None, *args, **kwargs):
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

        self.edit_sampling_rate = LineEditWithLabel(
            'Sampling rate', default=44100)
        layout.addWidget(self.edit_sampling_rate)

        self.edit_nchannels = LineEditWithLabel(
            'Number of Channels', default=2)
        layout.addWidget(self.edit_nchannels)

        layout.addWidget(QLabel('NFFT'))
        self.nfft_choice = self.get_nfft_box()
        layout.addWidget(self.nfft_choice)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.on_ok_clicked)
        buttons.rejected.connect(self.close)
        layout.addWidget(buttons)

    def get_nfft_box(self):
        ''' Return a QSlider for modifying FFT width'''
        b = QComboBox()
        self.nfft_options = [f*1024 for f in [1, 2, 4, 8, 16]]

        for fft_factor in self.nfft_options:
            b.addItem('%s' % fft_factor)

        b.setCurrentIndex(3)
        return b

    def on_ok_clicked(self):
        fftsize = int(self.nfft_choice.currentText())
        recorder = MicrophoneRecorder(
                        chunksize=512,
                        device_no=self.select_input.currentIndex(),
                        sampling_rate=int(self.edit_sampling_rate.value),
                        fftsize=fftsize,
                        nchannels=int(self.edit_nchannels.value))
        self.set_input_callback(recorder)
        self.hide()

    @classmethod
    def from_device_menu_settings(cls, settings, parent, accept=False):
        '''
        :param setting: instance of :py:class:`DeviceMenuSetting`
        :param parent: parent of instance
        :param ok: accept setting
        '''
        menu = cls(parent=parent)

        if settings.device_index is not None:
            menu.select_input.setCurrentIndex(settings.device_index)

        if accept:
            qc.QTimer().singleShot(10, menu.on_ok_clicked)

        return menu


class MenuWidget(QFrame):
    ''' Contains all widget of left-side panel menu'''
    def __init__(self, settings=None, *args, **kwargs):
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

        #layout.addWidget(QLabel('NFFT'), 3, 0)
        #self.nfft_choice = self.get_nfft_box()
        #layout.addWidget(self.nfft_choice, 3, 1)

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
        settings.set_menu(self)

    def setup_palette(self):
        pal = self.palette()
        pal.setColor(qg.QPalette.Background, qg.QColor(*_colors['aluminium3']))
        self.setPalette(pal)
        self.setAutoFillBackground(True)

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
        self.box_show_traces.stateChanged.connect(
            channel_views.show_trace_widgets)
        channel_views.show_trace_widgets(self.box_show_traces.isChecked())

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

        self.show_trace_widgets(False)

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

        self.trace_widget = PlotWidget()#parent=self)
        #self.trace_widget.set_xtick_increment(1.)
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
                              ignore_nan=True)#, ndecimate=2)
        self.trace_widget.update()
        self.spectrum.update()

    def show_trace_widget(self, show=True):
        self.trace_widget.setVisible(show)


class PitchLevelDifferenceViews(QWidget):
    def __init__(self, channel_views, *args, **kwargs):
        QWidget.__init__(self, *args, **kwargs)
        self.channel_views = channel_views
        layout = QGridLayout()
        self.setLayout(layout)
        self.widgets = []

        for i1, cv1 in enumerate(self.channel_views):
            for i2, cv2 in enumerate(self.channel_views):
                if i1>=i2:
                    continue
                w = GaugeWidget(parent=self)
                w.set_title('%s %s' % (i1, i2))
                self.widgets.append((cv1, cv2, w))
                layout.addWidget(w, i1, i2)

    def draw(self):
        for cv1, cv2, w in self.widgets:
            #w.set_data(
            #    cv1.channel.pitch.latest_frame_data(1) -
            #    cv2.channel.pitch.latest_frame_data(1))
            w.set_data(cv1.channel.pitch.latest_frame_data(1))
        self.repaint()

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
        self.figure.update()


class MainWidget(QWidget):
    ''' top level widget covering the central widget in the MainWindow.'''

    def __init__(self, settings, opengl, *args, **kwargs):
        QWidget.__init__(self, *args, **kwargs)

        self.setMouseTracking(True)
        self.top_layout = QHBoxLayout()
        self.setLayout(self.top_layout)

        self.refresh_timer = qc.QTimer()
        self.refresh_timer.timeout.connect(self.refreshwidgets)
        self.menu = MenuWidget(settings)

        self.input_dialog = DeviceMenu.from_device_menu_settings(
            settings, accept=settings.accept, parent=self)
        
        self.input_dialog.set_input_callback = self.set_input

        self.data_input = None

        self.make_connections()
        
        qc.QTimer().singleShot(0, self.set_input_dialog)

    def make_connections(self):
        menu = self.menu
        #worker = self.worker
        #core.device_no = menu.select_input.currentIndex()
        #menu.select_input.activated.connect(self.data_input.set_device_no)

        #menu.nfft_choice.activated.connect(self.set_fftsize)
        menu.input_button.clicked.connect(self.set_input_dialog)

        #menu.pause_button.clicked.connect(core.data_input.stop)
        #menu.play_button.clicked.connect(core.data_input.start)

        #menu.spectral_smoothing.stateChanged.connect(
        #    worker.set_spectral_smoothing)
        #menu.spectral_smoothing.setChecked(worker.spectral_smoothing)
        #menu.set_algorithms(worker.pitch_algorithms, default='yin')
        #menu.select_algorithm.activated.connect(worker.set_pitch_algorithm)

        #self.set_fftsize(3)

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

        self.worker = Worker(
            self.data_input.channels,
            buffer_length=10*60.)

        channel_views = []
        for ichannel, channel in enumerate(self.data_input.channels):
            channel_views.append(ChannelView(channel, color=_color_names[3*ichannel]))

        self.channel_views_widget = ChannelViews(channel_views)
        self.top_layout.addWidget(self.channel_views_widget)

        tabbed_pitch_widget = QTabWidget()
        tabbed_pitch_widget.setSizePolicy(QSizePolicy.Minimum,
                                          QSizePolicy.Minimum)

        self.top_layout.addWidget(tabbed_pitch_widget)

        self.pitch_view = PitchWidget(channel_views)
        self.menu.connect_pitch_view(self.pitch_view)
        self.menu.connect_channel_views(self.channel_views_widget)

        self.pitch_diff_view = PitchLevelDifferenceViews(channel_views)

        tabbed_pitch_widget.addTab(self.pitch_view, 'Pitches')
        tabbed_pitch_widget.addTab(self.pitch_diff_view, 'Diff')
        self.refresh_timer.start(57)

    def set_input(self, input):

        self.cleanup()

        self.data_input = input
        self.data_input.start_new_stream()

        self.reset()

    def refreshwidgets(self):
        self.data_input.flush()
        self.worker.process()
        self.channel_views_widget.draw()
        self.pitch_view.draw()
        self.pitch_diff_view.draw()

    def set_fftsize(self, size):
        self.cleanup()
        for channel in self.data_input.channels:
            channel.fftsize = size

        self.reset()

    def closeEvent(self, ev):
        '''Called when application is closed.'''
        logger.info('closing')
        self.core.data_input.terminate()
        self.cleanup()
        QWidget.closeEvent(self, ev)


class MainWindow(QMainWindow):
    ''' Top level Window. The entry point of the gui.'''
    def __init__(self, settings, opengl=True, *args, **kwargs):
        QMainWindow.__init__(self, *args, **kwargs)
        self.main_widget = MainWidget(settings, opengl=opengl)
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


def from_command_line(close_after=None, check_opengl=False,
                      disable_opengl=False, settings=None):
    ''' Start the GUI from command line'''
    if check_opengl:
        try:
            from PyQt5.QtWidgets import QOpenGLWidget
        except ImportError as e:
            logger.warning(str(e) + ' - opengl not supported')
        else:
            logger.info('opengl supported')
        finally:
            sys.exit()

    app = QApplication(sys.argv)

    settings = settings or DeviceMenuSetting()
    window = MainWindow(settings=settings)

    if close_after:
        close_timer = qc.QTimer()
        close_timer.timeout.connect(app.quit)
        close_timer.start(close_after * 1000.)
    
    app.exec_()


if __name__ == '__main__':
    from_command_line()
