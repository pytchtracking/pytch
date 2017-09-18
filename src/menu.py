from PyQt5 import QtWidgets as qw
from PyQt5 import QtCore as qc
from PyQt5 import QtGui as qg

import numpy as num

from .gui_util import FloatQLineEdit, LineEditWithLabel, _colors
from .util import cent2f
from .data import get_audio_devices, MicrophoneRecorder


class DeviceMenuSetting:
    device_index = 0
    accept = True
    show_traces = True

    def set_menu(self, m):
        if isinstance(m, MenuWidget):
            m.box_show_traces.setChecked(self.show_traces)


class DeviceMenu(qw.QDialog):
    ''' Pop up menu at program start devining basic settings'''

    def __init__(self, set_input_callback=None, *args, **kwargs):
        qw.QDialog.__init__(self, *args, **kwargs)
        self.setModal(True)
        self.set_input_callback = set_input_callback

        layout = qw.QVBoxLayout()
        self.setLayout(layout)

        layout.addWidget(qw.QLabel('Select Input Device'))
        self.select_input = qw.QComboBox()
        layout.addWidget(self.select_input)

        self.select_input.clear()
        devices = get_audio_devices()
        curr = len(devices)-1
        for idevice, device in enumerate(devices):
            self.select_input.addItem('%s: %s' % (idevice, device))
            if 'default' in device:
                curr = idevice

        self.select_input.setCurrentIndex(curr)

        self.edit_sampling_rate = LineEditWithLabel(
            'Sampling rate', default=44100)
        self.edit_sampling_rate.edit.setValidator(qg.QDoubleValidator())
        layout.addWidget(self.edit_sampling_rate)

        self.edit_nchannels = LineEditWithLabel(
            'Number of Channels', default=2)
        self.edit_nchannels.edit.setValidator(qg.QDoubleValidator())
        layout.addWidget(self.edit_nchannels)

        layout.addWidget(qw.QLabel('NFFT'))
        self.nfft_choice = self.get_nfft_box()
        layout.addWidget(self.nfft_choice)

        buttons = qw.QDialogButtonBox(
            qw.QDialogButtonBox.Ok | qw.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.on_ok_clicked)
        buttons.rejected.connect(self.close)
        layout.addWidget(buttons)

    def get_nfft_box(self):
        ''' Return a qw.QSlider for modifying FFT width'''
        b = qw.QComboBox()
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
                        fftsize=int(fftsize),
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


class MenuWidget(qw.QFrame):

    spectrum_type_selected = qc.pyqtSignal(str)

    ''' Contains all widget of left-side panel menu'''
    def __init__(self, settings=None, *args, **kwargs):
        qw.QFrame.__init__(self, *args, **kwargs)
        layout = qw.QGridLayout()
        self.setLayout(layout)

        self.input_button = qw.QPushButton('Set Input')
        layout.addWidget(self.input_button, 0, 0)

        self.play_button = qw.QPushButton('Play')
        layout.addWidget(self.play_button, 0, 1)

        self.pause_button = qw.QPushButton('Pause')
        layout.addWidget(self.pause_button, 1, 0)

        self.save_as_button = qw.QPushButton('Save as')
        layout.addWidget(self.save_as_button, 1, 1)

        layout.addWidget(qw.QLabel('Confidence Threshold'), 4, 0)
        self.noise_thresh_slider = qw.QSlider()
        self.noise_thresh_slider.setRange(0, 15)
        self.noise_thresh_slider.setTickPosition(qw.QSlider.TicksBelow)
        self.noise_thresh_slider.setOrientation(qc.Qt.Horizontal)
        self.noise_thresh_slider.valueChanged.connect(
            lambda x: self.noise_thresh_label.setText(str(x/10.))
        )
        layout.addWidget(self.noise_thresh_slider, 4, 1)

        self.noise_thresh_label = qw.QLabel('')
        layout.addWidget(self.noise_thresh_label, 4, 2)

        layout.addWidget(qw.QLabel('Derivative Filter'), 5, 0)
        self.derivative_filter_slider = qw.QSlider()
        self.derivative_filter_slider.setRange(0., 10000.)
        self.derivative_filter_slider.setValue(1000.)
        self.derivative_filter_slider.setOrientation(qc.Qt.Horizontal)
        layout.addWidget(self.derivative_filter_slider, 5, 1)
        derivative_filter_label = qw.QLabel('')
        layout.addWidget(derivative_filter_label, 5, 2)
        self.derivative_filter_slider.valueChanged.connect(
            lambda x: derivative_filter_label.setText(str(x))
        )

        layout.addWidget(qw.QLabel('Select Algorithm'), 7, 0)
        self.select_algorithm = qw.QComboBox(self)
        layout.addWidget(self.select_algorithm, 7, 1)

        layout.addWidget(qw.QLabel('Traces'), 8, 0)
        self.box_show_traces = qw.QCheckBox()
        self.box_show_traces.setChecked(True)
        layout.addWidget(self.box_show_traces, 8, 1)

        layout.addWidget(qw.QLabel('Spectra'), 9, 0)
        self.box_show_spectra = qw.QCheckBox()
        self.box_show_spectra.setChecked(True)
        layout.addWidget(self.box_show_spectra, 9, 1)

        layout.addWidget(qw.QLabel('Spectrogram'), 10, 0)
        self.box_show_spectrograms = qw.QCheckBox()
        layout.addWidget(self.box_show_spectrograms, 10, 1)

        layout.addWidget(qw.QLabel('Products'), 11, 0)
        self.box_show_products = qw.QCheckBox()
        self.box_show_products.setChecked(True)
        layout.addWidget(self.box_show_products, 11, 1)

        self.f_standard_mode = qw.QComboBox()
        self.f_standard_mode.addItem('Select')
        self.f_standard_mode.addItem('Adaptive')
        self.f_standard_mode.currentTextChanged.connect(
            self.on_f_standard_mode_changed)

        layout.addWidget(qw.QLabel('Standard Frequency Mode'), 12, 0)
        layout.addWidget(self.f_standard_mode, 12, 1)

        self.freq_box = FloatQLineEdit(parent=self, default=220)
        layout.addWidget(qw.QLabel('Standard Frequency [Hz]'), 13, 0)
        layout.addWidget(self.freq_box, 13, 1)

        self.pitch_shift_box = FloatQLineEdit(parent=self, default='0.')
        layout.addWidget(qw.QLabel('Pitch Shift [Cent]'), 14, 0)
        layout.addWidget(self.pitch_shift_box, 14, 1)

        layout.addWidget(qw.QLabel('Spectral type'), 15, 0)
        select_spectral_type = qw.QComboBox(self)
        layout.addWidget(select_spectral_type, 15, 1)
        for stype in ['log', 'linear', 'pitch']:
            select_spectral_type.addItem(stype)
        select_spectral_type.currentTextChanged.connect(
            self.on_spectrum_type_select)

        layout.addItem(qw.QSpacerItem(40, 20), 16, 1, qc.Qt.AlignTop)

        self.setLineWidth(1)
        self.setFrameStyle(qw.QFrame.Sunken)
        self.setFrameShape(qw.QFrame.Box)
        self.setSizePolicy(qw.QSizePolicy.Minimum, qw.QSizePolicy.Minimum)
        self.setup_palette()
        settings.set_menu(self)

    def setup_palette(self):
        pal = self.palette()
        pal.setColor(qg.QPalette.Background, qg.QColor(*_colors['aluminium1']))
        self.setPalette(pal)
        self.setAutoFillBackground(True)

    def set_algorithms(self, algorithms, default=None):
        ''' Query device list and set the drop down menu'''
        self.select_algorithm.clear()
        for alg in algorithms:
            self.select_algorithm.addItem('%s' % alg)

        if default:
            self.select_algorithm.setCurrentIndex(algorithms.index(default))

    @qc.pyqtSlot(str)
    def on_f_standard_mode_changed(self, text):
        if text == 'Adaptive':
            self.freq_box.setReadOnly(True)
        else:
            self.freq_box.setReadOnly(False)

    @qc.pyqtSlot(float)
    def on_adapt_standard_frequency(self, f):
        if self.freq_box.isReadOnly() and f != num.nan:
            fref = num.clip(float(self.freq_box.text()), 100., 3000.)
            self.freq_box.setText(str((cent2f(num.round(f, 2), fref) + fref)/2.))
            self.freq_box.do_check()

    @qc.pyqtSlot(str)
    def on_spectrum_type_select(self, arg):
        self.spectrum_type_selected.emit(arg)

    def connect_to_confidence_threshold(self, widget):
        self.noise_thresh_slider.valueChanged.connect(
            widget.on_confidence_threshold_changed)
        self.noise_thresh_slider.setValue(widget.confidence_threshold*10)

    def connect_channel_views(self, channel_views):
        self.box_show_traces.stateChanged.connect(
            channel_views.show_trace_widgets)

        channel_views.show_trace_widgets(
            self.box_show_traces.isChecked())

        self.box_show_spectrograms.stateChanged.connect(
            channel_views.show_spectrogram_widgets)

        self.box_show_products.stateChanged.connect(
            channel_views.show_product_widgets)

        self.freq_box.accepted_value.connect(
            channel_views.on_standard_frequency_changed)

        self.box_show_spectra.stateChanged.connect(
            channel_views.show_spectrum_widgets)
 
        self.pitch_shift_box.accepted_value.connect(
            channel_views.on_pitch_shift_changed)

        for cv in channel_views.views:
            self.spectrum_type_selected.connect(cv.on_spectrum_type_select)

    def sizeHint(self):
        return qc.QSize(200, 200)


