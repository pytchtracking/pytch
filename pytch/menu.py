from PyQt5 import QtWidgets as qw
from PyQt5 import QtCore as qc
from PyQt5 import QtGui as qg

import numpy as num
import logging
import os

from .gui_util import FloatQLineEdit, LineEditWithLabel, _colors
from .util import cent2f
from .data import get_input_devices, MicrophoneRecorder, is_input_device
from .data import get_sampling_rate_options
from .config import get_config


logger = logging.getLogger("pytch.menu")


class ChannelSelector(qw.QWidget):
    def __init__(self, nchannels, channels_enabled):
        super().__init__()
        self.setLayout(qw.QVBoxLayout())

        self.buttons = []
        for i in range(nchannels):
            button = qw.QPushButton("Channel %i" % (i + 1))
            button.setCheckable(True)
            button.setChecked(i in channels_enabled)
            self.buttons.append(button)
            self.layout().addWidget(button)

    def get_selected_channels(self):
        return [i for i, button in enumerate(self.buttons) if button.isChecked()]


class DeviceMenu(qw.QDialog):
    """Pop up menu at program start devining basic settings"""

    def __init__(self, set_input_callback=None, *args, **kwargs):
        qw.QDialog.__init__(self, *args, **kwargs)
        self.setModal(True)
        self.set_input_callback = set_input_callback

        layout = qw.QGridLayout()
        self.setLayout(layout)

        layout.addWidget(qw.QLabel("Input Device"))
        self.select_input = qw.QComboBox()
        layout.addWidget(self.select_input)

        self.select_input.clear()
        self.devices = get_input_devices()

        default_device = (0, self.devices[0])
        for idevice, device in enumerate(self.devices):
            if is_input_device(device):
                extra = ""
                if not default_device[0]:
                    default_device = (idevice, device)
            else:
                extra = "(No Input)"

            self.select_input.addItem("{} {}{}".format(idevice, device["name"], extra))

        # select sampling rate
        layout.addWidget(qw.QLabel("Sampling Rate"))
        self.edit_sampling_rate = qw.QComboBox()
        layout.addWidget(self.edit_sampling_rate)

        # select chunksize
        layout.addWidget(qw.QLabel("Chunksize in Samples"))
        self.nfft_choice = self.get_nfft_box()
        layout.addWidget(self.nfft_choice)

        self.channel_selector_scroll = qw.QScrollArea()
        layout.addWidget(qw.QLabel("Select Channels"), 0, 2, 1, 1)
        layout.addWidget(self.channel_selector_scroll, 1, 2, 6, 1)

        buttons = qw.QDialogButtonBox(
            qw.QDialogButtonBox.Ok | qw.QDialogButtonBox.Cancel
        )

        self.select_input.currentIndexChanged.connect(self.update_channel_info)
        self.select_input.setCurrentIndex(default_device[0])

        buttons.accepted.connect(self.on_ok_clicked)
        buttons.rejected.connect(self.close)
        layout.addWidget(buttons)
        c = get_config()
        if get_config().accept:
            self.on_ok_clicked()

    @qc.pyqtSlot(int)
    def update_channel_info(self, index):
        device = self.devices[index]
        nmax_channels = device["maxInputChannels"]

        sampling_rate_options = get_sampling_rate_options()
        self.channel_selector = ChannelSelector(
            nchannels=nmax_channels, channels_enabled=[0, 1]
        )

        self.channel_selector_scroll.setWidget(self.channel_selector)

        self.edit_sampling_rate.addItems(
            [str(int(v)) for v in sampling_rate_options[device["index"]]]
        )

    def get_nfft_box(self):
        """Return a qw.QSlider for modifying FFT width"""
        b = qw.QComboBox()
        b.addItems([str(f * 1024) for f in [1, 2, 4, 8, 16]])
        b.setCurrentIndex(3)
        return b

    @qc.pyqtSlot()
    def on_ok_clicked(self):
        selected_channels = self.channel_selector.get_selected_channels()
        logger.debug("selected channels: %s" % selected_channels)
        fftsize = int(self.nfft_choice.currentText())
        recorder = MicrophoneRecorder(
            chunksize=1024,
            device_no=self.select_input.currentIndex(),
            sampling_rate=int(self.edit_sampling_rate.currentText()),
            fftsize=int(fftsize),
            selected_channels=selected_channels,
        )

        self.set_input_callback(recorder)
        self.hide()


class ProcessingMenu(qw.QFrame):
    spectrum_type_selected = qc.pyqtSignal(str)

    """ Contains all widget of left-side panel menu"""

    def __init__(self, *args, **kwargs):
        qw.QFrame.__init__(self, *args, **kwargs)

        layout = qw.QGridLayout()
        self.setLayout(layout)

        self.input_button = qw.QPushButton("Set Input")
        layout.addWidget(self.input_button, 0, 0)

        self.play_button = qw.QPushButton("Play")
        layout.addWidget(self.play_button, 0, 1)

        self.pause_button = qw.QPushButton("Pause")
        layout.addWidget(self.pause_button, 1, 0)

        self.save_as_button = qw.QPushButton("Save as")
        layout.addWidget(self.save_as_button, 1, 1)

        layout.addWidget(qw.QLabel("Confidence Threshold"), 4, 0)
        self.noise_thresh_slider = qw.QSlider()
        self.noise_thresh_slider.setRange(0, 15)
        self.noise_thresh_slider.setTickPosition(qw.QSlider.TicksBelow)
        self.noise_thresh_slider.setOrientation(qc.Qt.Horizontal)
        self.noise_thresh_slider.valueChanged.connect(
            lambda x: self.noise_thresh_label.setText(str(x / 10.0))
        )
        layout.addWidget(self.noise_thresh_slider, 4, 1)

        self.noise_thresh_label = qw.QLabel("")
        layout.addWidget(self.noise_thresh_label, 4, 2)

        layout.addWidget(qw.QLabel("Derivative Filter"), 5, 0)
        self.derivative_filter_slider = qw.QSlider()
        self.derivative_filter_slider.setRange(0, 10000)
        self.derivative_filter_slider.setValue(1000)
        self.derivative_filter_slider.setOrientation(qc.Qt.Horizontal)
        layout.addWidget(self.derivative_filter_slider, 5, 1)
        derivative_filter_label = qw.QLabel("")
        layout.addWidget(derivative_filter_label, 5, 2)
        self.derivative_filter_slider.valueChanged.connect(
            lambda x: derivative_filter_label.setText(str(x))
        )

        layout.addWidget(qw.QLabel("Select Algorithm"), 7, 0)
        self.select_algorithm = qw.QComboBox(self)
        algorithms = [
            "default",
            "schmitt",
            "fcomb",
            "mcomb",
            "specacf",
            "yin",
            "yinfft",
            "yinfast",
        ]
        self.select_algorithm.addItems(algorithms)
        self.select_algorithm.setCurrentIndex(algorithms.index("yinfast"))

        layout.addWidget(self.select_algorithm, 7, 1)

        layout.addWidget(qw.QLabel("Traces"), 8, 0)
        self.box_show_traces = qw.QCheckBox()
        self.box_show_traces.setChecked(get_config().show_traces)
        layout.addWidget(self.box_show_traces, 8, 1)

        layout.addWidget(qw.QLabel("Spectra"), 9, 0)
        self.box_show_spectra = qw.QCheckBox()
        self.box_show_spectra.setChecked(True)
        layout.addWidget(self.box_show_spectra, 9, 1)

        layout.addWidget(qw.QLabel("Spectrogram"), 10, 0)
        self.box_show_spectrograms = qw.QCheckBox()
        layout.addWidget(self.box_show_spectrograms, 10, 1)
        layout.addWidget(qw.QLabel("rotated"), 10, 2)
        self.box_rotate_spectrograms = qw.QCheckBox()
        layout.addWidget(self.box_rotate_spectrograms, 10, 3)

        layout.addWidget(qw.QLabel("Products"), 11, 0)
        self.box_show_products = qw.QCheckBox()
        self.box_show_products.setChecked(True)
        layout.addWidget(self.box_show_products, 11, 1)

        self.f_standard_mode = qw.QComboBox()
        self.f_standard_mode.addItems(["Select", "Adaptive (High)", "Adaptive (Low)"])
        self.f_standard_mode.currentTextChanged.connect(self.on_f_standard_mode_changed)

        layout.addWidget(qw.QLabel("Reference Frequency Mode"), 12, 0)
        layout.addWidget(self.f_standard_mode, 12, 1)

        self.freq_box = FloatQLineEdit(parent=self, default=220)
        layout.addWidget(qw.QLabel("Reference Frequency [Hz]"), 13, 0)
        layout.addWidget(self.freq_box, 13, 1)

        self.pitch_shift_box = FloatQLineEdit(parent=self, default="0.")
        layout.addWidget(qw.QLabel("Pitch Shift [Cent]"), 14, 0)
        layout.addWidget(self.pitch_shift_box, 14, 1)

        layout.addWidget(qw.QLabel("Spectral type"), 15, 0)
        select_spectral_type = qw.QComboBox(self)
        select_spectral_type.addItems(["log", "linear", "pitch"])
        select_spectral_type.currentTextChanged.connect(self.on_spectrum_type_select)
        layout.addWidget(select_spectral_type, 15, 1)

        layout.addItem(qw.QSpacerItem(40, 20), 16, 1, qc.Qt.AlignTop)

        self.setLineWidth(1)
        self.get_adaptive_f = num.nanmin
        self.setFrameStyle(qw.QFrame.Sunken)
        self.setFrameShape(qw.QFrame.Box)
        self.setSizePolicy(qw.QSizePolicy.Minimum, qw.QSizePolicy.Minimum)
        self.setup_palette()

    def setup_palette(self):
        pal = self.palette()
        pal.setColor(qg.QPalette.Background, qg.QColor(*_colors["aluminium1"]))
        self.setPalette(pal)
        self.setAutoFillBackground(True)

    @qc.pyqtSlot(str)
    def on_f_standard_mode_changed(self, text):
        if text == "Adaptive (High)":
            self.freq_box.setReadOnly(True)
            self.get_adaptive_f = num.nanmin
        elif text == "Adaptive (Low)":
            self.freq_box.setReadOnly(True)
            self.get_adaptive_f = num.nanmax
        elif "Adaptive (Channel" in text:
            self.freq_box.setReadOnly(True)
            ichannel = int(text[-2]) - 1
            self.get_adaptive_f = lambda x: x[ichannel]
        else:
            self.freq_box.setReadOnly(False)
            self.get_adaptive_f = num.nanmax

    @qc.pyqtSlot(num.ndarray)
    def on_adapt_standard_frequency(self, fs):
        f = self.get_adaptive_f(fs)
        if self.freq_box.isReadOnly() and f != num.nan:
            fref = num.clip(float(self.freq_box.text()), -3000.0, 3000.0)
            txt = str(num.round((cent2f(f, fref) + fref) / 2.0, 2))
            if txt != "nan":
                self.freq_box.setText(txt)
            self.freq_box.do_check()

    @qc.pyqtSlot(str)
    def on_spectrum_type_select(self, arg):
        self.spectrum_type_selected.emit(arg)

    def connect_to_confidence_threshold(self, widget):
        self.noise_thresh_slider.valueChanged.connect(
            widget.on_confidence_threshold_changed
        )
        self.noise_thresh_slider.setValue(int(widget.confidence_threshold * 10))

    def connect_channel_views(self, channel_views):
        self.box_show_traces.stateChanged.connect(channel_views.show_trace_widgets)

        channel_views.show_trace_widgets(self.box_show_traces.isChecked())

        self.box_show_spectrograms.stateChanged.connect(
            channel_views.show_spectrogram_widgets
        )

        self.box_rotate_spectrograms.stateChanged.connect(
            channel_views.rotate_spectrogram_widgets
        )

        self.box_show_products.stateChanged.connect(channel_views.show_product_widgets)

        self.freq_box.accepted_value.connect(
            channel_views.on_standard_frequency_changed
        )

        self.box_show_spectra.stateChanged.connect(channel_views.show_spectrum_widgets)

        self.pitch_shift_box.accepted_value.connect(
            channel_views.on_pitch_shift_changed
        )

        for i, cv in enumerate(channel_views.views):
            self.spectrum_type_selected.connect(cv.on_spectrum_type_select)

        for i, cv in enumerate(channel_views.views[:-1]):
            self.f_standard_mode.addItem("Adaptive (Channel %s)" % (i + 1))

    def sizeHint(self):
        return qc.QSize(200, 200)
