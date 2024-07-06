from PyQt6 import QtWidgets as qw
from PyQt6 import QtCore as qc
from PyQt6 import QtGui as qg

import numpy as num
import logging

from .util import cent2f
from .audio import get_input_devices, AudioProcessor
from .audio import get_sampling_rate_options
from .config import get_config, _colors


logger = logging.getLogger("pytch.menu")


class FloatQLineEdit(qw.QLineEdit):
    accepted_value = qc.pyqtSignal(float)

    def __init__(self, default=None, *args, **kwargs):
        qw.QLineEdit.__init__(self, *args, **kwargs)
        self.setValidator(qg.QDoubleValidator())
        self.setFocusPolicy(qc.Qt.FocusPolicy.ClickFocus | qc.Qt.FocusPolicy.TabFocus)
        self.returnPressed.connect(self.do_check)
        p = self.parent()
        if p:
            self.returnPressed.connect(p.setFocus)
        if default:
            self.setText(str(default))

    def do_check(self):
        text = self.text()
        val = float(text)
        self.accepted_value.emit(val)


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
            self.select_input.addItem("{} {}".format(idevice, device["name"]))

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
            qw.QDialogButtonBox.StandardButton.Ok
            | qw.QDialogButtonBox.StandardButton.Cancel
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
        nmax_channels = device["max_input_channels"]

        sampling_rate_options = get_sampling_rate_options(index)
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
        b.setCurrentIndex(1)
        return b

    @qc.pyqtSlot()
    def on_ok_clicked(self):
        selected_channels = self.channel_selector.get_selected_channels()
        logger.debug("selected channels: %s" % selected_channels)
        fftsize = int(self.nfft_choice.currentText())
        recorder = AudioProcessor(
            blocksize=1024,
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
        layout.addWidget(self.pause_button, 1, 1)

        self.save_as_button = qw.QPushButton("Save as")
        layout.addWidget(self.save_as_button, 1, 0)

        layout.addItem(qw.QSpacerItem(40, 20), 2, 0, qc.Qt.AlignmentFlag.AlignTop)

        ###################
        channel_view = qw.QGroupBox("Channel View (Center)")
        cv_layout = qw.QGridLayout()
        cv_layout.addWidget(qw.QLabel("Levels"), 0, 0)
        self.box_show_levels = qw.QCheckBox()
        self.box_show_levels.setChecked(True)
        self.box_show_levels.setChecked(True)
        cv_layout.addWidget(self.box_show_levels, 0, 1, 1, 1)

        cv_layout.addWidget(qw.QLabel("Spectra"), 1, 0)
        self.box_show_spectra = qw.QCheckBox()
        self.box_show_spectra.setChecked(True)
        cv_layout.addWidget(self.box_show_spectra, 1, 1, 1, 1)

        cv_layout.addWidget(qw.QLabel("Spectrograms"), 2, 0)
        self.box_show_spectrograms = qw.QCheckBox()
        self.box_show_spectrograms.setChecked(True)
        cv_layout.addWidget(self.box_show_spectrograms, 2, 1, 1, 1)

        cv_layout.addWidget(qw.QLabel("Products"), 3, 0)
        self.box_show_products = qw.QCheckBox()
        self.box_show_products.setChecked(True)
        cv_layout.addWidget(self.box_show_products, 3, 1, 1, 1)

        self.freq_min = FloatQLineEdit(parent=self, default=20)
        cv_layout.addWidget(qw.QLabel("Minimum Frequency"), 4, 0)
        cv_layout.addWidget(self.freq_min, 4, 1, 1, 1)
        cv_layout.addWidget(qw.QLabel("Hz"), 4, 2)

        self.freq_max = FloatQLineEdit(parent=self, default=1000)
        cv_layout.addWidget(qw.QLabel("Maximum Frequency"), 5, 0)
        cv_layout.addWidget(self.freq_max, 5, 1, 1, 1)
        cv_layout.addWidget(qw.QLabel("Hz"), 5, 2)

        cv_layout.addWidget(qw.QLabel("Magnitude Scale"), 6, 0)
        select_spectral_type = qw.QComboBox(self)
        select_spectral_type.addItems(["log", "linear"])
        select_spectral_type.currentTextChanged.connect(self.on_spectrum_type_select)
        cv_layout.addWidget(select_spectral_type, 6, 1, 1, 1)

        channel_view.setLayout(cv_layout)
        layout.addWidget(channel_view, 3, 0, 1, 2)
        layout.addItem(qw.QSpacerItem(40, 20), 4, 0, qc.Qt.AlignmentFlag.AlignTop)

        ###################
        pitch_view = qw.QGroupBox("Trajectory View (Right)")
        pv_layout = qw.QGridLayout()

        pv_layout.addWidget(qw.QLabel("Select Algorithm"), 0, 0)
        self.select_algorithm = qw.QComboBox(self)
        algorithms = [
            "YIN",
            "SWIPE",
        ]
        self.select_algorithm.addItems(algorithms)
        self.select_algorithm.setCurrentIndex(algorithms.index("YIN"))
        pv_layout.addWidget(self.select_algorithm, 0, 1, 1, 1)

        pv_layout.addWidget(qw.QLabel("Confidence Threshold"), 1, 0)
        self.noise_thresh_slider = qw.QSlider()
        self.noise_thresh_slider.setRange(0, 15)
        # self.noise_thresh_slider.setTickPosition(qw.QSlider.TicksBelow)
        self.noise_thresh_slider.setOrientation(qc.Qt.Orientation.Horizontal)
        self.noise_thresh_slider.valueChanged.connect(
            lambda x: self.noise_thresh_label.setText(str(x / 10.0))
        )
        pv_layout.addWidget(self.noise_thresh_slider, 1, 1, 1, 1)

        self.noise_thresh_label = qw.QLabel(
            f"{self.noise_thresh_slider.value() / 10.0}"
        )
        pv_layout.addWidget(self.noise_thresh_label, 1, 2)

        pv_layout.addWidget(qw.QLabel("Derivative Filter"), 2, 0)
        self.derivative_filter_slider = qw.QSlider()
        self.derivative_filter_slider.setRange(0, 9999)
        self.derivative_filter_slider.setValue(1000)
        self.derivative_filter_slider.setOrientation(qc.Qt.Orientation.Horizontal)
        derivative_filter_label = qw.QLabel(f"{self.derivative_filter_slider.value()}")
        pv_layout.addWidget(derivative_filter_label, 2, 2)
        self.derivative_filter_slider.valueChanged.connect(
            lambda x: derivative_filter_label.setText(str(x))
        )
        pv_layout.addWidget(self.derivative_filter_slider, 2, 1, 1, 1)

        self.f_standard_mode = qw.QComboBox()
        self.f_standard_mode.addItems(["Fixed Freq.", "Highest", "Lowest"])
        self.f_standard_mode.currentTextChanged.connect(self.on_f_standard_mode_changed)

        pv_layout.addWidget(qw.QLabel("Reference Voice"), 3, 0)
        pv_layout.addWidget(self.f_standard_mode, 3, 1, 1, 1)

        self.freq_box = FloatQLineEdit(parent=self, default=220)
        pv_layout.addWidget(qw.QLabel("Reference Frequency"), 4, 0)
        pv_layout.addWidget(self.freq_box, 4, 1, 1, 1)
        pv_layout.addWidget(qw.QLabel("Hz"), 4, 2)

        self.pitch_min = FloatQLineEdit(parent=self, default=-1500)
        pv_layout.addWidget(qw.QLabel("Minimum Pitch"), 5, 0)
        pv_layout.addWidget(self.pitch_min, 5, 1, 1, 1)
        pv_layout.addWidget(qw.QLabel("Cents"), 5, 2)

        self.pitch_max = FloatQLineEdit(parent=self, default=1500)
        pv_layout.addWidget(qw.QLabel("Maximum Pitch"), 6, 0)
        pv_layout.addWidget(self.pitch_max, 6, 1, 1, 1)
        pv_layout.addWidget(qw.QLabel("Cents"), 6, 2)

        # self.pitch_shift_box = FloatQLineEdit(parent=self, default="0.")
        # pv_layout.addWidget(qw.QLabel("Pitch Shift"), 5, 0)
        # pv_layout.addWidget(self.pitch_shift_box, 5, 1, 1, 1)
        # pv_layout.addWidget(qw.QLabel("Cents"), 5, 2)

        pitch_view.setLayout(pv_layout)
        layout.addWidget(pitch_view, 7, 0, 4, 2)

        # layout.addItem(qw.QSpacerItem(40, 20), 7, 0, qc.Qt.AlignTop)

        self.setLineWidth(1)
        self.get_adaptive_f = num.nanmin
        self.setFrameStyle(qw.QFrame.Shadow.Sunken)
        self.setFrameShape(qw.QFrame.Shape.Box)
        self.setSizePolicy(qw.QSizePolicy.Policy.Minimum, qw.QSizePolicy.Policy.Minimum)
        self.setup_palette()

    def setup_palette(self):
        pal = self.palette()
        pal.setColor(qg.QPalette.ColorRole.Window, qg.QColor(*_colors["aluminium1"]))
        self.setPalette(pal)
        self.setAutoFillBackground(True)

    @qc.pyqtSlot(str)
    def on_f_standard_mode_changed(self, text):
        if text == "Highest":
            self.freq_box.setReadOnly(True)
            self.get_adaptive_f = num.nanmax
        elif text == "Lowest":
            self.freq_box.setReadOnly(True)
            self.get_adaptive_f = num.nanmin
        elif "Channel" in text:
            self.freq_box.setReadOnly(True)
            ichannel = int(text[-2:]) - 1
            self.get_adaptive_f = lambda x: x[ichannel]
        else:
            self.get_adaptive_f = lambda x: x
            self.freq_box.setText("220")
            self.freq_box.do_check()
            self.freq_box.setReadOnly(False)

    @qc.pyqtSlot(num.ndarray)
    def on_adapt_standard_frequency(self, fs):
        if num.isnan(fs):
            f = num.nan
        else:
            f = self.get_adaptive_f(fs)
        if self.freq_box.isReadOnly() and not num.isnan(f):
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
        self.box_show_levels.stateChanged.connect(channel_views.show_level_widgets)

        self.box_show_spectrograms.stateChanged.connect(
            channel_views.show_spectrogram_widgets
        )

        self.box_show_products.stateChanged.connect(channel_views.show_product_widgets)

        self.freq_box.accepted_value.connect(
            channel_views.on_standard_frequency_changed
        )

        self.freq_min.accepted_value.connect(channel_views.on_min_freq_changed)
        self.freq_max.accepted_value.connect(channel_views.on_max_freq_changed)

        self.box_show_spectra.stateChanged.connect(channel_views.show_spectrum_widgets)

        # self.pitch_shift_box.accepted_value.connect(
        #     channel_views.on_pitch_shift_changed
        # )

        for i, cv in enumerate(channel_views.views):
            self.spectrum_type_selected.connect(cv.on_spectrum_type_select)

        for i, cv in enumerate(channel_views.views[:-1]):
            self.f_standard_mode.addItem("Channel %s" % (i + 1))

    def sizeHint(self):
        return qc.QSize(200, 200)
