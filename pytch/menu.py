from PyQt6 import QtWidgets as qw
from PyQt6 import QtCore as qc
from PyQt6 import QtGui as qg

import numpy as np
import logging

from .util import cent2f
from .audio import get_input_devices, get_fs_options
from .config import _colors


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

    def __init__(self, gui_callback, *args, **kwargs):
        qw.QDialog.__init__(self, *args, **kwargs)
        self.setModal(True)

        layout = qw.QGridLayout()
        self.setLayout(layout)

        layout.addWidget(qw.QLabel("Input Device"))
        self.input_options = qw.QComboBox()
        layout.addWidget(self.input_options)

        self.input_options.clear()
        self.devices = get_input_devices()

        default_device = (0, self.devices[0])
        for idevice, device in enumerate(self.devices):
            self.input_options.addItem("{} {}".format(idevice, device["name"]))

        # select sampling rate
        layout.addWidget(qw.QLabel("Sampling Rate"))
        self.fs_options = qw.QComboBox()
        layout.addWidget(self.fs_options)

        # select fft size
        layout.addWidget(qw.QLabel("FFT Size in Samples"))
        self.nfft_options = self.get_nfft_box()
        layout.addWidget(self.nfft_options)

        self.channel_options = qw.QScrollArea()
        layout.addWidget(qw.QLabel("Select Channels"), 0, 2, 1, 1)
        layout.addWidget(self.channel_options, 1, 2, 6, 1)

        buttons = qw.QDialogButtonBox(
            qw.QDialogButtonBox.StandardButton.Ok
            | qw.QDialogButtonBox.StandardButton.Cancel
        )

        self.input_options.currentIndexChanged.connect(self.update_channel_info)
        self.input_options.setCurrentIndex(default_device[0])

        buttons.accepted.connect(self.on_ok_clicked)
        buttons.rejected.connect(self.close)
        layout.addWidget(buttons)

        self.gui_callback = gui_callback
        self.selected_channels = None
        self.selected_device = None
        self.selected_fftsize = None
        self.selected_fs = None

    @qc.pyqtSlot(int)
    def update_channel_info(self, index):
        """Updates available channels in input menu"""
        device = self.devices[index]
        nmax_channels = device["max_input_channels"]

        sampling_rate_options = get_fs_options(index)
        self.channel_selector = ChannelSelector(
            nchannels=nmax_channels, channels_enabled=[0, 1]
        )

        self.channel_options.setWidget(self.channel_selector)
        self.fs_options.clear()
        self.fs_options.addItems([str(int(v)) for v in sampling_rate_options])

    def get_nfft_box(self):
        """Return a qw.QSlider for modifying FFT width"""
        b = qw.QComboBox()
        b.addItems([str(f * 256) for f in [1, 2, 4, 8, 16]])
        b.setCurrentIndex(1)
        return b

    @qc.pyqtSlot()
    def on_ok_clicked(self):
        self.selected_channels = self.channel_selector.get_selected_channels()
        self.selected_device = self.input_options.currentIndex()
        self.selected_fftsize = int(self.nfft_options.currentText())
        self.selected_fs = int(self.fs_options.currentText())
        self.hide()
        self.gui_callback()


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
        self.noise_thresh_slider.setRange(0, 10)
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

        self.ref_freq_mode = "Fixed"
        self.ref_freq_mode_menu = qw.QComboBox()
        self.ref_freq_mode_menu.addItems(["Fixed", "Highest", "Lowest"])

        pv_layout.addWidget(qw.QLabel("Reference Frequency"), 3, 0)
        pv_layout.addWidget(self.ref_freq_mode_menu, 3, 1, 1, 1)

        self.freq_box = FloatQLineEdit(parent=self, default=220)
        # pv_layout.addWidget(qw.QLabel("Reference Frequency"), 4, 0)
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
        self.get_reference_frequency = np.nanmin
        self.setFrameStyle(qw.QFrame.Shadow.Sunken)
        self.setFrameShape(qw.QFrame.Shape.Box)
        self.setSizePolicy(qw.QSizePolicy.Policy.Minimum, qw.QSizePolicy.Policy.Minimum)
        self.setup_palette()

    def setup_palette(self):
        pal = self.palette()
        pal.setColor(qg.QPalette.ColorRole.Window, qg.QColor(*_colors["aluminium1"]))
        self.setPalette(pal)
        self.setAutoFillBackground(True)

    def on_reference_frequency_mode_changed(self, text):
        if (text == "Highest") or (text == "Lowest") or ("Channel" in text):
            self.freq_box.setReadOnly(True)
        else:
            self.freq_box.setText("220")
            self.freq_box.do_check()
            self.freq_box.setReadOnly(False)

        self.ref_freq_mode = text

    @qc.pyqtSlot(np.ndarray)
    def update_reference_frequency(self, f):
        if self.freq_box.isReadOnly() and not np.isnan(f):
            fref = np.clip(float(self.freq_box.text()), -3000.0, 3000.0)
            txt = str(np.round((cent2f(f, fref) + fref) / 2.0, 2))
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

    def connect_channel_views(self, channel_views, pitch_view, pitch_view_diff):
        self.box_show_levels.stateChanged.connect(channel_views.show_level_widgets)

        self.box_show_spectrograms.stateChanged.connect(
            channel_views.show_spectrogram_widgets
        )

        self.box_show_products.stateChanged.connect(channel_views.show_product_widgets)

        self.freq_box.accepted_value.connect(pitch_view.on_reference_frequency_changed)
        self.freq_box.accepted_value.connect(
            pitch_view_diff.on_reference_frequency_changed
        )
        self.ref_freq_mode_menu.currentTextChanged.connect(
            self.on_reference_frequency_mode_changed
        )

        self.freq_min.accepted_value.connect(channel_views.on_min_freq_changed)
        self.freq_max.accepted_value.connect(channel_views.on_max_freq_changed)

        self.box_show_spectra.stateChanged.connect(channel_views.show_spectrum_widgets)

        for i, cv in enumerate(channel_views.views):
            self.spectrum_type_selected.connect(cv.on_spectrum_type_select)

        for i, cv in enumerate(channel_views.views[:-1]):
            self.ref_freq_mode_menu.addItem("Channel %s" % (i + 1))

    def sizeHint(self):
        return qc.QSize(200, 200)
