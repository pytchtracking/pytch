#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""GUI Functions"""

import logging
import sys
import numpy as np
import importlib.metadata

from .utils import (
    consecutive,
    index_gradient_filter,
    f2cent,
    cent2f,
    FloatQLineEdit,
    QHLine,
)
from .config import _color_names, _colors
from .audio import AudioProcessor, get_input_devices, get_fs_options

from PyQt6 import QtCore as qc
from PyQt6 import QtGui as qg
from PyQt6 import QtWidgets as qw
from PyQt6.QtWidgets import QVBoxLayout
from PyQt6.QtWidgets import (
    QDockWidget,
    QFrame,
    QSizePolicy,
    QSplitter,
    QHBoxLayout,
    QPushButton,
    QStatusBar,
)

import matplotlib
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.figure import Figure
import matplotlib.colors
import matplotlib.pyplot as plt

logger = logging.getLogger("pytch.gui")


def start_gui():
    """Starts the GUI"""
    app = qw.QApplication(sys.argv)
    input_dialog = InputMenu()
    if input_dialog.exec() == qw.QDialog.DialogCode.Accepted:
        device, channels, fs, fft_size = input_dialog.get_input_settings()
        main_window = MainWindow(
            device=device, channels=channels, fs=fs, fft_size=fft_size
        )
        main_window.showMaximized()
    sys.exit(app.exec())


class InputMenu(qw.QDialog):
    """Pop up menu at program start that offers user to customise basic settings"""

    def __init__(self, *args, **kwargs):
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
        self.fft_size_options = self.get_nfft_box()
        layout.addWidget(self.fft_size_options)

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

    def on_ok_clicked(self):
        self.accept()  # closes the window

    def get_input_settings(self):
        device = self.input_options.currentIndex()
        channels = self.channel_selector.get_selected_channels()
        fs = int(self.fs_options.currentText())
        fft_size = int(self.fft_size_options.currentText())
        return device, channels, fs, fft_size


class MainWindow(qw.QMainWindow):
    """Main window that includes the main widget for the menu and all visualizations."""

    def __init__(self, device, channels, fs, fft_size):
        super().__init__()

        # default settings for the entire GUI.
        self.version = importlib.metadata.version("pytch")
        self.device = device
        self.channels = channels
        self.fs = fs
        self.fft_size = fft_size
        self.f0_algorithms = ["YIN", "SWIPE"]
        self.buf_len_sec = 30.0
        self.disp_pitch_lims = [
            -1500,
            1500,
        ]  # limits in cents for pitch trajectory view
        self.disp_freq_lims = [20, 1000]  # limits in Hz for spectrum/spectrogram view
        self.gui_refresh_ms = int(1000 / 25)  # equivalent to 25 fps
        self.spec_scale_types = ["log", "linear"]
        self.ref_freq_modes = ["fixed", "highest", "lowest"]
        self.ref_freq = 220
        self.conf_threshold = 0.5
        self.derivative_tol = 1000
        self.disp_t_lvl = 1
        self.disp_t_spec = 1
        self.disp_t_stft = 20
        self.disp_t_f0 = 20
        self.disp_t_conf = 20

        # status variables
        self.is_running = False
        self.menu_visible = True
        self.cur_spec_scale_type = self.spec_scale_types[0]
        self.cur_ref_freq = self.ref_freq
        self.cur_ref_freq_mode = self.ref_freq_modes[0]

        matplotlib.rcParams.update({"font.size": 9})
        pal = self.palette()
        self.setAutoFillBackground(True)
        pal.setColor(qg.QPalette.ColorRole.Window, qg.QColor(*_colors["white"]))
        self.setPalette(pal)

        # initialize and start audio processor
        self.audio_processor = AudioProcessor(
            fs=self.fs,
            buf_len_sec=self.buf_len_sec,
            fft_len=self.fft_size,
            channels=self.channels,
            device_no=self.device,
            f0_algorithm=self.f0_algorithms[0],
        )

        # initialize GUI
        self.setWindowTitle(f"Pytch {self.version}")
        central_widget = qw.QWidget()  # contains all contents
        self.setCentralWidget(central_widget)

        splitter = QSplitter(
            qc.Qt.Orientation.Horizontal, central_widget
        )  # split GUI horizontally

        self.menu = ProcessingMenu(self)  # left-hand menu
        self.channel_views = ChannelViews(self)  # channel views
        self.trajectory_views = TrajectoryViews(self)  # trajectory views

        splitter.addWidget(self.menu)
        splitter.addWidget(self.channel_views)
        splitter.addWidget(self.trajectory_views)

        # define how much space each widget gets
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 2)

        layout = QVBoxLayout(central_widget)  # sets the layout of the central widget
        layout.addLayout(self.menu_toggle_button())  # top bar with menu toggle button
        layout.addWidget(splitter)

        # refresh timer
        self.refresh_timer = qc.QTimer()
        self.refresh_timer.timeout.connect(self.refresh_gui)

        self.play_pause()  # start recording and plotting

    def play_pause(self):
        if self.is_running:
            self.audio_processor.stop_stream()
            self.refresh_timer.stop()
            self.is_running = False
            self.menu.play_pause_button.setText("Play")
        else:
            self.audio_processor.start_stream()
            self.refresh_timer.start(self.gui_refresh_ms)
            self.is_running = True
            self.menu.play_pause_button.setText("Pause")

    def refresh_gui(self):
        lvl, stft, f0, conf = self.audio_processor.read_latest_frames(
            t_lvl=self.disp_t_lvl,
            t_stft=self.disp_t_stft,
            t_f0=self.disp_t_f0,
            t_conf=self.disp_t_conf,
        )
        self.channel_views.on_draw(lvl, stft, f0, conf)
        self.trajectory_views.on_draw(f0, conf)

    def menu_toggle_button(self):
        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(0, 0, 0, 0)
        top_bar.setSpacing(0)
        self.toggle_button = QPushButton("☰ Hide Menu")
        self.toggle_button.setFixedSize(100, 10)
        self.toggle_button.clicked.connect(self.toggle_menu)
        self.toggle_button.setStyleSheet("border :none")
        self.toggle_button.adjustSize()
        self.toggle_button.isFlat()
        top_bar.addWidget(self.toggle_button)
        top_bar.addStretch()
        return top_bar

    def toggle_menu(self):
        if self.menu_visible:
            self.menu.hide()
            self.toggle_button.setText("☰ Show Menu")
        else:
            self.menu.show()
            self.toggle_button.setText("☰ Hide Menu")
        self.menu_visible = not self.menu_visible

    def closeEvent(self, a0):
        self.refresh_timer.stop()
        self.audio_processor.stop_stream()
        self.audio_processor.close_stream()


class ProcessingMenu(QFrame):
    """Contains all widget of left-side panel menu"""

    def __init__(self, main_window: MainWindow, *args, **kwargs):
        qw.QFrame.__init__(self, *args, **kwargs)

        # main menu layout
        main_layout = qw.QGridLayout()
        self.setLayout(main_layout)

        # buttons
        self.play_pause_button = qw.QPushButton("Play")
        main_layout.addWidget(self.play_pause_button, 0, 0, 1, 2)
        self.play_pause_button.clicked.connect(main_window.play_pause)

        # channel view settings
        settings = qw.QGroupBox()
        layout = qw.QGridLayout()
        layout.setAlignment(qc.Qt.AlignmentFlag.AlignTop)

        cv_label = qw.QLabel("Channel View (left)")
        cv_label.setStyleSheet("font-weight: bold")
        layout.addWidget(cv_label, 0, 0)

        layout.addWidget(qw.QLabel("Levels"), 1, 0)
        self.box_show_levels = qw.QCheckBox()
        self.box_show_levels.setChecked(True)
        layout.addWidget(self.box_show_levels, 1, 1, 1, 1)

        layout.addWidget(qw.QLabel("Spectra"), 2, 0)
        self.box_show_spectra = qw.QCheckBox()
        self.box_show_spectra.setChecked(True)
        layout.addWidget(self.box_show_spectra, 2, 1, 1, 1)

        layout.addWidget(qw.QLabel("Spectrograms"), 3, 0)
        self.box_show_spectrograms = qw.QCheckBox()
        self.box_show_spectrograms.setChecked(True)
        layout.addWidget(self.box_show_spectrograms, 3, 1, 1, 1)

        layout.addWidget(qw.QLabel("Products"), 4, 0)
        self.box_show_products = qw.QCheckBox()
        self.box_show_products.setChecked(True)
        layout.addWidget(self.box_show_products, 4, 1, 1, 1)

        self.freq_min = FloatQLineEdit(
            parent=self, default=main_window.disp_freq_lims[0]
        )
        layout.addWidget(qw.QLabel("Minimum Frequency"), 5, 0)
        layout.addWidget(self.freq_min, 5, 1, 1, 1)
        layout.addWidget(qw.QLabel("Hz"), 5, 2)

        self.freq_max = FloatQLineEdit(
            parent=self, default=main_window.disp_freq_lims[1]
        )
        layout.addWidget(qw.QLabel("Maximum Frequency"), 6, 0)
        layout.addWidget(self.freq_max, 6, 1, 1, 1)
        layout.addWidget(qw.QLabel("Hz"), 6, 2)

        layout.addWidget(qw.QLabel("Magnitude Scale"), 7, 0)
        select_spectral_type = qw.QComboBox(self)
        select_spectral_type.addItems(main_window.spec_scale_types)
        select_spectral_type.currentTextChanged.connect(self.on_spectrum_type_select)
        layout.addWidget(select_spectral_type, 7, 1, 1, 1)

        layout.addItem(qw.QSpacerItem(5, 30), 8, 0)

        # Trajectory view settings
        tv_label = qw.QLabel("Trajectory View (right)")
        tv_label.setStyleSheet("font-weight: bold")
        layout.addWidget(tv_label, 9, 0)

        layout.addWidget(qw.QLabel("Show"), 10, 0)
        self.box_show_levels = qw.QCheckBox()
        self.box_show_levels.setChecked(True)
        layout.addWidget(self.box_show_levels, 10, 1, 1, 1)

        layout.addWidget(qw.QLabel("F0 Algorithm"), 11, 0)
        self.select_algorithm = qw.QComboBox(self)
        self.select_algorithm.addItems(main_window.f0_algorithms)
        self.select_algorithm.setCurrentIndex(0)
        layout.addWidget(self.select_algorithm, 11, 1, 1, 1)

        layout.addWidget(qw.QLabel("Confidence Threshold"), 12, 0)
        self.noise_thresh_slider = qw.QSlider()
        self.noise_thresh_slider.setRange(0, 10)
        self.noise_thresh_slider.setValue(int(main_window.conf_threshold * 10))
        self.noise_thresh_slider.setOrientation(qc.Qt.Orientation.Horizontal)
        self.noise_thresh_slider.valueChanged.connect(
            lambda x: self.noise_thresh_label.setText(str(x / 10.0))
        )
        layout.addWidget(self.noise_thresh_slider, 12, 1, 1, 1)
        self.noise_thresh_label = qw.QLabel(
            f"{self.noise_thresh_slider.value() / 10.0}"
        )
        layout.addWidget(self.noise_thresh_label, 12, 2)

        layout.addWidget(qw.QLabel("Derivative Filter"), 13, 0)
        self.derivative_filter_slider = qw.QSlider()
        self.derivative_filter_slider.setRange(0, 9999)
        self.derivative_filter_slider.setValue(main_window.derivative_tol)
        self.derivative_filter_slider.setOrientation(qc.Qt.Orientation.Horizontal)
        derivative_filter_label = qw.QLabel(f"{self.derivative_filter_slider.value()}")
        layout.addWidget(derivative_filter_label, 13, 2)
        self.derivative_filter_slider.valueChanged.connect(
            lambda x: derivative_filter_label.setText(str(x))
        )
        layout.addWidget(self.derivative_filter_slider, 13, 1, 1, 1)

        self.ref_freq_mode_menu = qw.QComboBox()
        self.ref_freq_mode_menu.addItems(main_window.ref_freq_modes)
        self.ref_freq_mode_menu.setCurrentIndex(0)
        layout.addWidget(qw.QLabel("Reference Mode"), 14, 0)
        layout.addWidget(self.ref_freq_mode_menu, 14, 1, 1, 1)

        self.freq_box = FloatQLineEdit(parent=self, default=main_window.ref_freq)
        layout.addWidget(qw.QLabel("Reference Frequency"), 15, 0)
        layout.addWidget(self.freq_box, 15, 1, 1, 1)
        layout.addWidget(qw.QLabel("Hz"), 15, 2)

        self.pitch_min = FloatQLineEdit(
            parent=self, default=main_window.disp_pitch_lims[0]
        )
        layout.addWidget(qw.QLabel("Minimum Pitch"), 16, 0)
        layout.addWidget(self.pitch_min, 16, 1, 1, 1)
        layout.addWidget(qw.QLabel("Cents"), 16, 2)

        self.pitch_max = FloatQLineEdit(
            parent=self, default=main_window.disp_pitch_lims[1]
        )
        layout.addWidget(qw.QLabel("Maximum Pitch"), 17, 0)
        layout.addWidget(self.pitch_max, 17, 1, 1, 1)
        layout.addWidget(qw.QLabel("Cents"), 17, 2)

        settings.setLayout(layout)
        main_layout.addWidget(settings, 3, 0, 1, 2)
        # main_layout.setRowStretch(main_layout.rowCount(), 1)

    def on_reference_frequency_mode_changed(self, text):
        if (text == "Highest") or (text == "Lowest") or ("Channel" in text):
            self.freq_box.setReadOnly(True)
        else:
            self.freq_box.setText("220")
            self.freq_box.do_check()
            self.freq_box.setReadOnly(False)

        self.ref_freq_mode = text

    def update_reference_frequency(self, f):
        if self.freq_box.isReadOnly() and not np.isnan(f):
            fref = np.clip(float(self.freq_box.text()), -3000.0, 3000.0)
            txt = str(np.round((cent2f(f, fref) + fref) / 2.0, 2))
            if txt != "nan":
                self.freq_box.setText(txt)
            self.freq_box.do_check()

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
        return qc.QSize(100, 200)


class ChannelViews(qw.QWidget):
    """Creates and contains the channel widgets."""

    def __init__(self, main_window: MainWindow):
        qw.QWidget.__init__(self)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.views = []
        for ch_id in range(len(main_window.channels)):
            self.views.append(
                ChannelView(
                    main_window=main_window,
                    ch_id=ch_id,
                    is_product=False,
                )
            )

        self.views.append(ChannelView(main_window=main_window, is_product=True))

        for i, c_view in enumerate(self.views):
            if i == len(self.views) - 1:
                self.layout.addWidget(QHLine())
            self.layout.addWidget(c_view)
        self.layout.setContentsMargins(0, 25, 0, 0)

        self.setSizePolicy(qw.QSizePolicy.Policy.Minimum, qw.QSizePolicy.Policy.Minimum)
        self.show_level_widgets(True)
        self.show_spectrum_widgets(True)
        self.show_spectrogram_widgets(True)

    def show_level_widgets(self, show):
        for view in self.views:
            view.show_level_widget(show)

    def show_spectrum_widgets(self, show):
        for view in self.views:
            view.show_spectrum_widget(show)

    def show_spectrogram_widgets(self, show):
        for view in self.views:
            view.show_spectrogram_widget(show)

    def show_product_widgets(self, show):
        self.views[-1].setVisible(show)

    def on_min_freq_changed(self, f):
        for cv in self.views:
            cv.on_min_freq_changed(f)

    def on_max_freq_changed(self, f):
        for cv in self.views:
            cv.on_max_freq_changed(f)

    @qc.pyqtSlot(float)
    def on_pitch_shift_changed(self, f):
        for view in self.views:
            view.on_pitch_shift_changed(f)

    @qc.pyqtSlot()
    def on_draw(self, lvl, stft, f0, conf):
        for view in self.views:
            view.on_draw(lvl, stft, f0, conf)

    def sizeHint(self):
        return qc.QSize(400, 200)

    def __iter__(self):
        yield from self.views


class ChannelView(qw.QWidget):
    """
    Visual representation of a Channel instance.

    This is a per-channel container. It contains the level-view,
    spectrogram-view and the sepctrum-view of a single channel.
    """

    def __init__(
        self, main_window: MainWindow, ch_id=None, is_product=False, *args, **kwargs
    ):
        qw.QWidget.__init__(self, *args, **kwargs)
        self.setLayout(qw.QHBoxLayout())
        self.main_window = main_window

        self.color = "black" if ch_id is None else _color_names[3 * ch_id]
        self.is_product = is_product
        self.ch_id = ch_id

        self.confidence_threshold = 0.8  # default

        channel_label = "Product" if ch_id is None else f"Channel {ch_id + 1}"
        self.level_widget = LevelWidget(channel_label=channel_label)
        self.spectrogram_widget = SpectrogramWidget(self.main_window)
        self.spectrum_widget = SpectrumWidget(self.main_window)

        layout = self.layout()
        layout.addWidget(self.level_widget, 2)
        layout.addWidget(self.spectrum_widget, 7)
        layout.addWidget(self.spectrogram_widget, 7)
        layout.setContentsMargins(0, 0, 0, 0)

        if is_product:
            self.level_widget.ax.xaxis.set_visible(False)
            plt.setp(self.level_widget.ax.spines.values(), visible=False)
            self.level_widget.ax.tick_params(left=False, labelleft=False)
            self.level_widget.ax.patch.set_visible(False)
            self.level_widget.ax.yaxis.grid(False, which="both")

    @qc.pyqtSlot()
    def on_draw(self, lvl, stft, f0, conf):
        if (
            np.all(lvl == 0)
            or np.all(stft == 0)
            or np.all(f0 == 0)
            or np.all(conf == 0)
        ):
            return

        frames_spec = int(
            self.main_window.audio_processor.frame_rate * self.main_window.disp_t_spec
        )

        # prepare data
        if self.is_product:
            lvl_update = np.nan
            stft_update = np.flipud(np.prod(stft, axis=2))
            spec_update = np.prod(np.mean(stft[-frames_spec:, :, :], axis=0), axis=1)
            vline = None
        else:
            lvl_update = np.mean(lvl[:, self.ch_id])
            stft_update = np.flipud(stft[:, :, self.ch_id])
            spec_update = np.mean(stft[-frames_spec:, :, self.ch_id], axis=0)
            conf_update = np.mean(conf[-frames_spec:, self.ch_id])
            f0_update = np.mean(f0[-frames_spec:, self.ch_id])

            if conf_update > self.confidence_threshold:
                vline = f0_update
            else:
                vline = None

        # update widgets
        self.level_widget.on_draw(lvl_update)
        self.spectrum_widget.on_draw(spec_update, self.color, vline=vline)
        self.spectrogram_widget.on_draw(stft_update)

    def show_spectrum_widget(self, show):
        self.spectrum_widget.setVisible(show)

    def show_spectrogram_widget(self, show):
        self.spectrogram_widget.setVisible(show)

    def show_level_widget(self, show):
        self.level_widget.setVisible(show)

    @qc.pyqtSlot(int)
    def on_confidence_threshold_changed(self, threshold):
        """
        self.channel_views_widget.
        """
        self.confidence_threshold = threshold / 10.0

    @qc.pyqtSlot(float)
    def on_standard_frequency_changed(self, f):
        if isinstance(self.channel, list):
            for ch in self.channel:
                ch.standard_frequency = f
        else:
            self.channel.standard_frequency = f

    def on_min_freq_changed(self, f):
        self.spectrogram_widget.freq_min = f
        self.spectrum_widget.freq_min = f

    def on_max_freq_changed(self, f):
        self.spectrogram_widget.freq_max = f
        self.spectrum_widget.freq_max = f

    @qc.pyqtSlot(float)
    def on_pitch_shift_changed(self, shift):
        self.channel.pitch_shift = shift

    @qc.pyqtSlot(str)
    def on_spectrum_type_select(self, arg):
        """
        Slot to update the spectrum type
        """
        self.spectrum_widget.set_spectral_type(arg)
        self.spectrogram_widget.set_spectral_type(arg)


class LevelWidget(FigureCanvas):
    def __init__(self, channel_label):
        super(LevelWidget, self).__init__(Figure())

        self.figure = Figure(tight_layout=True)
        self.figure.tight_layout(pad=0)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title(None)
        self.ax.tick_params(axis="x", colors="white")
        self.ax.yaxis.grid(True, which="both")
        self.ax.set_xlabel("Level")
        self.ax.set_ylabel(f"{channel_label}", fontweight="bold")

        cvals = [0, 38, 40]
        colors = ["green", "yellow", "red"]
        norm = plt.Normalize(min(cvals), max(cvals))
        tuples = list(zip(map(norm, cvals), colors))
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)
        self.img = self.ax.imshow(
            np.linspace((0, 0), (40, 40), 400),
            cmap=cmap,
            aspect="auto",
            origin="lower",
            extent=[0, 1, -40, 0],
        )
        self.figure.subplots_adjust(
            left=0, right=0.8, top=1, bottom=0, wspace=0, hspace=0
        )
        self.ax.set_ylim((0, -40))
        self.ax.set_yticks([])
        self.ax.invert_yaxis()
        self.figure.set_tight_layout(True)

    def on_draw(self, lvl):
        if np.any(np.isnan(lvl)):
            plot_mat = np.full((2, 2), fill_value=np.nan)
        else:
            plot_val = np.max((0, 40 + lvl)).astype(int)
            plot_mat = np.linspace((0, 0), (40, 40), 400)
            plot_mat[plot_val * 10 + 10 :, :] = np.nan
        self.img.set_data(plot_mat)
        self.draw()


class SpectrumWidget(FigureCanvas):
    """
    Spectrum widget
    """

    def __init__(self, main_window: MainWindow):
        super(SpectrumWidget, self).__init__(Figure())

        self.main_window = main_window
        self.figure = Figure(tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111, position=[0, 0, 0, 0])
        self.ax.set_title("")
        self.ax.set_ylabel(None)
        self.ax.set_xlabel("Frequency [Hz]")
        self.ax.get_yaxis().set_visible(False)
        self.ax.xaxis.grid(True, which="both")
        self._line = None
        self._vline = None
        self.figure.tight_layout()

    def on_draw(self, data, color, vline=None):
        f_axis = self.main_window.audio_processor.fft_freqs
        if self._line is None:
            (self._line,) = self.ax.plot(
                f_axis, data, color=np.array(_colors[color]) / 256
            )
        else:
            self._line.set_data(f_axis, data)

        if self._vline is not None:
            try:
                self._vline.remove()
            except:
                pass
        if vline is not None:
            self._vline = self.ax.axvline(vline, c=np.array(_colors["black"]) / 256)

        self.ax.set_yscale(self.main_window.cur_spec_scale_type)
        self.ax.set_xlim(
            (self.main_window.disp_freq_lims[0], self.main_window.disp_freq_lims[1])
        )
        self.ax.relim()
        self.ax.autoscale(axis="y")
        self.figure.set_tight_layout(True)
        self.draw()


class SpectrogramWidget(FigureCanvas):
    """
    Spectrogram widget
    """

    def __init__(self, main_window: MainWindow):
        super(SpectrogramWidget, self).__init__(Figure())
        self.main_window = main_window
        self.figure = Figure(tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("")
        self.ax.set_ylabel(None)
        self.ax.get_yaxis().set_visible(False)
        self.img = None
        self.ax.set_xlim(
            (self.main_window.disp_freq_lims[0], self.main_window.disp_freq_lims[1])
        )
        self.ax.set_xlabel("Frequency [Hz]")
        self.ax.xaxis.grid(True, which="both")
        self.figure.tight_layout()

    def on_draw(self, data):
        if self.main_window.cur_spec_scale_type == "log":
            data = np.log(1 + 10 * data)

        if self.img is None:
            self.img = self.ax.imshow(
                data,
                origin="lower",
                aspect="auto",
                cmap="viridis",
                extent=[0, 4000, 0, 3],
            )
        else:
            self.img.set_data(data)
        self.img.set_clim(vmin=np.min(data), vmax=np.max(data))
        self.ax.set_xlim(
            (self.main_window.disp_freq_lims[0], self.main_window.disp_freq_lims[1])
        )

        self.figure.set_tight_layout(True)
        self.draw()


class PitchWidget(FigureCanvas):
    """Pitches of each trace as discrete samples."""

    low_pitch_changed = qc.pyqtSignal(np.ndarray)

    def __init__(self, main_window: MainWindow, *args, **kwargs):
        super(PitchWidget, self).__init__(Figure())
        self.main_window = main_window
        self.channel_views = main_window.channel_views.views[:-1]
        self.current_low_pitch = np.zeros(len(self.channel_views))
        self.current_low_pitch[:] = np.nan
        self.x_tick_pos = 0

        self.figure = Figure(tight_layout=True)
        self.ax = self.figure.add_subplot(111, position=[0, 0, 0, 0])
        self.ax.set_title(None)
        self.ax.set_ylabel("Relative Pitch [Cents]")
        self.ax.set_xlabel("Time")
        self.ax.yaxis.grid(True, which="both")
        self.ax.xaxis.grid(True, which="major")
        self.ax.set_ylim(main_window.disp_pitch_lims)
        self.ax.set_yticks(
            np.arange(
                main_window.disp_pitch_lims[0],
                main_window.disp_pitch_lims[1] + 100,
                100,
            )
        )
        self.ax.tick_params(
            labelbottom=True,
            labeltop=False,
            labelleft=True,
            labelright=True,
            bottom=True,
            top=False,
            left=True,
            right=True,
        )
        self.ax.set_xticklabels([])
        self._line = [None] * len(self.channel_views)
        self.figure.tight_layout()

        self.derivative_filter = 2000  # pitch/seconds
        self.reference_freq = 220  # Hz

        pal = self.palette()
        pal.setColor(qg.QPalette.ColorRole.Window, qg.QColor(*_colors["white"]))
        self.setPalette(pal)

    @qc.pyqtSlot(int)
    def on_derivative_filter_changed(self, max_derivative):
        self.derivative_filter = max_derivative

    def on_reference_frequency_changed(self, f):
        self.reference_freq = f

    def update_pitchlims(self):
        self.ax.set_ylim((self.parent.pitch_min, self.parent.pitch_max))
        self.ax.set_yticks(
            np.arange(self.parent.pitch_min, self.parent.pitch_max + 50, 50),
            minor=True,
        )
        self.ax.set_yticks(
            np.arange(self.parent.pitch_min, self.parent.pitch_max + 100, 100)
        )
        self.draw()

    @qc.pyqtSlot()
    def on_draw(self, f0, conf):
        if np.all(f0 == 0) or np.all(conf == 0):
            return

        cur_ref_freq_mode = self.main_window.cur_ref_freq_mode
        ref_freq = self.main_window.ref_freq

        # compute reference frequency
        if cur_ref_freq_mode == "fixed":
            cur_ref_freq = ref_freq
        elif cur_ref_freq_mode == "highest":
            cur_ref_freq = np.max(np.max(f0, axis=0))
        elif cur_ref_freq_mode == "lowest":
            cur_ref_freq = np.min(np.min(f0, axis=0))
        else:
            cur_ref_freq = f0[-1, int(cur_ref_freq_mode[-2:]) - 1]

        for i, cv in enumerate(self.channel_views):
            # filter f0 using confidence threshold and gradient filter
            index = np.where((conf[:, i] >= cv.confidence_threshold) & (f0[:, i] > 0))[
                0
            ]
            index_grad = index_gradient_filter(
                np.arange(f0.shape[0]), f0[:, i], self.derivative_filter
            )
            index = np.intersect1d(index, index_grad)

            f0_plot = np.full(f0.shape[0], np.nan)
            f0_plot[index] = f0[index, i]
            f0_plot = f2cent(f0_plot, cur_ref_freq)

            if self._line[i] is None:
                (self._line[i],) = self.ax.plot(
                    np.arange(len(f0_plot)),
                    f0_plot,
                    color=np.array(_colors[cv.color]) / 256,
                    linewidth="4",
                )
            else:
                self._line[i].set_ydata(f0_plot)

        self.ax.set_xlim(0, len(f0_plot))
        self.ax.set_xticks(
            np.arange(
                0,
                len(f0_plot),
                int(
                    np.round(
                        self.main_window.audio_processor.fs
                        / self.main_window.audio_processor.hop_len
                    )
                ),
            )
        )
        self.figure.set_tight_layout(True)
        self.draw()


class DifferentialPitchWidget(FigureCanvas):
    """Diffs as line"""

    def __init__(self, main_window: MainWindow, *args, **kwargs):
        super(DifferentialPitchWidget, self).__init__(Figure())
        self.main_window = main_window
        self.channel_views = main_window.channel_views.views[:-1]

        self.figure = Figure(tight_layout=True)
        self.ax = self.figure.add_subplot(111, position=[0, 0, 0, 0])
        self.ax.set_title(None)
        self.ax.set_ylabel("Pitch Difference [Cents]")
        self.ax.set_xlabel("Time")
        self.ax.yaxis.grid(True, which="both")
        self.ax.xaxis.grid(True, which="major")
        self.ax.set_ylim(main_window.disp_pitch_lims)
        self.ax.set_yticks(
            np.arange(
                main_window.disp_pitch_lims[0],
                main_window.disp_pitch_lims[1] + 100,
                100,
            )
        )
        self.ax.tick_params(
            labelbottom=True,
            labeltop=False,
            labelleft=True,
            labelright=True,
            bottom=True,
            top=False,
            left=True,
            right=True,
        )
        self.ax.set_xticklabels([])
        self._line = [[[None, None]] * len(self.channel_views)] * len(
            self.channel_views
        )
        self.figure.tight_layout()

        self.derivative_filter = 2000  # pitch/seconds
        self.reference_freq = 220

        pal = self.palette()
        pal.setColor(qg.QPalette.ColorRole.Window, qg.QColor(*_colors["white"]))
        self.setPalette(pal)

    @qc.pyqtSlot(int)
    def on_derivative_filter_changed(self, max_derivative):
        self.derivative_filter = max_derivative

    def on_reference_frequency_changed(self, f):
        self.reference_freq = f

    def update_pitchlims(self):
        self.ax.set_ylim((self.parent.pitch_min, self.parent.pitch_max))
        self.ax.set_yticks(
            np.arange(self.parent.pitch_min, self.parent.pitch_max + 50, 50),
            minor=True,
        )
        self.ax.set_yticks(
            np.arange(self.parent.pitch_min, self.parent.pitch_max + 100, 100)
        )
        self.draw()

    @qc.pyqtSlot()
    def on_draw(self, f0, conf):
        if len(self.main_window.audio_processor.channels) == 1:
            return

        if np.all(f0 == 0) or np.all(conf == 0):
            return

        cur_ref_freq_mode = self.main_window.cur_ref_freq_mode
        cur_ref_freq = self.main_window.cur_ref_freq

        # compute reference frequency
        if cur_ref_freq_mode == "fixed":
            cur_ref_freq = cur_ref_freq
        elif cur_ref_freq_mode == "highest":
            cur_ref_freq = np.max(np.max(f0, axis=0))
        elif cur_ref_freq_mode == "lowest":
            cur_ref_freq = np.min(np.min(f0, axis=0))
        else:
            cur_ref_freq = f0[-1, int(cur_ref_freq_mode[-2:]) - 1]

        for ch0, cv0 in enumerate(self.channel_views):
            for ch1, cv1 in enumerate(self.channel_views):
                if ch0 >= ch1:
                    continue

                index = np.where(
                    (conf[:, ch0] >= cv0.confidence_threshold)
                    & (conf[:, ch1] >= cv1.confidence_threshold)
                    & (f0[:, ch0] > 0)
                    & (f0[:, ch1] > 0)
                )[0]
                index_grad0 = index_gradient_filter(
                    np.arange(f0.shape[0]), f0[:, ch0], self.derivative_filter
                )
                index_grad1 = index_gradient_filter(
                    np.arange(f0.shape[0]), f0[:, ch1], self.derivative_filter
                )
                index = np.intersect1d(np.intersect1d(index, index_grad0), index_grad1)

                f0_plot = np.full(f0.shape[0], np.nan)
                f0_plot[index] = f2cent(f0[index, ch0], cur_ref_freq) - f2cent(
                    f0[index, ch1], cur_ref_freq
                )

                if self._line[ch0][ch1][0] is None:
                    (self._line[ch0][ch1][0],) = self.ax.plot(
                        np.arange(len(f0_plot)),
                        f0_plot,
                        color=np.array(_colors[cv0.color]) / 256,
                        linewidth="4",
                        linestyle="-",
                    )
                else:
                    self._line[ch0][ch1][0].set_ydata(f0_plot)

                if self._line[ch0][ch1][1] is None:
                    (self._line[ch0][ch1][1],) = self.ax.plot(
                        np.arange(len(f0_plot)),
                        f0_plot,
                        color=np.array(_colors[cv1.color]) / 256,
                        linewidth="4",
                        linestyle="--",
                    )
                else:
                    self._line[ch0][ch1][1].set_ydata(f0_plot)

            self.ax.set_xlim(0, len(f0_plot))
            self.ax.set_xticks(
                np.arange(
                    0,
                    len(f0_plot),
                    int(
                        np.round(
                            self.main_window.audio_processor.fs
                            / self.main_window.audio_processor.hop_len
                        )
                    ),
                )
            )
            self.figure.set_tight_layout(True)
            self.draw()


class TrajectoryViews(qw.QTabWidget):
    """Widget for right tabs."""

    def __init__(self, main_window: MainWindow, *args, **kwargs):
        qw.QTabWidget.__init__(self, *args, **kwargs)

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.setSizePolicy(
            qw.QSizePolicy.Policy.MinimumExpanding,
            qw.QSizePolicy.Policy.MinimumExpanding,
        )
        # self.setAutoFillBackground(True)
        # pal = self.palette()
        # pal.setColor(qg.QPalette.ColorRole.Window, qg.QColor(*_colors["white"]))
        # self.setPalette(pal)

        self.setStyleSheet(
            """
                    QTabWidget::pane {
                        background-color: white;
                    }
                    QTabWidget::tab-bar {
                        background: white;
                    }
                    QWidget {
                        background-color: white;
                    }
                """
        )

        self.pitch_view = PitchWidget(main_window)
        self.pitch_view_all_diff = DifferentialPitchWidget(main_window)

        # remove old tabs from pitch view
        self.addTab(self.pitch_view, "Pitches")
        self.addTab(self.pitch_view_all_diff, "Differential")

    def on_draw(self, f0, conf):
        self.pitch_view.on_draw(f0, conf)
        self.pitch_view_all_diff.on_draw(f0, conf)

    def sizeHint(self):
        """Makes sure the widget is show with the right aspect."""
        return qc.QSize(500, 200)


class MainWidget(qw.QWidget):
    """Main widget that contains the menu and the visualization widgets."""

    signal_widgets_clear = qc.pyqtSignal()
    signal_widgets_draw = qc.pyqtSignal()

    def __init__(self, *args, **kwargs):
        qw.QWidget.__init__(self, *args, **kwargs)
        self.tabbed_pitch_widget = TrajectoryViews(parent=self)

        pal = self.palette()
        self.setAutoFillBackground(True)
        pal.setColor(qg.QPalette.ColorRole.Window, qg.QColor(*_colors["white"]))
        self.setPalette(pal)

        self.setMouseTracking(True)
        self.top_layout = qw.QGridLayout()
        self.setLayout(self.top_layout)

        self.refresh_timer = qc.QTimer()
        self.refresh_timer.timeout.connect(self.refresh_widgets)
        self.menu = ProcessingMenu(self)

        self.data_input = None
        self.freq_max = 1000
        self.pitch_min = -1500
        self.pitch_max = 1500

    @qc.pyqtSlot(str)
    def on_algorithm_select(self, arg):
        """Change pitch algorithm."""
        for c in self.data_input.channels:
            c.f0_algorithms = arg

    def on_pitch_min_changed(self, p):
        """Update axis min limit."""
        self.pitch_min = p
        self.pitch_view.update_pitchlims()
        self.pitch_view_all_diff.update_pitchlims()

    def on_pitch_max_changed(self, p):
        """Update axis max limit."""
        self.pitch_max = p
        self.pitch_view.update_pitchlims()
        self.pitch_view_all_diff.update_pitchlims()

    def cleanup(self):
        """Clear all widgets."""
        if self.data_input:
            self.data_input.stop()
            self.data_input.terminate()

        while self.top_layout.count():
            item = self.top_layout.takeAt(0)
            item.widget().deleteLater()

    def audio_config_changed(self):
        """Initializes widgets and makes connections."""

        # setup and start audio processor
        audio_processor.fs = self.input_dialog.selected_fs
        audio_processor.fft_len = self.input_dialog.selected_fftsize
        audio_processor.channels = self.input_dialog.selected_channels
        audio_processor.device_no = self.input_dialog.selected_device
        audio_processor.init_buffers()
        audio_processor.start_stream()

        # prepare widgets
        self.channel_views_widget = ChannelViews(
            channels=len(audio_processor.channels), freq_max=self.freq_max
        )
        channel_views = self.channel_views_widget.views[:-1]
        for cv in channel_views:
            self.menu.connect_to_confidence_threshold(cv)
        self.signal_widgets_draw.connect(self.channel_views_widget.on_draw)

        self.top_layout.addWidget(self.channel_views_widget, 1, 0, 1, 1)

        self.pitch_view = PitchWidget(self, channel_views)
        self.pitch_view_all_diff = DifferentialPitchWidget(self, channel_views)

        # remove old tabs from pitch view
        self.tabbed_pitch_widget.clear()
        self.tabbed_pitch_widget.addTab(self.pitch_view, "Pitches")
        self.tabbed_pitch_widget.addTab(self.pitch_view_all_diff, "Differential")

        self.menu.derivative_filter_slider.valueChanged.connect(
            self.pitch_view.on_derivative_filter_changed
        )
        self.menu.derivative_filter_slider.valueChanged.connect(
            self.pitch_view_all_diff.on_derivative_filter_changed
        )
        self.menu.connect_channel_views(
            self.channel_views_widget, self.pitch_view, self.pitch_view_all_diff
        )

        self.signal_widgets_draw.connect(self.pitch_view.on_draw)
        self.signal_widgets_draw.connect(self.pitch_view_all_diff.on_draw)

        self.menu.select_algorithm.currentTextChanged.connect(self.on_algorithm_select)
        self.menu.pitch_max.accepted_value.connect(self.on_pitch_max_changed)
        self.menu.pitch_min.accepted_value.connect(self.on_pitch_min_changed)

        self.refresh_timer.start(gui_refresh)  # start refreshing GUI

    @qc.pyqtSlot()
    def refresh_widgets(self):
        """This is the main refresh loop."""
        # self.worker.process()
        self.signal_widgets_clear.emit()
        self.signal_widgets_draw.emit()

    def closeEvent(self, ev):
        """Called when application is closed."""
        logger.info("closing")
        self.data_input.terminate()
        self.cleanup()
        qw.QWidget.closeEvent(self, ev)


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
