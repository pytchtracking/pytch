#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""GUI Functions"""

import logging
import sys
import time
import numpy as np
import math
import importlib.metadata

from .utils import (
    consecutive,
    index_gradient_filter,
    f2cent,
    cent2f,
    FloatQLineEdit,
    QHLine,
    BlitManager
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
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

logger = logging.getLogger("pytch.gui")


def start_gui():
    """Starts the GUI"""
    app = qw.QApplication(sys.argv)
    input_dialog = InputMenu()
    if input_dialog.exec() == qw.QDialog.DialogCode.Accepted:
        device, channels, fs, fft_size = input_dialog.get_input_settings()
        main_window = MainWindow(
            sounddevice_idx=device, channels=channels, fs=fs, fft_size=fft_size
        )
        main_window.showMaximized()
        app.exec()
    sys.exit()


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

        default_device = self.devices[0]
        for idevice, device in enumerate(self.devices):
            self.input_options.addItem("{} {}".format(idevice, device[1]["name"]))

        # select sampling rate
        layout.addWidget(qw.QLabel("Sampling Rate"))
        self.fs_options = qw.QComboBox()
        layout.addWidget(self.fs_options)

        # select fft size
        layout.addWidget(qw.QLabel("FFT Size in Samples"))
        self.fft_size_options = self.get_nfft_box()
        layout.addWidget(self.fft_size_options)

        self.channel_options = qw.QScrollArea()
        self.channel_options.setMaximumSize(30000, 200)
        layout.addWidget(qw.QLabel("Select Channels"), 0, 2, 1, 1)
        layout.addWidget(self.channel_options, 1, 2, 6, 1)

        buttons = qw.QDialogButtonBox(
            qw.QDialogButtonBox.StandardButton.Ok
            | qw.QDialogButtonBox.StandardButton.Cancel
        )

        self.input_options.currentIndexChanged.connect(self.update_channel_info)
        self.input_options.setCurrentIndex(0)
        self.update_channel_info(0)

        buttons.accepted.connect(self.on_ok_clicked)
        buttons.rejected.connect(self.close)
        layout.addWidget(buttons)

    def update_channel_info(self, menu_index):
        """Updates available channels in input menu"""
        sounddevice_index, device = self.devices[menu_index]
        nmax_channels = device["max_input_channels"]

        sampling_rate_options = get_fs_options(sounddevice_index)
        self.channel_selector = ChannelSelector(
            nchannels=nmax_channels, channels_enabled=[0]
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
        sounddevice_idx = self.devices[self.input_options.currentIndex()][0]
        channels = self.channel_selector.get_selected_channels()
        fs = int(self.fs_options.currentText())
        fft_size = int(self.fft_size_options.currentText())
        return sounddevice_idx, channels, fs, fft_size


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


class MainWindow(qw.QMainWindow):
    """Main window that includes the main widget for the menu and all visualizations."""

    def __init__(self, sounddevice_idx, channels, fs, fft_size):
        super().__init__()

        # default settings for the entire GUI.
        self.version = importlib.metadata.version("pytch")
        self.sounddevice_idx = sounddevice_idx
        self.channels = channels
        self.fs = fs
        self.fft_size = fft_size
        self.f0_algorithms = ["YIN", "SWIPE"]
        self.buf_len_sec = 30.0
        self.disp_pitch_lims = [
            -1500,
            1500,
        ]  # limits in cents for pitch trajectory view
        self.gui_refresh_ms = int(1000 / 25)  # equivalent to 25 fps
        self.spec_scale_types = ["log", "linear"]
        self.ref_freq_modes = ["fixed", "highest", "lowest"]
        self.disp_t_lvl = 1
        self.disp_t_spec = 1
        self.disp_t_stft = 25
        self.disp_t_f0 = 25
        self.disp_t_conf = 25
        self.cur_disp_freq_lims = [20, 1000]  # limits in Hz for spectrum/spectrogram view
        self.cur_spec_scale_type = self.spec_scale_types[0]
        self.cur_ref_freq_mode = self.ref_freq_modes[0]
        self.cur_ref_freq = 220
        self.cur_conf_threshold = 0.5
        self.cur_derivative_tol = 600

        # status variables
        self.is_running = False
        self.menu_visible = True

        # styling
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
            device_no=self.sounddevice_idx,
            f0_algorithm=self.f0_algorithms[0],
        )

        # initialize GUI
        self.setWindowTitle(f"Pytch {self.version}")
        central_widget = qw.QWidget()  # contains all contents
        self.setCentralWidget(central_widget)

        splitter = QSplitter(
            qc.Qt.Orientation.Horizontal, central_widget
        )  # split GUI horizontally

        self.channel_views = ChannelViews(self)  # channel views
        self.trajectory_views = TrajectoryViews(self)  # trajectory views
        self.menu = ProcessingMenu(self)  # left-hand menu

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
        self.last_refresh = time.time()

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
        cur_t = time.time()
        logger.info(f"Diff last refresh {cur_t-self.last_refresh}")
        self.last_refresh = cur_t
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

        self.main_window = main_window

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
        self.box_show_levels.stateChanged.connect(
            main_window.channel_views.show_level_widgets
        )
        layout.addWidget(self.box_show_levels, 1, 1, 1, 1)

        layout.addWidget(qw.QLabel("Spectra"), 2, 0)
        self.box_show_spectra = qw.QCheckBox()
        self.box_show_spectra.setChecked(True)
        self.box_show_spectra.stateChanged.connect(
            main_window.channel_views.show_spectrum_widgets
        )
        layout.addWidget(self.box_show_spectra, 2, 1, 1, 1)

        layout.addWidget(qw.QLabel("Spectrograms"), 3, 0)
        self.box_show_spectrograms = qw.QCheckBox()
        self.box_show_spectrograms.setChecked(True)
        self.box_show_spectrograms.stateChanged.connect(
            main_window.channel_views.show_spectrogram_widgets
        )
        layout.addWidget(self.box_show_spectrograms, 3, 1, 1, 1)

        layout.addWidget(qw.QLabel("Products"), 4, 0)
        self.box_show_products = qw.QCheckBox()
        self.box_show_products.setChecked(True)
        self.box_show_products.stateChanged.connect(
            main_window.channel_views.show_product_widgets
        )
        layout.addWidget(self.box_show_products, 4, 1, 1, 1)

        layout.addWidget(qw.QLabel("Minimum Frequency"), 5, 0)
        self.freq_min = FloatQLineEdit(
            parent=self, default=main_window.cur_disp_freq_lims[0]
        )
        layout.addWidget(self.freq_min, 5, 1, 1, 1)
        self.freq_min.accepted_value.connect(self.on_min_freq_changed)
        layout.addWidget(qw.QLabel("Hz"), 5, 2)

        layout.addWidget(qw.QLabel("Maximum Frequency"), 6, 0)
        self.freq_max = FloatQLineEdit(
            parent=self, default=main_window.cur_disp_freq_lims[1]
        )
        layout.addWidget(self.freq_max, 6, 1, 1, 1)
        self.freq_max.accepted_value.connect(self.on_max_freq_changed)
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
        self.box_show_tv = qw.QCheckBox()
        self.box_show_tv.setChecked(True)
        self.box_show_tv.stateChanged.connect(
            main_window.trajectory_views.show_trajectory_views
        )
        layout.addWidget(self.box_show_tv, 10, 1, 1, 1)

        layout.addWidget(qw.QLabel("F0 Algorithm"), 11, 0)
        self.select_algorithm = qw.QComboBox(self)
        self.select_algorithm.addItems(main_window.f0_algorithms)
        self.select_algorithm.setCurrentIndex(0)
        self.select_algorithm.currentTextChanged.connect(self.on_algorithm_select)
        layout.addWidget(self.select_algorithm, 11, 1, 1, 1)

        layout.addWidget(qw.QLabel("Confidence Threshold"), 12, 0)
        self.noise_thresh_slider = qw.QSlider()
        self.noise_thresh_slider.setRange(0, 10)
        self.noise_thresh_slider.setValue(int(main_window.cur_conf_threshold * 10))
        self.noise_thresh_slider.setOrientation(qc.Qt.Orientation.Horizontal)
        self.noise_thresh_slider.valueChanged.connect(self.on_conf_threshold_changed)
        self.noise_thresh_label = qw.QLabel(
            f"{self.noise_thresh_slider.value() / 10.0}"
        )
        layout.addWidget(self.noise_thresh_slider, 12, 1, 1, 1)
        layout.addWidget(self.noise_thresh_label, 12, 2)

        layout.addWidget(qw.QLabel("Pitchslide Tolerance [Cents]"), 13, 0)
        self.derivative_tol_slider = qw.QSlider()
        self.derivative_tol_slider.setRange(0, 1200)
        self.derivative_tol_slider.setValue(main_window.cur_derivative_tol)
        self.derivative_tol_slider.setOrientation(qc.Qt.Orientation.Horizontal)
        self.derivative_tol_label = qw.QLabel(f"{self.derivative_tol_slider.value()}")
        self.derivative_tol_slider.valueChanged.connect(self.on_derivative_tol_changed)
        layout.addWidget(self.derivative_tol_label, 13, 2)
        layout.addWidget(self.derivative_tol_slider, 13, 1, 1, 1)

        layout.addWidget(qw.QLabel("Reference Mode"), 14, 0)
        self.ref_freq_mode_menu = qw.QComboBox()
        if len(self.main_window.channels) > 1:
            self.ref_freq_mode_menu.addItems(main_window.ref_freq_modes)
        else:
            self.ref_freq_mode_menu.addItems([main_window.ref_freq_modes[0]])
        self.ref_freq_mode_menu.setCurrentIndex(0)
        self.ref_freq_mode_menu.currentTextChanged.connect(self.on_reference_frequency_mode_changed)
        layout.addWidget(self.ref_freq_mode_menu, 14, 1, 1, 1)

        layout.addWidget(qw.QLabel("Reference Frequency"), 15, 0)
        self.freq_box = FloatQLineEdit(parent=self, default=main_window.cur_ref_freq)
        self.freq_box.accepted_value.connect(self.on_reference_frequency_changed)
        layout.addWidget(self.freq_box, 15, 1, 1, 1)
        layout.addWidget(qw.QLabel("Hz"), 15, 2)

        layout.addWidget(qw.QLabel("Minimum Pitch"), 16, 0)
        self.pitch_min = FloatQLineEdit(
            parent=self, default=main_window.disp_pitch_lims[0]
        )
        self.pitch_min.accepted_value.connect(self.on_pitch_min_changed)
        layout.addWidget(self.pitch_min, 16, 1, 1, 1)
        layout.addWidget(qw.QLabel("Cents"), 16, 2)

        layout.addWidget(qw.QLabel("Maximum Pitch"), 17, 0)
        self.pitch_max = FloatQLineEdit(
            parent=self, default=main_window.disp_pitch_lims[1]
        )
        self.pitch_max.accepted_value.connect(self.on_pitch_max_changed)
        layout.addWidget(self.pitch_max, 17, 1, 1, 1)
        layout.addWidget(qw.QLabel("Cents"), 17, 2)

        settings.setLayout(layout)
        main_layout.addWidget(settings, 3, 0, 1, 2)

    def on_min_freq_changed(self, f):
        self.main_window.cur_disp_freq_lims[0] = int(f)

    def on_max_freq_changed(self, f):
        self.main_window.cur_disp_freq_lims[1] = int(f)

    def on_algorithm_select(self, algorithm):
        self.main_window.audio_processor.f0_algorithm = algorithm

    def on_conf_threshold_changed(self, val):
        self.noise_thresh_label.setText(str(val / 10.0))
        self.main_window.cur_conf_threshold = val / 10.0

    def on_derivative_tol_changed(self, val):
        self.derivative_tol_label.setText(str(val))
        self.main_window.cur_derivative_tol = val

    def on_reference_frequency_mode_changed(self, text):
        if (text == "Highest") or (text == "Lowest") or ("Channel" in text):
            self.freq_box.setReadOnly(True)
        else:
            self.freq_box.setText("220")
            self.freq_box.do_check()
            self.freq_box.setReadOnly(False)

        self.main_window.cur_ref_freq_mode = text

    def on_reference_frequency_changed(self, val):
        self.main_window.cur_ref_freq = val

    def on_pitch_min_changed(self, val):
        self.main_window.disp_pitch_lims[0] = int(val)

    def on_pitch_max_changed(self, val):
        self.main_window.disp_pitch_lims[1] = int(val)

    def on_spectrum_type_select(self, arg):
        self.main_window.cur_spec_scale_type = arg

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
                self.h_line = QHLine()
                self.layout.addWidget(self.h_line)
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
        self.h_line.setVisible(show)

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

        self.color = "black" if ch_id is None else np.array(_colors[_color_names[3 * ch_id]]) / 256
        self.is_product = is_product
        self.ch_id = ch_id

        channel_label = "Product" if ch_id is None else f"Channel {ch_id + 1}"
        self.level_widget = LevelWidget(channel_label=channel_label)
        self.spectrogram_widget = SpectrogramWidget(self.main_window, self.color)
        self.spectrum_widget = SpectrumWidget(self.main_window, self.color)

        layout = self.layout()
        layout.addWidget(self.level_widget, 2)
        layout.addWidget(self.spectrum_widget, 7)
        layout.addWidget(self.spectrogram_widget, 7)

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
            stft_update = np.prod(stft, axis=2)
            spec_update = np.prod(np.mean(stft[-frames_spec:, :, :], axis=0), axis=1)
            vline = np.nan
        else:
            lvl_update = np.mean(lvl[:, self.ch_id])
            stft_update = stft[:, :, self.ch_id]
            spec_update = np.mean(stft[-frames_spec:, :, self.ch_id], axis=0)
            conf_update = np.mean(conf[-frames_spec:, self.ch_id])
            f0_update = np.mean(f0[-frames_spec:, self.ch_id])

            if self.ch_id == 3:
                pass

            if ((conf_update > self.main_window.cur_conf_threshold) and
                    (lvl_update >= self.level_widget.cvals[0])):
                vline = f0_update
            else:
                vline = np.nan

        # update widgets
        self.level_widget.on_draw(lvl_update)
        self.spectrum_widget.on_draw(spec_update, vline=vline)
        self.spectrogram_widget.on_draw(stft_update)

    def show_spectrum_widget(self, show):
        self.spectrum_widget.setVisible(show)

    def show_spectrogram_widget(self, show):
        self.spectrogram_widget.setVisible(show)

    def show_level_widget(self, show):
        self.level_widget.setVisible(show)


class LevelWidget(FigureCanvas):
    def __init__(self, channel_label):
        self.figure = Figure(tight_layout=True)
        super(LevelWidget, self).__init__(self.figure)

        self.ax = self.figure.add_subplot(111)
        self.ax.set_title(None)
        self.ax.tick_params(axis="x", colors="white")
        self.ax.set_yticks([])
        self.ax.yaxis.grid(True, which="both")
        self.ax.set_xlabel("Level")
        self.ax.set_ylabel(f"{channel_label}", fontweight="bold")

        self.cvals = [-80, -12, 0]
        colors = ["green", "yellow", "red"]
        norm = plt.Normalize(min(self.cvals), max(self.cvals))
        tuples = list(zip(map(norm, self.cvals), colors))
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)
        self.plot_mat_tmp = np.linspace(self.cvals[0],
                                        self.cvals[-1],
                                        np.abs(self.cvals[0]-self.cvals[-1])).reshape(-1, 1)
        self.img = self.ax.imshow(
            self.plot_mat_tmp * np.nan,
            cmap=cmap,
            aspect="auto",
            origin="lower",
            extent=[0, 1, self.cvals[0], self.cvals[-1]],
        )
        self.img.set_clim(vmin=self.cvals[0], vmax=self.cvals[-1])
        self.ax.set_ylim((self.cvals[-1], self.cvals[0]))
        self.ax.invert_yaxis()
        self.frame = [self.ax.spines[side] for side in self.ax.spines]
        self.draw()
        self.bm = BlitManager(self.figure.canvas, [self.img] + self.frame)

    def on_draw(self, lvl):
        if np.any(np.isnan(lvl)):
            plot_mat = self.plot_mat_tmp.copy() * np.nan
        else:
            lvl_clip = int(np.round(np.clip(lvl, a_min=self.cvals[0], a_max=self.cvals[-1])))
            plot_mat = self.plot_mat_tmp.copy()
            plot_mat[lvl_clip:, :] = np.nan

        self.img.set_data(plot_mat)

        self.bm.update()


class SpectrumWidget(FigureCanvas):
    """
    Spectrum widget
    """

    def __init__(self, main_window: MainWindow, color):
        self.figure = Figure(tight_layout=True)
        super(SpectrumWidget, self).__init__(self.figure)

        self.main_window = main_window
        self.color = color
        self.f_axis = self.main_window.audio_processor.fft_freqs
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("")
        self.ax.set_xlabel("Frequency [Hz]")
        self.ax.set_ylabel("Magnitude")
        self.ax.get_yaxis().set_visible(False)
        self.ax.xaxis.grid(True, which="major")
        self.ax.xaxis.minorticks_on()
        (self._line, ) = self.ax.plot(
                self.f_axis, np.zeros_like(self.f_axis), color=self.color
            )
        self._vline = self.ax.axvline(np.nan, c=self.color, linestyle="--")
        self.cur_disp_freq_lims = self.main_window.cur_disp_freq_lims.copy()
        self.ax.set_xlim(self.main_window.cur_disp_freq_lims)
        self.ax.set_ylim((0, 1))
        self.draw()
        self.bm = BlitManager(self.figure.canvas, [self._line, self._vline])

    def on_draw(self, data, vline=None):
        if self.main_window.cur_spec_scale_type == "log":
            data = np.log(1 + 1 * data)

        data_plot = data / np.max(np.abs(data))

        if self.cur_disp_freq_lims != self.main_window.cur_disp_freq_lims:
            self.ax.set_xlim(self.main_window.cur_disp_freq_lims)
            self.cur_disp_freq_lims = self.main_window.cur_disp_freq_lims.copy()
            self.bm.update_bg()

        self._line.set_ydata(data_plot)
        self._vline.set_xdata([vline, vline])

        self.bm.update()


class SpectrogramWidget(FigureCanvas):
    """
    Spectrogram widget
    """

    def __init__(self, main_window: MainWindow, color):
        self.figure = Figure(tight_layout=True)
        super(SpectrogramWidget, self).__init__(self.figure)

        self.main_window = main_window
        # custom colormap or viridis
        self.cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap",
                                                               list(zip([0, 1], ["w", color])))
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("")
        self.ax.set_ylabel("Time")
        self.ax.get_yaxis().set_visible(False)
        self.ax.xaxis.grid(True, which="major")
        self.ax.xaxis.minorticks_on()
        self.grid_lines = [line for line in self.ax.get_xgridlines()]
        self.frame = [self.ax.spines[side] for side in self.ax.spines]
        self.img = self.ax.imshow(
                np.zeros((len(self.main_window.audio_processor.fft_freqs),
                          int(np.round(self.main_window.audio_processor.frame_rate *
                                       self.main_window.disp_t_spec)))),
                origin="lower",
                aspect="auto",
                cmap=self.cmap,
                extent=[0, self.main_window.audio_processor.fs // 2, 0, self.main_window.disp_t_spec],
            )
        self.img.set_clim(vmin=0, vmax=1)
        self.ax.set_xlabel("Frequency [Hz]")
        self.cur_disp_freq_lims = self.main_window.cur_disp_freq_lims.copy()
        self.ax.set_xlim(self.main_window.cur_disp_freq_lims)
        self.draw()
        self.bm = BlitManager(self.figure.canvas, [self.img] + self.grid_lines + self.frame)

    def on_draw(self, data):
        if self.main_window.cur_spec_scale_type == "log":
            data = np.log(1 + 1 * data)

        data_plot = data / np.max(np.abs(data))
        self.img.set_data(data_plot)

        if self.cur_disp_freq_lims != self.main_window.cur_disp_freq_lims:
            self.ax.set_xlim(self.main_window.cur_disp_freq_lims)
            self.cur_disp_freq_lims = self.main_window.cur_disp_freq_lims.copy()
            self.bm.update_bg()

        self.bm.update()


class TrajectoryViews(qw.QTabWidget):
    """Widget for right tabs."""

    def __init__(self, main_window: MainWindow, *args, **kwargs):
        qw.QTabWidget.__init__(self, *args, **kwargs)

        self.main_window = main_window

        # styling
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.setSizePolicy(
            qw.QSizePolicy.Policy.MinimumExpanding,
            qw.QSizePolicy.Policy.MinimumExpanding,
        )
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
        if len(main_window.channels) > 1:
            self.addTab(self.pitch_view_all_diff, "Differential")

    def on_draw(self, f0, conf):
        self.pitch_view.on_draw(f0, conf)
        if len(self.main_window.channels) > 1:
            self.pitch_view_all_diff.on_draw(f0, conf)

    def show_trajectory_views(self, show):
        self.setVisible(show)

    def sizeHint(self):
        """Makes sure the widget is show with the right aspect."""
        return qc.QSize(500, 200)


class PitchWidget(FigureCanvas):
    """Pitches of each trace as discrete samples."""

    low_pitch_changed = qc.pyqtSignal(np.ndarray)

    def __init__(self, main_window: MainWindow, *args, **kwargs):
        self.figure = Figure(tight_layout=True)
        super(PitchWidget, self).__init__(self.figure)

        self.main_window = main_window
        self.channel_views = main_window.channel_views.views[:-1]

        self.ax = self.figure.add_subplot(111, position=[0, 0, 0, 0])
        self.ax.set_title(None)
        self.ax.set_ylabel("Relative Pitch [Cents]")
        self.ax.set_xlabel("Time")
        self.ax.yaxis.grid(True, which="both")
        self.ax.xaxis.grid(True, which="major")
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
        self.t_axis = np.arange(int(np.round(self.main_window.audio_processor.frame_rate *
                                             self.main_window.disp_t_f0)))
        self.f0_tmp = np.full(len(self.t_axis), np.nan)
        self._lines = []
        for i in range(len(self.channel_views)):
            self._lines.append(self.ax.plot(
                self.t_axis,
                self.f0_tmp,
                color=self.channel_views[i].color,
                linewidth="3",
            )[0])
        self.cur_disp_pitch_lims = self.main_window.disp_pitch_lims.copy()
        self.set_axes_limits()

        pal = self.palette()
        pal.setColor(qg.QPalette.ColorRole.Window, qg.QColor(*_colors["white"]))
        self.setPalette(pal)

        self.draw()

        self.grid_lines = ([line for line in self.ax.get_xgridlines()] +
                           [line for line in self.ax.get_ygridlines()])

        self.bm = BlitManager(self.figure.canvas, self._lines + self.grid_lines)

    def set_axes_limits(self):
        self.ax.set_xlim(0, len(self.t_axis))
        self.ax.set_xticks(
            np.arange(
                0,
                len(self.t_axis),
                self.main_window.audio_processor.frame_rate,
            )
        )
        self.ax.set_xticklabels([])
        self.ax.set_ylim(self.cur_disp_pitch_lims)
        self.ax.set_yticks(
            np.arange(self.cur_disp_pitch_lims[0],
                      self.cur_disp_pitch_lims[1] + 100, 100)
        )

    @qc.pyqtSlot()
    def on_draw(self, f0, conf):
        if np.all(f0 == 0) or np.all(conf == 0):
            return

        cur_ref_freq_mode = self.main_window.cur_ref_freq_mode
        ref_freq = self.main_window.cur_ref_freq

        # compute reference frequency
        if cur_ref_freq_mode == "fixed":
            cur_ref_freq = ref_freq
        elif cur_ref_freq_mode == "highest":
            cur_ref_freq = np.max(np.mean(f0, axis=0))
        elif cur_ref_freq_mode == "lowest":
            cur_ref_freq = np.min(np.mean(f0, axis=0))
        else:
            cur_ref_freq = f0[-1, int(cur_ref_freq_mode[-2:]) - 1]

        for i, cv in enumerate(self.channel_views):
            # filter f0 using confidence threshold and gradient filter
            index = np.where((conf[:, i] >= self.main_window.cur_conf_threshold) & (f0[:, i] > 0))[
                0
            ]
            index_grad = index_gradient_filter(
                np.arange(f0.shape[0]), f0[:, i], self.main_window.cur_derivative_tol
            )
            index = np.intersect1d(index, index_grad)

            f0_plot = self.f0_tmp.copy()
            f0_plot[index] = f0[index, i]
            f0_plot = f2cent(f0_plot, cur_ref_freq)
            self._lines[i].set_ydata(f0_plot)

        if self.cur_disp_pitch_lims != self.main_window.disp_pitch_lims:
            self.cur_disp_pitch_lims = self.main_window.disp_pitch_lims.copy()
            self.set_axes_limits()
            self.bm.update_bg()

        self.bm.update()


class DifferentialPitchWidget(FigureCanvas):
    """Diffs as line"""

    def __init__(self, main_window: MainWindow, *args, **kwargs):
        self.figure = Figure(tight_layout=True)
        super(DifferentialPitchWidget, self).__init__(self.figure)
        self.main_window = main_window
        self.channel_views = main_window.channel_views.views[:-1]

        self.ax = self.figure.add_subplot(111, position=[0, 0, 0, 0])
        self.ax.set_title(None)
        self.ax.set_ylabel("Pitch Difference [Cents]")
        self.ax.set_xlabel("Time")
        self.ax.yaxis.grid(True, which="both")
        self.ax.xaxis.grid(True, which="major")
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
        self.t_axis = np.arange(int(np.round(self.main_window.audio_processor.frame_rate *
                                             self.main_window.disp_t_f0)))
        self.f0_tmp = np.full(len(self.t_axis), np.nan)
        self._lines = []
        for ch0, cv0 in enumerate(self.channel_views):
            for ch1, cv1 in enumerate(self.channel_views):
                if ch0 >= ch1:
                    continue

                (line1,) = self.ax.plot(
                    self.t_axis,
                    self.f0_tmp,
                    color=cv0.color,
                    linewidth="3",
                    linestyle="-",
                )

                (line2,) = self.ax.plot(
                    self.t_axis,
                    self.f0_tmp,
                    color=cv1.color,
                    linewidth="3",
                    linestyle=":",
                )

                self._lines.append([line1, line2])

        self.cur_disp_pitch_lims = self.main_window.disp_pitch_lims.copy()
        self.set_axes_limits()

        pal = self.palette()
        pal.setColor(qg.QPalette.ColorRole.Window, qg.QColor(*_colors["white"]))
        self.setPalette(pal)

        self.draw()

        self.grid_lines = ([line for line in self.ax.get_xgridlines()] +
                           [line for line in self.ax.get_ygridlines()])
        flat_list = []
        for row in self._lines:
            flat_list += row
        self.bm = BlitManager(self.figure.canvas, flat_list + self.grid_lines)

    def set_axes_limits(self):
        self.ax.set_xlim(0, len(self.t_axis))
        self.ax.set_xticks(
            np.arange(
                0,
                len(self.t_axis),
                self.main_window.audio_processor.frame_rate,
            )
        )
        self.ax.set_xticklabels([])
        self.ax.set_ylim(self.cur_disp_pitch_lims)
        self.ax.set_yticks(
            np.arange(self.cur_disp_pitch_lims[0],
                      self.cur_disp_pitch_lims[1] + 100, 100)
        )

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

        pair_num = 0
        for ch0, cv0 in enumerate(self.channel_views):
            for ch1, cv1 in enumerate(self.channel_views):
                if ch0 >= ch1:
                    continue

                index = np.where(
                    (conf[:, ch0] >= self.main_window.cur_conf_threshold)
                    & (conf[:, ch1] >= self.main_window.cur_conf_threshold)
                    & (f0[:, ch0] > 0)
                    & (f0[:, ch1] > 0)
                )[0]
                index_grad0 = index_gradient_filter(
                    self.t_axis, f0[:, ch0], self.main_window.cur_derivative_tol
                )
                index_grad1 = index_gradient_filter(
                    self.t_axis, f0[:, ch1], self.main_window.cur_derivative_tol
                )
                index = np.intersect1d(np.intersect1d(index, index_grad0), index_grad1)

                f0_plot = self.f0_tmp.copy()
                f0_plot[index] = f2cent(f0[index, ch0], cur_ref_freq) - f2cent(
                    f0[index, ch1], cur_ref_freq
                )

                self._lines[pair_num][0].set_ydata(f0_plot)
                self._lines[pair_num][1].set_ydata(f0_plot)

                pair_num += 1

        if self.cur_disp_pitch_lims != self.main_window.disp_pitch_lims:
            self.cur_disp_pitch_lims = self.main_window.disp_pitch_lims.copy()
            self.set_axes_limits()
            self.bm.update_bg()

        self.bm.update()
