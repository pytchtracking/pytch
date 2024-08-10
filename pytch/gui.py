#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""GUI Functions"""

import logging
import threading
import sys
import time
import numpy as np
import importlib.metadata

from .utils import FloatQLineEdit, QHLine, BlitManager
from .audio import AudioProcessor, get_input_devices, get_fs_options

from PyQt6 import QtCore as qc
from PyQt6 import QtGui as qg
from PyQt6 import QtWidgets as qw
from PyQt6.QtWidgets import QVBoxLayout
from PyQt6.QtWidgets import (
    QFrame,
    QSplitter,
    QHBoxLayout,
    QPushButton,
)

import matplotlib
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.figure import Figure
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

logger = logging.getLogger("pytch.gui")
_refresh_lock = threading.Lock()


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
        b.setCurrentIndex(2)
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
        self.f0_algorithms = ["YIN"]
        self.buf_len_sec = 30.0
        self.disp_pitch_lims = [
            -1500,
            1500,
        ]  # limits in cents for pitch trajectory view
        self.spec_scale_types = ["log", "linear"]
        self.ref_freq_modes = ["fixed", "highest", "lowest"]
        self.disp_t_lvl = 1
        self.disp_t_spec = 1
        self.disp_t_stft = 1
        self.disp_t_f0 = 15
        self.disp_t_conf = 15
        self.lvl_cvals = [-80, -12, 0]
        self.lvl_colors = ["green", "yellow", "red"]
        self.ch_colors = matplotlib.colormaps["Set2"].colors
        self.ch_colors = [self.ch_colors[i] for i in [0, 1, 2, 5, 3, 4, 6, 7]]
        self.cur_disp_freq_lims = [
            20,
            1000,
        ]  # limits in Hz for spectrum/spectrogram view
        self.cur_spec_scale_type = self.spec_scale_types[0]
        self.cur_ref_freq_mode = self.ref_freq_modes[0]
        self.cur_ref_freq = 220
        self.cur_conf_threshold = 0.5
        self.cur_derivative_tol = 600
        self.cur_smoothing_len = 3
        self.last_refresh = time.time()

        # status variables
        self.is_running = False
        self.menu_visible = True

        # styling
        matplotlib.rcParams.update({"font.size": 9})
        pal = self.palette()
        self.setAutoFillBackground(True)
        pal.setColor(qg.QPalette.ColorRole.Window, qg.QColor("white"))
        self.setPalette(pal)

        # initialize and start audio processor
        self.audio_processor = AudioProcessor(
            fs=self.fs,
            buf_len_sec=self.buf_len_sec,
            fft_len=self.fft_size,
            channels=self.channels,
            device_no=self.sounddevice_idx,
            f0_algorithm=self.f0_algorithms[0],
            gui=self,
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
        self.refresh_timer = GUIRefreshTimer()
        self.refresh_timer.refresh_signal.connect(self.refresh_gui)
        self.refresh_timer.start()

        self.play_pause()  # start recording and plotting

    def play_pause(self):
        if self.is_running:
            self.audio_processor.stop_stream()
            self.refresh_timer.stop_emitting()
            self.is_running = False
            self.menu.play_pause_button.setText("Play")
        else:
            self.audio_processor.start_stream()
            self.refresh_timer.start_emitting()
            self.is_running = True
            self.menu.play_pause_button.setText("Pause")

    def refresh_gui(self):
        with _refresh_lock:  # only update when last update has finished
            if self.audio_processor.new_gui_data_available:
                # get preprocessed audio data from audio processor
                lvl, spec, inst_f0, stft, f0, diff = (
                    self.audio_processor.get_latest_gui_data()
                )

                self.channel_views.on_draw(lvl, spec, inst_f0, stft)
                self.trajectory_views.on_draw(f0, diff)
        # logger.info(f"Last refresh finished {time.time() - self.last_refresh}s ago")
        self.last_refresh = time.time()

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
        self.refresh_timer.terminate()
        self.audio_processor.stop_stream()
        self.audio_processor.close_stream()
        sys.exit()


class GUIRefreshTimer(qc.QThread):
    """Timer for GUI refreshes"""

    refresh_signal = qc.pyqtSignal()

    def __init__(self):
        super().__init__()
        self.emit_signal = True

    def run(self):
        while 1:
            time.sleep(1 / 25)  # ideally update with 60 fps
            if self.emit_signal:
                with _refresh_lock:  # make sure last refresh is done
                    self.refresh_signal.emit()

    def stop_emitting(self):
        self.emit_signal = False

    def start_emitting(self):
        self.emit_signal = True


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

        layout.addWidget(qw.QLabel("Median Smoothing [Frames]"), 13, 0)
        self.smoothing_slider = qw.QSlider()
        self.smoothing_slider.setRange(0, 100)
        self.smoothing_slider.setValue(self.main_window.cur_smoothing_len)
        self.smoothing_slider.setOrientation(qc.Qt.Orientation.Horizontal)
        self.smoothing_slider.valueChanged.connect(self.on_conf_smoothing_changed)
        self.smoothing_label = qw.QLabel(f"{self.smoothing_slider.value()}")
        layout.addWidget(self.smoothing_slider, 13, 1, 1, 1)
        layout.addWidget(self.smoothing_label, 13, 2)

        layout.addWidget(qw.QLabel("Pitchslide Tolerance [Cents]"), 14, 0)
        self.derivative_tol_slider = qw.QSlider()
        self.derivative_tol_slider.setRange(0, 1200)
        self.derivative_tol_slider.setValue(main_window.cur_derivative_tol)
        self.derivative_tol_slider.setOrientation(qc.Qt.Orientation.Horizontal)
        self.derivative_tol_label = qw.QLabel(f"{self.derivative_tol_slider.value()}")
        self.derivative_tol_slider.valueChanged.connect(self.on_derivative_tol_changed)
        layout.addWidget(self.derivative_tol_label, 14, 2)
        layout.addWidget(self.derivative_tol_slider, 14, 1, 1, 1)

        layout.addWidget(qw.QLabel("Reference Mode"), 15, 0)
        self.ref_freq_mode_menu = qw.QComboBox()
        if len(self.main_window.channels) > 1:
            self.ref_freq_mode_menu.addItems(main_window.ref_freq_modes)
        else:
            self.ref_freq_mode_menu.addItems([main_window.ref_freq_modes[0]])
        self.ref_freq_mode_menu.setCurrentIndex(0)
        self.ref_freq_mode_menu.currentTextChanged.connect(
            self.on_reference_frequency_mode_changed
        )
        layout.addWidget(self.ref_freq_mode_menu, 15, 1, 1, 1)

        layout.addWidget(qw.QLabel("Reference Frequency"), 16, 0)
        self.freq_box = FloatQLineEdit(parent=self, default=main_window.cur_ref_freq)
        self.freq_box.accepted_value.connect(self.on_reference_frequency_changed)
        layout.addWidget(self.freq_box, 16, 1, 1, 1)
        layout.addWidget(qw.QLabel("Hz"), 16, 2)

        layout.addWidget(qw.QLabel("Minimum Pitch"), 17, 0)
        self.pitch_min = FloatQLineEdit(
            parent=self, default=main_window.disp_pitch_lims[0]
        )
        self.pitch_min.accepted_value.connect(self.on_pitch_min_changed)
        layout.addWidget(self.pitch_min, 17, 1, 1, 1)
        layout.addWidget(qw.QLabel("Cents"), 17, 2)

        layout.addWidget(qw.QLabel("Maximum Pitch"), 18, 0)
        self.pitch_max = FloatQLineEdit(
            parent=self, default=main_window.disp_pitch_lims[1]
        )
        self.pitch_max.accepted_value.connect(self.on_pitch_max_changed)
        layout.addWidget(self.pitch_max, 18, 1, 1, 1)
        layout.addWidget(qw.QLabel("Cents"), 18, 2)

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

    def on_conf_smoothing_changed(self, val):
        self.smoothing_label.setText(str(val))
        self.main_window.cur_smoothing_len = val

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
        self.layout.setSpacing(0)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.views = []
        for ch_id in range(len(main_window.channels)):
            self.views.append(
                ChannelView(
                    main_window=main_window,
                    ch_id=ch_id,
                    is_product=False,
                    has_xlabel=False,
                )
            )

        self.views.append(
            ChannelView(main_window=main_window, is_product=True, has_xlabel=True)
        )

        for i, c_view in enumerate(self.views):
            if i == len(self.views) - 1:
                self.h_line = QHLine()
                self.layout.addWidget(self.h_line, 0)
            self.layout.addWidget(c_view, 0)

        self.setLayout(self.layout)

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
    def on_draw(self, lvl, spec, inst_f0, stft):
        for view in self.views:
            view.on_draw(lvl, spec, inst_f0, stft)

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
        self,
        main_window: MainWindow,
        ch_id=None,
        is_product=False,
        has_xlabel=True,
        *args,
        **kwargs,
    ):
        qw.QWidget.__init__(self, *args, **kwargs)
        self.setLayout(qw.QHBoxLayout())
        self.main_window = main_window

        self.color = "black" if ch_id is None else main_window.ch_colors[ch_id]
        self.is_product = is_product
        self.ch_id = ch_id

        channel_label = "Product" if ch_id is None else f"Channel {ch_id + 1}"
        self.level_widget = LevelWidget(
            self.main_window, channel_label=channel_label, has_xlabel=has_xlabel
        )
        self.spectrogram_widget = SpectrogramWidget(
            self.main_window, self.color, has_xlabel=has_xlabel
        )
        self.spectrum_widget = SpectrumWidget(
            self.main_window, self.color, has_xlabel=has_xlabel
        )

        layout = self.layout()
        layout.addWidget(self.level_widget, 1)
        layout.addWidget(self.spectrum_widget, 8)
        layout.addWidget(self.spectrogram_widget, 8)

        if is_product:
            self.level_widget.ax.xaxis.set_visible(False)
            plt.setp(self.level_widget.ax.spines.values(), visible=False)
            self.level_widget.ax.tick_params(left=False, labelleft=False)
            self.level_widget.ax.patch.set_visible(False)
            self.level_widget.ax.yaxis.grid(False, which="both")

    @qc.pyqtSlot()
    def on_draw(self, lvl, spec, inst_f0, stft):
        if (
            np.all(lvl == 0)
            or np.all(spec == 0)
            or np.all(inst_f0 == 0)
            or np.all(stft == 0)
        ):
            return

        # prepare data
        if self.is_product:
            lvl_update = lvl[:, -1]
            stft_update = stft[:, :, -1]
            spec_update = spec[:, -1]
            inst_f0_update = inst_f0[:, -1]
        else:
            lvl_update = lvl[:, self.ch_id]
            stft_update = stft[:, :, self.ch_id]
            spec_update = spec[:, self.ch_id]
            inst_f0_update = inst_f0[:, self.ch_id]

        # update widgets
        self.level_widget.on_draw(lvl_update.reshape(-1, 1))
        self.spectrum_widget.on_draw(spec_update, inst_f0=inst_f0_update)
        self.spectrogram_widget.on_draw(stft_update)

    def show_spectrum_widget(self, show):
        self.spectrum_widget.setVisible(show)

    def show_spectrogram_widget(self, show):
        self.spectrogram_widget.setVisible(show)

    def show_level_widget(self, show):
        self.level_widget.setVisible(show)


class LevelWidget(FigureCanvas):
    def __init__(self, main_window: MainWindow, channel_label, has_xlabel=True):
        self.figure = Figure(tight_layout=True)
        super(LevelWidget, self).__init__(self.figure)

        self.main_window = main_window

        self.ax = self.figure.add_subplot(111)
        self.ax.set_title(None)
        self.ax.tick_params(axis="x", colors="white")
        self.ax.set_yticks([])
        self.ax.yaxis.grid(True, which="both")
        if has_xlabel:
            self.ax.set_xlabel("Level")
        self.ax.set_ylabel(f"{channel_label}", fontweight="bold")

        norm = plt.Normalize(min(main_window.lvl_cvals), max(main_window.lvl_cvals))
        tuples = list(zip(map(norm, main_window.lvl_cvals), main_window.lvl_colors))
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)
        self.plot_mat_tmp = np.linspace(
            main_window.lvl_cvals[0],
            main_window.lvl_cvals[-1],
            np.abs(main_window.lvl_cvals[0] - main_window.lvl_cvals[-1]),
        ).reshape(-1, 1)
        self.img = self.ax.imshow(
            self.plot_mat_tmp * np.nan,
            cmap=cmap,
            aspect="auto",
            origin="lower",
            extent=[0, 1, main_window.lvl_cvals[0], main_window.lvl_cvals[-1]],
        )
        self.img.set_clim(vmin=main_window.lvl_cvals[0], vmax=main_window.lvl_cvals[-1])
        self.ax.set_ylim((main_window.lvl_cvals[-1], main_window.lvl_cvals[0]))
        self.ax.invert_yaxis()
        self.frame = [self.ax.spines[side] for side in self.ax.spines]
        self.draw()
        self.bm = BlitManager(self.figure.canvas, [self.img] + self.frame)
        self.figure.tight_layout()
        if has_xlabel:
            self.figure.tight_layout(rect=(0.5, 0.15, 1, 1))
        else:
            self.figure.tight_layout(rect=(0.5, 0.1, 1, 1))

    def on_draw(self, plot_mat):
        self.img.set_data(plot_mat)
        self.bm.update()


class SpectrumWidget(FigureCanvas):
    """
    Spectrum widget
    """

    def __init__(self, main_window: MainWindow, color, has_xlabel=True):
        self.figure = Figure(tight_layout=True)
        super(SpectrumWidget, self).__init__(self.figure)

        self.main_window = main_window
        self.color = color
        self.f_axis = self.main_window.audio_processor.fft_freqs
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("")
        if has_xlabel:
            self.ax.set_xlabel("Frequency [Hz]")
        self.ax.set_ylabel("Magnitude")
        self.ax.get_yaxis().set_visible(False)
        self.ax.xaxis.grid(True, which="major")
        self.ax.xaxis.minorticks_on()
        (self._line,) = self.ax.plot(
            self.f_axis, np.zeros_like(self.f_axis), color=self.color
        )
        self._vline = self.ax.axvline(np.nan, c=self.color, linestyle="--")
        self.cur_disp_freq_lims = self.main_window.cur_disp_freq_lims.copy()
        self.ax.set_xlim(self.main_window.cur_disp_freq_lims)
        self.ax.set_ylim((0, 1))
        self.draw()
        self.bm = BlitManager(self.figure.canvas, [self._line, self._vline])
        if has_xlabel:
            self.figure.tight_layout(rect=(0.05, 0.15, 0.95, 1))
        else:
            self.figure.tight_layout(rect=(0.05, 0.1, 0.95, 1))

    def on_draw(self, data_plot, inst_f0=None):
        if self.cur_disp_freq_lims != self.main_window.cur_disp_freq_lims:
            self.ax.set_xlim(self.main_window.cur_disp_freq_lims)
            self.cur_disp_freq_lims = self.main_window.cur_disp_freq_lims.copy()
            self.bm.update_bg()

        self._line.set_ydata(data_plot)
        self._vline.set_xdata([inst_f0, inst_f0])
        self.bm.update()


class SpectrogramWidget(FigureCanvas):
    """
    Spectrogram widget
    """

    def __init__(self, main_window: MainWindow, color, has_xlabel=True):
        self.figure = Figure(tight_layout=True)
        super(SpectrogramWidget, self).__init__(self.figure)

        self.main_window = main_window
        # custom colormap or viridis
        self.cmap = mcolors.LinearSegmentedColormap.from_list(
            "custom_cmap", list(zip([0, 1], ["w", color]))
        )
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("")
        self.ax.set_ylabel("Time")
        self.ax.get_yaxis().set_visible(False)
        self.ax.xaxis.minorticks_on()
        self.default_spec = np.zeros(
            (
                len(self.main_window.audio_processor.fft_freqs),
                int(
                    np.round(
                        self.main_window.audio_processor.frame_rate
                        * self.main_window.disp_t_spec
                    )
                ),
            )
        )
        self.img = self.ax.imshow(
            self.default_spec,
            origin="lower",
            aspect="auto",
            cmap=self.cmap,
            extent=(
                0,
                self.main_window.audio_processor.fs // 2,
                0,
                self.main_window.disp_t_spec,
            ),
        )
        self.ax.xaxis.grid(True, "major")
        self.img.set_clim(vmin=0, vmax=1)
        if has_xlabel:
            self.ax.set_xlabel("Frequency [Hz]")
        self.cur_disp_freq_lims = self.main_window.cur_disp_freq_lims.copy()
        self.ax.set_xlim(self.main_window.cur_disp_freq_lims)
        self.draw()
        self.grid_lines = [line for line in self.ax.get_xgridlines()]
        self.frame = [self.ax.spines[side] for side in self.ax.spines]
        self.bm = BlitManager(
            self.figure.canvas, [self.img] + self.grid_lines + self.frame
        )
        if has_xlabel:
            self.figure.tight_layout(rect=(0.05, 0.15, 0.95, 1))
        else:
            self.figure.tight_layout(rect=(0.05, 0.1, 0.95, 1))

    def on_draw(self, data_plot):
        if self.cur_disp_freq_lims != self.main_window.cur_disp_freq_lims:
            self.ax.set_xlim(self.main_window.cur_disp_freq_lims)
            self.cur_disp_freq_lims = self.main_window.cur_disp_freq_lims.copy()
            self.bm.update_bg()

        self.img.set_data(data_plot)
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

    def on_draw(self, f0, diff):
        self.pitch_view.on_draw(f0)
        if len(self.main_window.channels) > 1:
            self.pitch_view_all_diff.on_draw(diff)

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
        self.t_axis = np.arange(
            int(
                np.round(
                    self.main_window.audio_processor.frame_rate
                    * self.main_window.disp_t_f0
                )
            )
        )
        self.f0_tmp = np.full(len(self.t_axis), np.nan)
        self._lines = []
        for i in range(len(self.channel_views)):
            self._lines.append(
                self.ax.plot(
                    self.t_axis,
                    self.f0_tmp,
                    color=self.channel_views[i].color,
                    linewidth="2",
                )[0]
            )
        self.cur_disp_pitch_lims = self.main_window.disp_pitch_lims.copy()
        self.set_axes_limits()

        pal = self.palette()
        pal.setColor(qg.QPalette.ColorRole.Window, qg.QColor("white"))
        self.setPalette(pal)

        self.draw()

        self.grid_lines = [line for line in self.ax.get_xgridlines()] + [
            line for line in self.ax.get_ygridlines()
        ]

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
            np.arange(
                self.cur_disp_pitch_lims[0], self.cur_disp_pitch_lims[1] + 100, 100
            )
        )

    @qc.pyqtSlot()
    def on_draw(self, f0):
        for i in range(f0.shape[1]):
            self._lines[i].set_ydata(f0[:, i])

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
        self.t_axis = np.arange(
            int(
                np.round(
                    self.main_window.audio_processor.frame_rate
                    * self.main_window.disp_t_f0
                )
            )
        )
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
                    linewidth="2",
                    linestyle="-",
                )

                (line2,) = self.ax.plot(
                    self.t_axis,
                    self.f0_tmp,
                    color=cv1.color,
                    linewidth="2",
                    linestyle=":",
                )

                self._lines.append([line1, line2])

        self.cur_disp_pitch_lims = self.main_window.disp_pitch_lims.copy()
        self.set_axes_limits()

        pal = self.palette()
        pal.setColor(qg.QPalette.ColorRole.Window, qg.QColor("white"))
        self.setPalette(pal)

        self.draw()

        self.grid_lines = [line for line in self.ax.get_xgridlines()] + [
            line for line in self.ax.get_ygridlines()
        ]
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
            np.arange(
                self.cur_disp_pitch_lims[0], self.cur_disp_pitch_lims[1] + 100, 100
            )
        )

    @qc.pyqtSlot()
    def on_draw(self, diff):
        for i in range(diff.shape[1]):
            self._lines[i][0].set_ydata(diff[:, i])
            self._lines[i][1].set_ydata(diff[:, i])

        if self.cur_disp_pitch_lims != self.main_window.disp_pitch_lims:
            self.cur_disp_pitch_lims = self.main_window.disp_pitch_lims.copy()
            self.set_axes_limits()
            self.bm.update_bg()

        self.bm.update()
