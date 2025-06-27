#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""GUI Functions"""

import logging
import sys
import numpy as np
import importlib.metadata

from .gui_utils import FloatQLineEdit, QHLine, disable_interactivity, colors
from .audio import AudioProcessor, get_input_devices, get_fs_options

from PyQt6 import QtCore as qc
from PyQt6 import QtGui as qg
from PyQt6 import QtWidgets as qw

import pyqtgraph as pg

logger = logging.getLogger("pytch.gui")


def start_gui():
    """Starts the GUI, first show input menu, then open the main GUI"""
    app = qw.QApplication(sys.argv)
    input_dialog = InputMenu()
    if input_dialog.exec() == qw.QDialog.DialogCode.Accepted:
        device, channels, fs, fft_size, out_path = input_dialog.get_input_settings()
        main_window = MainWindow(
            sounddevice_idx=device,
            channels=channels,
            fs=fs,
            fft_size=fft_size,
            out_path=out_path,
        )
        main_window.showMaximized()
        app.exec()
    sys.exit()


class InputMenu(qw.QDialog):
    """Pop up menu at program start that offers user to customise input settings"""

    def __init__(self, *args, **kwargs):
        qw.QDialog.__init__(self, *args, **kwargs)
        self.setModal(True)

        layout = qw.QGridLayout()
        self.setLayout(layout)

        # select input device
        layout.addWidget(qw.QLabel("Input Device:"), 0, 0, 1, 1)
        self.input_options = qw.QComboBox()
        layout.addWidget(self.input_options, 1, 0, 1, 1)
        self.devices = get_input_devices()
        for idevice, device in enumerate(self.devices):
            self.input_options.addItem("{} {}".format(idevice, device[1]["name"]))
        self.input_options.currentIndexChanged.connect(self.update_channel_info)

        # select channels
        layout.addWidget(qw.QLabel("Select Channels:"), 0, 1, 1, 1)
        self.channel_options = qw.QScrollArea()
        self.channel_options.setMaximumSize(30000, 200)
        layout.addWidget(self.channel_options, 1, 1, 6, 1)

        # select sampling rate
        layout.addWidget(qw.QLabel("Sampling Rate:"), 2, 0, 1, 1)
        self.fs_options = qw.QComboBox()
        layout.addWidget(self.fs_options, 3, 0, 1, 1)

        # select fft size
        layout.addWidget(qw.QLabel("FFT Size in Samples:"), 4, 0, 1, 1)
        self.fft_size_options = self.get_nfft_box()
        layout.addWidget(self.fft_size_options, 5, 0, 1, 1)

        # select output directory
        layout.addWidget(qw.QLabel("Output Directory (Optional):"), 6, 0, 1, 1)
        self.out_path = ""
        dir_btn = qw.QPushButton("Browse")
        dir_btn.clicked.connect(self.open_dir_dialog)
        self.dir_name_edit = qw.QLineEdit()
        layout.addWidget(self.dir_name_edit, 7, 0, 1, 1)
        layout.addWidget(dir_btn, 7, 1, 1, 1)

        # OK and Cancel button
        self.buttons = qw.QDialogButtonBox(
            qw.QDialogButtonBox.StandardButton.Ok
            | qw.QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.button(qw.QDialogButtonBox.StandardButton.Ok).setEnabled(False)
        self.buttons.accepted.connect(self.on_ok_clicked)
        self.buttons.rejected.connect(self.close)
        layout.addWidget(self.buttons)

        # load default device
        self.input_options.setCurrentIndex(0)
        self.update_channel_info(0)

    def update_channel_info(self, menu_index):
        """Updates available channels in input menu"""
        sounddevice_index, device = self.devices[menu_index]
        nmax_channels = device["max_input_channels"]

        sampling_rate_options = get_fs_options(sounddevice_index)
        self.channel_selector = ChannelSelector(
            n_channels=nmax_channels, menu_buttons=self.buttons
        )

        self.channel_options.setWidget(self.channel_selector)
        self.fs_options.clear()
        self.fs_options.addItems([str(int(v)) for v in sampling_rate_options])
        self.buttons.button(qw.QDialogButtonBox.StandardButton.Ok).setEnabled(False)

    @staticmethod
    def get_nfft_box():
        """Return a qw.QSlider for modifying FFT width"""
        b = qw.QComboBox()
        b.addItems([str(f * 256) for f in [1, 2, 4, 8, 16]])
        b.setCurrentIndex(2)
        return b

    def open_dir_dialog(self):
        """Opens an os dialogue for selecting a directory"""
        dir_name = qw.QFileDialog.getExistingDirectory(self, "Select a Directory")
        if dir_name:
            self.out_path = str(dir_name)
            self.dir_name_edit.setText(self.out_path)

    def on_ok_clicked(self):
        """Close the menu when the user clicks ok and signal that main GUI can be opened"""
        self.accept()

    def get_input_settings(self):
        """Returns user-configured input settings"""
        sounddevice_idx = self.devices[self.input_options.currentIndex()][0]
        channels = self.channel_selector.get_selected_channels()
        fs = int(self.fs_options.currentText())
        fft_size = int(self.fft_size_options.currentText())
        return sounddevice_idx, channels, fs, fft_size, self.out_path


class ChannelSelector(qw.QWidget):
    """Widget for the channel buttons on the right side of the input menu"""

    def __init__(self, n_channels, menu_buttons):
        super().__init__()
        self.setLayout(qw.QVBoxLayout())

        self.menu_buttons = menu_buttons
        self.ch_buttons = []
        self.press_order = []
        for i in range(n_channels):
            button = qw.QPushButton("Channel %i" % (i + 1))
            button.setCheckable(True)
            button.clicked.connect(
                lambda checked, index=i: self.track_button_press(index)
            )
            self.ch_buttons.append(button)
            self.layout().addWidget(button)

    def get_selected_channels(self):
        """Returns selected channels by the user in order"""
        return self.press_order

    def track_button_press(self, index):
        """Tracks the order of button presses"""
        if index in self.press_order:
            self.press_order.remove(index)
        else:
            self.press_order.append(index)

        self.menu_buttons.button(qw.QDialogButtonBox.StandardButton.Ok).setEnabled(
            len(self.press_order) > 0
        )


class MainWindow(qw.QMainWindow):
    """Main window that includes the main widget for the menu and all visualizations."""

    def __init__(self, sounddevice_idx, channels, fs, fft_size, out_path):
        super().__init__()

        # default settings for the entire GUI.
        self.version = importlib.metadata.version("pytch")
        self.sounddevice_idx = sounddevice_idx
        self.channels = channels
        self.fs = fs
        self.fft_size = fft_size
        self.out_path = out_path
        self.f0_algorithms = ["YIN"]
        self.buf_len_sec = 30.0
        self.spec_scale_types = ["log", "linear"]
        self.ref_freq_modes = ["fixed", "highest", "lowest"]
        self.disp_t_lvl = 1
        self.disp_t_spec = 1
        self.disp_t_stft = 5
        self.disp_t_f0 = 10
        self.disp_t_conf = 10
        self.lvl_cvals = [-80, -12, 0]
        self.lvl_colors = ["green", "yellow", "red"]
        self.ch_colors = colors
        self.cur_disp_freq_lims = [
            20,
            1000,
        ]  # limits in Hz for spectrum/spectrogram view
        self.cur_disp_pitch_lims = [
            -1500,
            1500,
        ]  # limits in cents for pitch trajectory view
        self.cur_spec_scale_type = self.spec_scale_types[0]
        self.cur_ref_freq_mode = self.ref_freq_modes[0]
        self.cur_ref_freq = 220
        self.cur_conf_threshold = 0.5
        self.cur_derivative_tol = 600
        self.cur_smoothing_len = 3
        self.gui_refresh_ms = int(np.round(1000 / 60))  # 60 fps

        # status variables
        self.is_running = False
        self.menu_visible = True

        # styling
        pal = self.palette()
        self.setAutoFillBackground(True)
        pal.setColor(qg.QPalette.ColorRole.Window, qg.QColor("white"))
        self.setPalette(pal)

        pg.setConfigOption("background", "w")  # Set background to white
        pg.setConfigOption("foreground", "k")  # Set foreground to black (text)
        pg.mkPen("k")  # Set pen color to black
        self.line_width = 4

        # initialize and start audio processor
        self.audio_processor = AudioProcessor(
            fs=self.fs,
            buf_len_sec=self.buf_len_sec,
            fft_len=self.fft_size,
            channels=self.channels,
            device_no=self.sounddevice_idx,
            f0_algorithm=self.f0_algorithms[0],
            out_path=out_path,
        )

        # initialize GUI
        self.setWindowTitle(f"Pytch {self.version}")
        central_widget = qw.QWidget()  # contains all contents
        self.setCentralWidget(central_widget)

        splitter = qw.QSplitter(
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

        layout = qw.QVBoxLayout(central_widget)  # sets the layout of the central widget
        layout.addLayout(self.menu_toggle_button())  # top bar with menu toggle button
        layout.addWidget(splitter)

        # refresh timer
        self.refresh_timer = qc.QTimer()
        self.refresh_timer.timeout.connect(self.refresh_gui)
        self.refresh_timer.start(self.gui_refresh_ms)

        self.play_pause()  # start recording and plotting

    def play_pause(self):
        """Starts or stops the GUI"""
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

    @qc.pyqtSlot()
    def refresh_gui(self):
        """GUI refresh function, needs to be as fast as possible"""

        # get preprocessed audio data from audio processor
        lvl, spec, inst_f0, stft, f0, diff = self.audio_processor.get_gui_data(
            disp_t_lvl=self.disp_t_lvl,
            disp_t_spec=self.disp_t_spec,
            disp_t_stft=self.disp_t_stft,
            disp_t_f0=self.disp_t_f0,
            disp_t_conf=self.disp_t_conf,
            lvl_cvals=self.lvl_cvals,
            cur_spec_scale_type=self.cur_spec_scale_type,
            cur_smoothing_len=self.cur_smoothing_len,
            cur_conf_threshold=self.cur_conf_threshold,
            cur_ref_freq_mode=self.cur_ref_freq_mode,
            cur_ref_freq=self.cur_ref_freq,
            cur_derivative_tol=self.cur_derivative_tol,
        )

        # update widgets
        self.channel_views.on_draw(lvl, spec, inst_f0, stft)
        self.trajectory_views.on_draw(f0, diff)

    def menu_toggle_button(self):
        """The button for toggeling the menu"""
        top_bar = qw.QHBoxLayout()
        top_bar.setContentsMargins(0, 0, 0, 0)
        top_bar.setSpacing(0)
        self.toggle_button = qw.QPushButton("☰ Hide Menu")
        self.toggle_button.setFixedSize(100, 10)
        self.toggle_button.clicked.connect(self.toggle_menu)
        self.toggle_button.setStyleSheet("border :none")
        self.toggle_button.adjustSize()
        self.toggle_button.isFlat()
        top_bar.addWidget(self.toggle_button)
        top_bar.addStretch()
        return top_bar

    def toggle_menu(self):
        """Make menu appear or disappear"""
        if self.menu_visible:
            self.menu.hide()
            self.toggle_button.setText("☰ Show Menu")
        else:
            self.menu.show()
            self.toggle_button.setText("☰ Hide Menu")
        self.menu_visible = not self.menu_visible

    def closeEvent(self, a0):
        """Clean up when GUI is closed"""
        self.refresh_timer.stop()
        self.audio_processor.stop_stream()
        self.audio_processor.close_stream()
        sys.exit()


class ProcessingMenu(qw.QFrame):
    """The processing menu on the left side of the main window"""

    def __init__(self, main_window: MainWindow, *args, **kwargs):
        qw.QFrame.__init__(self, *args, **kwargs)

        self.main_window = main_window

        # main menu layout
        main_layout = qw.QGridLayout()
        self.setLayout(main_layout)

        # play/pause button
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
            parent=self, default=main_window.cur_disp_pitch_lims[0]
        )
        self.pitch_min.accepted_value.connect(self.on_pitch_min_changed)
        layout.addWidget(self.pitch_min, 17, 1, 1, 1)
        layout.addWidget(qw.QLabel("Cents"), 17, 2)

        layout.addWidget(qw.QLabel("Maximum Pitch"), 18, 0)
        self.pitch_max = FloatQLineEdit(
            parent=self, default=main_window.cur_disp_pitch_lims[1]
        )
        self.pitch_max.accepted_value.connect(self.on_pitch_max_changed)
        layout.addWidget(self.pitch_max, 18, 1, 1, 1)
        layout.addWidget(qw.QLabel("Cents"), 18, 2)

        settings.setLayout(layout)
        main_layout.addWidget(settings, 3, 0, 1, 2)

    def on_min_freq_changed(self, f):
        self.main_window.cur_disp_freq_lims[0] = int(f)
        self.main_window.channel_views.on_disp_freq_lims_changed(
            self.main_window.cur_disp_freq_lims
        )

    def on_max_freq_changed(self, f):
        self.main_window.cur_disp_freq_lims[1] = int(f)
        self.main_window.channel_views.on_disp_freq_lims_changed(
            self.main_window.cur_disp_freq_lims
        )

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
        self.main_window.cur_disp_pitch_lims[0] = int(val)
        self.main_window.trajectory_views.on_disp_pitch_lims_changed(
            self.main_window.cur_disp_pitch_lims
        )

    def on_pitch_max_changed(self, val):
        self.main_window.cur_disp_pitch_lims[1] = int(val)
        self.main_window.trajectory_views.on_disp_pitch_lims_changed(
            self.main_window.cur_disp_pitch_lims
        )

    def on_spectrum_type_select(self, arg):
        self.main_window.cur_spec_scale_type = arg

    def sizeHint(self):
        return qc.QSize(100, 200)


class ChannelViews(qw.QWidget):
    """The central widget of the GUI that contains all channel views"""

    refresh_signal = qc.pyqtSignal(np.ndarray, np.ndarray, np.ndarray, np.ndarray)

    def __init__(self, main_window: MainWindow):
        qw.QWidget.__init__(self)
        self.layout = qw.QVBoxLayout()
        self.layout.setSpacing(0)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.views = []
        for ch_id, orig_ch in enumerate(main_window.channels):
            self.views.append(
                ChannelView(
                    main_window=main_window,
                    ch_id=ch_id,
                    orig_ch=orig_ch + 1,
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
                self.layout.addWidget(self.h_line, 1)
            self.layout.addWidget(c_view, 10)
            self.refresh_signal.connect(c_view.on_draw)

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

    def on_disp_freq_lims_changed(self, disp_freq_lims):
        for view in self.views:
            view.on_disp_freq_lims_changed(disp_freq_lims)

    @qc.pyqtSlot()
    def on_draw(self, lvl, spec, inst_f0, stft):
        self.refresh_signal.emit(lvl, spec, inst_f0, stft)

    def sizeHint(self):
        return qc.QSize(400, 200)

    def __iter__(self):
        yield from self.views


class ChannelLabel(qw.QWidget):
    """Widget that contains the vertical channel label"""

    def __init__(self, text):
        super().__init__()
        self.text = text

    def paintEvent(self, event):
        """Paints the label and updates it when necessary, e.g. when available space changes"""
        painter = qg.QPainter(self)
        painter.setFont(qg.QFont("Arial", 13, qg.QFont.Weight.Bold))
        painter.setPen(qg.QColor("black"))
        font_metrics = painter.fontMetrics()
        text_rect = font_metrics.boundingRect(self.text)  # calculate text size
        painter.translate(
            20, (self.rect().height() + text_rect.width()) // 2
        )  # vertical centering
        painter.rotate(-90)
        painter.drawText(0, 0, self.text)
        painter.end()


class ChannelView(qw.QWidget):
    """Widget that contains a channel label, level, spectrum and spectrogram,
    a.k.a. one row of the central GUI widget
    """

    level_refresh_signal = qc.pyqtSignal(float)
    spectrum_refresh_signal = qc.pyqtSignal(np.ndarray, float)
    spectrogram_refresh_signal = qc.pyqtSignal(np.ndarray)

    def __init__(
        self,
        main_window: MainWindow,
        ch_id=None,
        orig_ch=None,
        is_product=False,
        has_xlabel=True,
        *args,
        **kwargs,
    ):
        qw.QWidget.__init__(self, *args, **kwargs)
        self.layout = qw.QHBoxLayout()
        self.layout.setSpacing(0)  # keep GUI tight, remove frames around widgets
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.main_window = main_window

        self.color = "black" if ch_id is None else main_window.ch_colors[ch_id]
        self.is_product = is_product
        self.ch_id = ch_id

        # channel label
        label = ChannelLabel("Product" if ch_id is None else f"Channel {orig_ch}")

        self.level_widget = LevelWidget(self.main_window, has_xlabel=has_xlabel)
        self.spectrogram_widget = SpectrogramWidget(
            self.main_window, self.color, has_xlabel=has_xlabel
        )
        self.spectrum_widget = SpectrumWidget(
            self.main_window, self.color, has_xlabel=has_xlabel
        )

        self.level_refresh_signal.connect(self.level_widget.on_draw)
        self.spectrum_refresh_signal.connect(self.spectrum_widget.on_draw)
        self.spectrogram_refresh_signal.connect(self.spectrogram_widget.on_draw)

        self.layout.addWidget(label, 1)
        self.layout.addWidget(self.level_widget, 1)
        self.layout.addWidget(self.spectrum_widget, 10)
        self.layout.addWidget(self.spectrogram_widget, 10)

        if is_product:
            self.level_widget.clear()
            self.level_widget.setBackground(None)

        self.setLayout(self.layout)

    @qc.pyqtSlot(object, object, object, object)
    def on_draw(self, lvl, spec, inst_f0, stft):
        """Refreshes all widgets as fast as possible"""
        idx = -1 if self.is_product else self.ch_id

        if len(lvl) > 0 and not self.is_product:
            self.level_refresh_signal.emit(lvl[idx])

        if len(spec) > 0:
            self.spectrum_refresh_signal.emit(spec[:, idx], inst_f0[idx])

        if len(stft) > 0:
            self.spectrogram_refresh_signal.emit(stft[:, :, idx])

    def show_spectrum_widget(self, show):
        self.spectrum_widget.setVisible(show)

    def show_spectrogram_widget(self, show):
        self.spectrogram_widget.setVisible(show)

    def show_level_widget(self, show):
        self.level_widget.setVisible(show)

    def on_disp_freq_lims_changed(self, disp_freq_lims):
        self.spectrum_widget.on_disp_freq_lims_changed(disp_freq_lims)
        self.spectrogram_widget.on_disp_freq_lims_changed(disp_freq_lims)


class LevelWidget(pg.GraphicsLayoutWidget):
    """The level meter with color-coded dB levels"""

    def __init__(self, main_window: MainWindow, has_xlabel=True):
        super(LevelWidget, self).__init__()

        self.main_window = main_window

        # Create the plot item
        self.plot_item = self.addPlot()
        self.plot_item.setDefaultPadding(0)
        self.plot_item.getAxis("left").setVisible(True)
        self.plot_item.getAxis("left").setTicks([])
        self.plot_item.getAxis("right").setVisible(True)
        self.plot_item.getAxis("right").setTicks([])
        self.plot_item.getAxis("top").setVisible(True)
        self.plot_item.getAxis("top").setTicks([])
        self.plot_item.getAxis("bottom").setTicks([[(0, ""), (1, "invisible")]])
        self.plot_item.showGrid(x=False, y=False)
        if has_xlabel:
            self.plot_item.setLabel("bottom", "Level")

        # Initialize the image item
        self.img = pg.ImageItem()
        self.plot_item.addItem(self.img)

        # Set the colormap
        self.lvl_converter = lambda x: (x - self.main_window.lvl_cvals[0]) / np.abs(
            self.main_window.lvl_cvals[0]
        )
        color_map = pg.ColorMap(
            pos=self.lvl_converter(np.array(main_window.lvl_cvals)),
            color=main_window.lvl_colors,
        )
        self.img.setLevels([0, 1])
        self.img.setLookupTable(color_map.getLookupTable(nPts=256, alpha=False))

        # Set up initial empty data
        self.img.setImage(np.zeros((1, 1)))
        self.plot_item.setXRange(0, 1)
        self.plot_item.setYRange(0, np.abs(main_window.lvl_cvals[0]))

        disable_interactivity(self.plot_item)

    @qc.pyqtSlot(float)
    def on_draw(self, lvl):
        """Updates the image with new data."""
        lvl_conv = self.lvl_converter(lvl)
        plot_array = np.linspace(
            0, lvl_conv, int(lvl_conv * np.abs(self.main_window.lvl_cvals[0]))
        ).reshape(1, -1)

        if not plot_array.size:
            plot_array = np.zeros((1, 1))

        self.img.setImage(plot_array)
        self.img.setLevels([0, 1])


class SpectrumWidget(pg.GraphicsLayoutWidget):
    """Spectrum plot with current fundamental frequency as dashed line"""

    def __init__(self, main_window: MainWindow, color, has_xlabel=True):
        super(SpectrumWidget, self).__init__()

        self.main_window = main_window
        self.color = color
        self.f_axis = self.main_window.audio_processor.fft_freqs

        # Create the plot item
        self.plot_item = self.addPlot()
        if has_xlabel:
            self.plot_item.setLabel("bottom", "Frequency [Hz]")
        self.plot_item.getAxis("left").setVisible(True)
        self.plot_item.getAxis("left").setTicks([])
        self.plot_item.getAxis("right").setVisible(True)
        self.plot_item.getAxis("right").setTicks([])
        self.plot_item.getAxis("top").setVisible(True)
        self.plot_item.getAxis("top").setTicks([])
        self.plot_item.setDefaultPadding(0)
        self.plot_item.showGrid(x=True, y=True)
        self.plot_item.setXRange(*self.main_window.cur_disp_freq_lims)
        self.plot_item.setYRange(0, 1)

        # Create the line for the spectrum
        self._line = pg.PlotDataItem(
            self.f_axis,
            np.zeros_like(self.f_axis),
            pen=pg.mkPen(
                self.color,
                width=main_window.line_width,
                style=pg.QtCore.Qt.PenStyle.SolidLine,
            ),
        )
        self.plot_item.addItem(self._line)

        # Create the line for the fundamental frequency
        self._inst_f0_line = pg.InfiniteLine(
            angle=90,
            pen=pg.mkPen(
                self.color,
                style=pg.QtCore.Qt.PenStyle.DashLine,
                width=main_window.line_width,
            ),
        )
        self.plot_item.addItem(self._inst_f0_line)

        disable_interactivity(self.plot_item)

    def on_disp_freq_lims_changed(self, disp_freq_lims):
        self.plot_item.setXRange(*disp_freq_lims)

    @qc.pyqtSlot(object, float)
    def on_draw(self, data_plot, inst_f0=None):
        """Updates the spectrum and the fundamental frequency line."""
        self._line.setData(x=self.f_axis, y=data_plot)  # Update the spectrum line

        if inst_f0 is not None:
            self._inst_f0_line.setPos(inst_f0)  # Update the fundamental frequency line


class SpectrogramWidget(pg.GraphicsLayoutWidget):
    """Spectrogram widget"""

    def __init__(self, main_window: MainWindow, color, has_xlabel=True):
        super().__init__()

        self.main_window = main_window

        # Initialize the plot
        self.plot_item = self.addPlot()
        if has_xlabel:
            self.plot_item.setLabel("bottom", "Frequency [Hz]")
        self.plot_item.getAxis("left").setVisible(True)
        self.plot_item.getAxis("left").setTicks([])
        self.plot_item.getAxis("right").setVisible(True)
        self.plot_item.getAxis("right").setTicks([])
        self.plot_item.getAxis("top").setVisible(True)
        self.plot_item.getAxis("top").setTicks([])
        self.plot_item.showGrid(x=True, y=True)
        self.plot_item.setDefaultPadding(0)

        # Create an ImageItem to display the spectrogram
        self.img = pg.ImageItem()
        self.plot_item.addItem(self.img)

        # Initialize the default spectrogram data
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
        ).T
        self.img.setImage(self.default_spec)
        self.img.setRect(
            qc.QRectF(
                0,
                0,
                self.main_window.audio_processor.fft_freqs[-1],
                self.default_spec.shape[0],
            )
        )
        self.plot_item.setXRange(*self.main_window.cur_disp_freq_lims)

        # Apply custom colormap to ImageItem
        self.cmap = pg.ColorMap(pos=[0, 1], color=[(255, 255, 255), color])
        lut = self.cmap.getLookupTable(nPts=256)
        self.img.setLookupTable(lut)
        self.img.setLevels([0, 1])

        disable_interactivity(self.plot_item)

    def on_disp_freq_lims_changed(self, disp_freq_lims):
        self.plot_item.setXRange(*disp_freq_lims)

    @qc.pyqtSlot(object)
    def on_draw(self, data_plot):
        """Updates the spectrogram with new data."""
        self.img.setImage(data_plot.T, autoLevels=False)
        self.img.setRect(
            qc.QRectF(
                0,
                0,
                self.main_window.audio_processor.fft_freqs[-1],
                self.default_spec.shape[0],
            )
        )


class TrajectoryViews(qw.QTabWidget):
    """Right-hand widget that contains the visualization of the F0-trajectories and the differential"""

    def __init__(self, main_window: MainWindow, *args, **kwargs):
        qw.QTabWidget.__init__(self, *args, **kwargs)

        self.main_window = main_window

        # styling
        self.layout = qw.QVBoxLayout()
        self.layout.setSpacing(0)
        self.layout.setContentsMargins(0, 0, 0, 0)
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
        self.pitch_diff_view = DifferentialPitchWidget(main_window)

        # remove old tabs from pitch view
        self.addTab(self.pitch_view, "Pitches")
        if (
            len(main_window.channels) > 1
        ):  # can't compute differential with only one voice
            self.addTab(self.pitch_diff_view, "Differential")

        self.on_disp_pitch_lims_changed(self.main_window.cur_disp_pitch_lims)
        self.setLayout(self.layout)

    @qc.pyqtSlot(object, object)
    def on_draw(self, f0, diff):
        if len(f0) > 0:
            self.pitch_view.on_draw(f0)

        if len(self.main_window.channels) > 1 and len(diff) > 0:
            self.pitch_diff_view.on_draw(diff)

    def on_disp_pitch_lims_changed(self, disp_pitch_lims):
        self.change_pitch_lims(self.pitch_view, disp_pitch_lims)
        if len(self.main_window.channels) > 1:
            self.change_pitch_lims(self.pitch_diff_view, disp_pitch_lims)

    @staticmethod
    def change_pitch_lims(view, disp_pitch_lims):
        # Set the x-axis range
        view.plot_item.setXRange(0, len(view.t_axis))

        # Set y-axis range
        view.plot_item.setYRange(disp_pitch_lims[0], disp_pitch_lims[1])

        # Set y-ticks with labels
        y_ticks = np.arange(disp_pitch_lims[0], disp_pitch_lims[1] + 100, 100)
        view.plot_item.getAxis("left").setTicks([[(y, str(y)) for y in y_ticks]])

    def show_trajectory_views(self, show):
        self.setVisible(show)

    def sizeHint(self):
        """Makes sure the widget is show with the right aspect."""
        return qc.QSize(500, 800)


class PitchWidget(pg.GraphicsLayoutWidget):
    """Visualization of the F0-trajectories of each channel"""

    low_pitch_changed = qc.pyqtSignal(np.ndarray)

    def __init__(self, main_window: MainWindow, *args, **kwargs):
        super(PitchWidget, self).__init__(*args, **kwargs)

        self.main_window = main_window
        self.channel_views = main_window.channel_views.views[:-1]
        self.ci.layout.setContentsMargins(0, 0, 0, 0)
        self.ci.layout.setSpacing(0)

        # Create the plot item
        self.plot_item = self.addPlot()
        self.plot_item.setLabel("left", "Relative Pitch [Cents]")
        self.plot_item.setLabel("bottom", "Time")
        self.plot_item.getAxis("right").setVisible(True)
        self.plot_item.getAxis("right").setTicks([])
        self.plot_item.getAxis("top").setVisible(True)
        self.plot_item.getAxis("top").setTicks([])
        self.plot_item.showGrid(x=True, y=True)
        self.plot_item.setDefaultPadding(0)
        self.plot_item.setYRange(*main_window.cur_disp_pitch_lims)

        self.t_axis = np.arange(
            int(
                np.round(
                    self.main_window.audio_processor.frame_rate
                    * self.main_window.disp_t_f0
                )
            )
        )
        self.f0_tmp = np.full(len(self.t_axis), np.nan)

        # Create lines for each channel
        self._lines = []
        for channel_view in self.channel_views:
            line = pg.PlotDataItem(
                self.t_axis,
                self.f0_tmp,
                pen=pg.mkPen(channel_view.color, width=main_window.line_width + 2),
            )
            self.plot_item.addItem(line)
            self._lines.append(line)

        # Set empty x-ticks
        x_ticks = np.arange(0, len(self.t_axis), main_window.audio_processor.frame_rate)
        self.plot_item.getAxis("bottom").setTicks([[(x, "") for x in x_ticks]])

        disable_interactivity(self.plot_item)

    @qc.pyqtSlot(object)
    def on_draw(self, f0):
        """Updates the F0 trajectories for each channel."""
        if len(f0) > 0:
            for i in range(f0.shape[1]):
                self._lines[i].setData(self.t_axis, f0[:, i])  # Update the line data


class DifferentialPitchWidget(pg.GraphicsLayoutWidget):
    """Visualization of the pair-wise F0-differences"""

    def __init__(self, main_window: MainWindow, *args, **kwargs):
        super(DifferentialPitchWidget, self).__init__(*args, **kwargs)
        self.main_window = main_window
        self.channel_views = main_window.channel_views.views[:-1]
        self.ci.layout.setContentsMargins(0, 0, 0, 0)
        self.ci.layout.setSpacing(0)

        # Create the plot item
        self.plot_item = self.addPlot()
        self.plot_item.setLabel("left", "Pitch Difference [Cents]")
        self.plot_item.setLabel("bottom", "Time")
        self.plot_item.getAxis("right").setVisible(True)
        self.plot_item.getAxis("right").setTicks([])
        self.plot_item.getAxis("top").setVisible(True)
        self.plot_item.getAxis("top").setTicks([])
        self.plot_item.showGrid(x=True, y=True)
        self.plot_item.setDefaultPadding(0)
        self.plot_item.setYRange(*main_window.cur_disp_pitch_lims)

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

                # Create two lines for alternating colors
                line1 = pg.PlotDataItem(
                    self.t_axis,
                    self.f0_tmp,
                    pen=pg.mkPen(
                        cv0.color,
                        width=main_window.line_width + 2,
                        style=pg.QtCore.Qt.PenStyle.SolidLine,
                    ),
                )
                line2 = pg.PlotDataItem(
                    self.t_axis,
                    self.f0_tmp,
                    pen=pg.mkPen(
                        cv1.color,
                        width=main_window.line_width + 2,
                        style=pg.QtCore.Qt.PenStyle.DashLine,
                    ),
                )

                self.plot_item.addItem(line1)
                self.plot_item.addItem(line2)
                self._lines.append([line1, line2])

        # Set empty x-ticks
        x_ticks = np.arange(0, len(self.t_axis), main_window.audio_processor.frame_rate)
        self.plot_item.getAxis("bottom").setTicks([[(x, "") for x in x_ticks]])

        disable_interactivity(self.plot_item)

    @qc.pyqtSlot(object)
    def on_draw(self, diff):
        """Updates the pitch differences for each channel pair."""
        if len(diff) > 0:
            for i in range(diff.shape[1]):
                self._lines[i][0].setData(
                    self.t_axis, diff[:, i]
                )  # Update the solid line
                self._lines[i][1].setData(
                    self.t_axis, diff[:, i]
                )  # Update the dashed line
