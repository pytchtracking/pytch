#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""GUI Functions"""

import logging
import sys
import numpy as np

from .util import consecutive, index_gradient_filter, f2cent
from .menu import DeviceMenu, ProcessingMenu
from .config import _color_names, _colors
from .audio import AudioProcessor

from PyQt6 import QtCore as qc
from PyQt6 import QtGui as qg
from PyQt6 import QtWidgets as qw
from PyQt6.QtWidgets import QVBoxLayout
from PyQt6.QtWidgets import QDockWidget, QFrame, QSizePolicy

import matplotlib
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.figure import Figure
import matplotlib.colors
import matplotlib.pyplot as plt

matplotlib.rcParams.update({"font.size": 9})
colormaps = ["viridis", "wb", "bw"]
logger = logging.getLogger("pytch.gui")
gui_refresh = int(1000 / 25)

audio_processor = AudioProcessor()


class ChannelViews(qw.QWidget):
    """Creates and contains the channel widgets."""

    def __init__(self, channels, freq_max):
        qw.QWidget.__init__(self)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.widget_ready_to_show = False

        self.views = []
        for ch_id in range(channels):
            self.views.append(
                ChannelView(
                    ch_id=ch_id,
                    is_product=False,
                )
            )

        self.views.append(ChannelView(is_product=True))

        for i, c_view in enumerate(self.views):
            if i == len(self.views) - 1:
                self.layout.addWidget(QHLine())
            self.layout.addWidget(c_view)
        self.layout.setContentsMargins(-5, -5, -5, -5)

        self.setSizePolicy(qw.QSizePolicy.Policy.Minimum, qw.QSizePolicy.Policy.Minimum)
        self.show_level_widgets(False)
        self.show_spectrum_widgets(False)
        self.show_spectrogram_widgets(False)

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
    def on_draw(self):
        if not self.widget_ready_to_show:
            self.widget_ready_to_show = True
            self.show_level_widgets(True)
            self.show_spectrum_widgets(True)
            self.show_spectrogram_widgets(True)
            self.show_product_widgets(True)

        for view in self.views:
            view.on_draw()

    def sizeHint(self):
        return qc.QSize(400, 200)

    def __iter__(self):
        yield from self.views


class QHLine(QFrame):
    """
    a horizontal separation line
    """

    def __init__(self):
        super().__init__()
        self.setMinimumWidth(1)
        self.setFixedHeight(20)
        self.setFrameShape(QFrame.Shape.HLine)
        self.setFrameShadow(QFrame.Shadow.Sunken)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)


class ChannelView(qw.QWidget):
    """
    Visual representation of a Channel instance.

    This is a per-channel container. It contains the level-view,
    spectrogram-view and the sepctrum-view of a single channel.
    """

    def __init__(self, ch_id=None, is_product=False, *args, **kwargs):
        qw.QWidget.__init__(self, *args, **kwargs)
        self.setLayout(qw.QHBoxLayout())

        self.color = "black" if ch_id is None else _color_names[3 * ch_id]
        self.is_product = is_product
        self.ch_id = ch_id

        self.confidence_threshold = 0.8  # default

        channel_label = "Product" if ch_id is None else f"Channel {ch_id + 1}"
        self.level_widget = LevelWidget(channel_label=channel_label)
        self.spectrogram_widget = SpectrogramWidget()
        self.spectrum_widget = SpectrumWidget()

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
    def on_draw(self):
        # get latest data
        lvl, stft, f0, conf = audio_processor.read_latest_frames(
            t_lvl=1, t_stft=15, t_f0=1, t_conf=1
        )

        if (
            np.all(lvl == 0)
            or np.all(stft == 0)
            or np.all(f0 == 0)
            or np.all(conf == 0)
        ):
            return

        # prepare data
        if self.is_product:
            lvl_update = np.nan
            stft_update = np.flipud(np.prod(stft, axis=2))
            spec_update = np.mean(stft_update[: lvl.shape[0]], axis=0)
            vline = None
        else:
            lvl_update = np.mean(lvl[:, self.ch_id])
            stft_update = np.flipud(stft[:, :, self.ch_id])
            spec_update = np.mean(stft_update[: lvl.shape[0]], axis=0)
            conf_update = np.mean(conf[:, self.ch_id])
            f0_update = np.mean(f0[:, self.ch_id])

            if conf_update > self.confidence_threshold:
                vline = f0_update
            else:
                vline = None

        # update widgets
        self.level_widget.update_level(lvl_update)
        self.spectrum_widget.update_spectrum(spec_update, self.color, vline=vline)
        self.spectrogram_widget.update_spectrogram(stft_update)

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
        logger.debug("update confidence threshold: %i" % self.confidence_threshold)

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

    def update_level(self, lvl):
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

    def __init__(self):
        super(SpectrumWidget, self).__init__(Figure())

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
        self.freq_min = 20
        self.freq_max = 1000
        self.spectral_type = "log"
        self.figure.tight_layout()

    def update_spectrum(self, data, color, vline=None):
        f_axis = audio_processor.fft_freqs
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

        self.ax.set_yscale(self.spectral_type)
        self.ax.set_xlim((self.freq_min, self.freq_max))
        self.ax.relim()
        self.ax.autoscale(axis="y")
        self.figure.set_tight_layout(True)
        self.draw()

    def set_spectral_type(self, type):
        self.spectral_type = type


class SpectrogramWidget(FigureCanvas):
    """
    Spectrogram widget
    """

    def __init__(self):
        super(SpectrogramWidget, self).__init__(Figure())

        self.figure = Figure(tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("")
        self.ax.set_ylabel(None)
        self.ax.get_yaxis().set_visible(False)
        self.freq_min = 20
        self.freq_max = 1000
        self.show_n_frames = 300
        self.img = None
        self.ax.set_xlim((self.freq_min, self.freq_max))
        self.ax.set_xlabel("Frequency [Hz]")
        self.ax.xaxis.grid(True, which="both")
        self.figure.tight_layout()
        self.spectral_type = "log"

    def update_spectrogram(self, data):
        if self.spectral_type == "log":
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
        self.ax.set_xlim((self.freq_min, self.freq_max))

        self.figure.set_tight_layout(True)
        self.draw()

    def set_spectral_type(self, type):
        self.spectral_type = type


class PitchWidget(FigureCanvas):
    """Pitches of each trace as discrete samples."""

    low_pitch_changed = qc.pyqtSignal(np.ndarray)

    def __init__(self, parent, channel_views, *args, **kwargs):
        super(PitchWidget, self).__init__(Figure())
        self.parent = parent
        self.channel_views = channel_views
        self.current_low_pitch = np.zeros(len(channel_views))
        self.current_low_pitch[:] = np.nan
        self.track_start = None
        self.x_tick_pos = 0

        self.figure = Figure(tight_layout=True)
        self.ax = self.figure.add_subplot(111, position=[0, 0, 0, 0])
        self.ax.set_title(None)
        self.ax.set_ylabel("Relative Pitch [Cents]")
        self.ax.set_xlabel(None)
        self.ax.yaxis.grid(True, which="both")
        self.ax.xaxis.grid(True, which="major")
        self.ax.set_ylim((parent.pitch_min, parent.pitch_max))
        self.ax.set_yticks(
            np.arange(parent.pitch_min, parent.pitch_max + 50, 50), minor=True
        )
        self.ax.set_yticks(np.arange(parent.pitch_min, parent.pitch_max + 100, 100))
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
        self._line = [None] * len(channel_views)
        self.figure.tight_layout()

        self.derivative_filter = 2000  # pitch/seconds
        self.reference_freq = 220  # Hz

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
    def on_draw(self):
        _, _, f0, conf = audio_processor.read_latest_frames(
            t_lvl=0, t_stft=0, t_f0=15, t_conf=15
        )

        if np.all(f0 == 0) or np.all(conf == 0):
            return

        for i, cv in enumerate(self.channel_views):
            # filter f0 using confidence threshold and gradient filter
            index = np.where((conf[:, i] >= cv.confidence_threshold) & (f0[:, i] > 0))[
                0
            ]
            index_grad = index_gradient_filter(
                np.arange(conf.shape[0]), conf[:, i], self.derivative_filter
            )
            index = np.intersect1d(index, index_grad)

            f0_plot = np.full(f0.shape[0], np.nan)
            f0_plot[index] = f0[index, i]
            f0_plot = f2cent(f0_plot, self.reference_freq)

            if self._line[i] is None:
                (self._line[i],) = self.ax.plot(
                    np.arange(len(f0_plot)),
                    f0_plot,
                    color=np.array(_colors[cv.color]) / 256,
                    linewidth="4",
                )
            else:
                self._line[i].set_data(np.arange(len(f0_plot)), f0_plot)

            # try:
            #     self.current_low_pitch[i] = y[indices_grouped[-1][-1]]
            # except IndexError as e:
            #     pass
            #
            # self.low_pitch_changed.emit(self.current_low_pitch)

        self.ax.set_xlim(0, len(f0_plot))
        self.ax.set_xticks(
            np.arange(
                0,
                len(f0_plot),
                int(np.round(audio_processor.fs / audio_processor.hop_len)),
            )
        )
        self.figure.set_tight_layout(True)
        self.draw()


class DifferentialPitchWidget(FigureCanvas):
    """Diffs as line"""

    def __init__(self, parent, channel_views, *args, **kwargs):
        super(DifferentialPitchWidget, self).__init__(Figure())
        self.parent = parent
        self.channel_views = channel_views
        self.track_start = None
        self.tfollow = 3.0

        self.figure = Figure(tight_layout=True)
        self.ax = self.figure.add_subplot(111, position=[0, 0, 0, 0])
        self.ax.set_title(None)
        self.ax.set_ylabel("Difference [Cents]")
        self.ax.set_xlabel(None)
        self.ax.yaxis.grid(True, which="both")
        self.ax.xaxis.grid(True, which="major")
        self.ax.set_ylim((parent.pitch_min, parent.pitch_max))
        self.ax.set_yticks(
            np.arange(parent.pitch_min, parent.pitch_max + 50, 50), minor=True
        )
        self.ax.set_yticks(np.arange(parent.pitch_min, parent.pitch_max + 100, 100))
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
        self._line = [[[None, None]] * len(channel_views)] * len(channel_views)
        self.figure.tight_layout()

        self.derivative_filter = 2000  # pitch/seconds

    @qc.pyqtSlot(int)
    def on_derivative_filter_changed(self, max_derivative):
        self.derivative_filter = max_derivative

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
    def on_draw(self):
        _, _, f0, conf = audio_processor.read_latest_frames(
            t_lvl=0, t_stft=0, t_f0=15, t_conf=15
        )

        if np.all(f0 == 0) or np.all(conf == 0):
            return

        # for i1, cv1 in enumerate(self.channel_views):
        #     x1, y1 = cv1.channel.pitch.latest_frame(tfollow, clip_min=True)
        #     xstart = np.min(x1)
        #     index1 = cv1.channel.latest_confident_indices(
        #         len(x1), cv1.confidence_threshold
        #     )
        #     index1_grad = index_gradient_filter(x1, y1, self.derivative_filter)
        #     index1 = np.intersect1d(index1, index1_grad)
        #     for i2, cv2 in enumerate(self.channel_views):
        #         if i1 >= i2:
        #             continue
        #         x2, y2 = cv2.channel.pitch.latest_frame(tfollow, clip_min=True)
        #         index2_grad = index_gradient_filter(x2, y2, self.derivative_filter)
        #         index2 = cv2.channel.latest_confident_indices(
        #             len(x2), cv2.confidence_threshold
        #         )
        #
        #         index2 = np.intersect1d(index2, index2_grad)
        #         indices = np.intersect1d(index1, index2)
        #         indices_grouped = consecutive(indices)
        #
        #         for group in indices_grouped:
        #             if len(group) == 0:
        #                 continue
        #
        #             y = y1[group] - y2[group]
        #
        #             x = x1[group]
        #             if self._line[i1][i2][0] is None:
        #                 (self._line[i1][i2][0],) = self.ax.plot(
        #                     x,
        #                     y,
        #                     color=np.array(_colors[cv1.color]) / 256,
        #                     linewidth="4",
        #                     linestyle="-",
        #                 )
        #             else:
        #                 if len(x) != 0:
        #                     self._line[i1][i1][0].set_data(x, y)
        #
        #             if self._line[i1][i2][1] is None:
        #                 (self._line[i1][i2][1],) = self.ax.plot(
        #                     x,
        #                     y,
        #                     color=np.array(_colors[cv2.color]) / 256,
        #                     linewidth="4",
        #                     linestyle="--",
        #                 )
        #             else:
        #                 if len(x) != 0:
        #                     self._line[i1][i1][1].set_data(x, y)
        #
        # self.ax.set_xticks(np.arange(0, xstart + self.tfollow, 1))
        # self.ax.set_xlim(xstart, xstart + self.tfollow)
        # self.figure.set_tight_layout(True)
        # self.draw()


class RightTabs(qw.QTabWidget):
    """Widget for right tabs."""

    def __init__(self, *args, **kwargs):
        qw.QTabWidget.__init__(self, *args, **kwargs)
        self.setSizePolicy(
            qw.QSizePolicy.Policy.MinimumExpanding,
            qw.QSizePolicy.Policy.MinimumExpanding,
        )
        self.setAutoFillBackground(True)

        pal = self.palette()
        pal.setColor(qg.QPalette.ColorRole.Window, qg.QColor(*_colors["white"]))
        self.setPalette(pal)

    def sizeHint(self):
        """Makes sure the widget is show with the right aspect."""
        return qc.QSize(500, 200)


class MainWidget(qw.QWidget):
    """Main widget that contains the menu and the visualization widgets."""

    signal_widgets_clear = qc.pyqtSignal()
    signal_widgets_draw = qc.pyqtSignal()

    def __init__(self, *args, **kwargs):
        qw.QWidget.__init__(self, *args, **kwargs)
        self.tabbed_pitch_widget = RightTabs(parent=self)

        pal = self.palette()
        self.setAutoFillBackground(True)
        pal.setColor(qg.QPalette.ColorRole.Window, qg.QColor(*_colors["white"]))
        self.setPalette(pal)

        self.setMouseTracking(True)
        self.top_layout = qw.QGridLayout()
        self.setLayout(self.top_layout)

        self.refresh_timer = qc.QTimer()
        self.refresh_timer.timeout.connect(self.refresh_widgets)
        self.menu = ProcessingMenu()
        self.input_dialog = DeviceMenu(self.audio_config_changed)

        self.data_input = None
        self.freq_max = 1000
        self.pitch_min = -1500
        self.pitch_max = 1500

        qc.QTimer().singleShot(0, self.set_input_dialog)  # show input dialog

    @qc.pyqtSlot(str)
    def on_algorithm_select(self, arg):
        """Change pitch algorithm."""
        for c in self.data_input.channels:
            c.f0_algorithm = arg

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

    def set_input_dialog(self):
        """Query device list and set the drop down menu"""
        audio_processor.stop_stream()
        audio_processor.close_stream()

        self.refresh_timer.stop()
        self.input_dialog.show()
        self.input_dialog.raise_()
        self.input_dialog.activateWindow()

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
        self.pitch_view.low_pitch_changed.connect(self.menu.update_reference_frequency)
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
        self.menu.connect_channel_views(self.channel_views_widget, self.pitch_view)

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


class MainWindow(qw.QMainWindow):
    """Main window that includes the main widget for the menu and all visualizations."""

    def __init__(self):
        super().__init__()

        # main widget inside window
        self.main_widget = MainWidget()
        self.main_widget.setFocusPolicy(qc.Qt.FocusPolicy.StrongFocus)
        self.setCentralWidget(self.main_widget)

        # add dock widget for menu on left side
        menu_dock_widget = QDockWidget()
        menu_dock_widget.setWidget(self.main_widget.menu)
        self.addDockWidget(qc.Qt.DockWidgetArea.LeftDockWidgetArea, menu_dock_widget)

        # add dock widget for trajectory views on right side
        views_dock_widget = QDockWidget()
        views_dock_widget.setWidget(self.main_widget.tabbed_pitch_widget)
        self.addDockWidget(qc.Qt.DockWidgetArea.RightDockWidgetArea, views_dock_widget)

        self.showMaximized()  # maximize window always

    def closeEvent(self, a0):
        audio_processor.stop_stream()
        audio_processor.close_stream()


def start_gui():
    """Starts the GUI"""
    app = qw.QApplication(sys.argv)
    _ = MainWindow()
    app.exec()


# start GUI by executing this file
if __name__ == "__main__":
    start_gui()
