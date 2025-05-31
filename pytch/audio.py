#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Audio Functions"""
import threading
from time import sleep, time
import numpy as np
from numba import njit
import logging
import sounddevice
import soundfile as sf
import libf0
from scipy.ndimage import median_filter
from datetime import datetime
import csv

_audio_lock = threading.Lock()  # lock for raw audio buffer
_feat_lock = threading.Lock()  # lock for feature buffers
_gui_lock = threading.Lock()  # lock for communication with GUI
logger = logging.getLogger("pytch.audio")

eps = np.finfo(float).eps


def get_input_devices():
    """Returns a list of devices."""
    input_devices = []
    for device_id, device in enumerate(sounddevice.query_devices()):
        if device["max_input_channels"] > 0:
            input_devices.append((device_id, device))
    return input_devices


def get_fs_options(device_idx):
    """Returns a dictionary of supported sampling rates for all devices."""
    candidates = [8000.0, 11025.0, 16000.0, 22050.0, 32000.0, 37800.0, 44100.0, 48000.0]
    supported_fs = []
    for c in candidates:
        if check_fs(device_idx, int(c)):
            supported_fs.append(c)

    return supported_fs


def check_fs(device_index, fs):
    """Validates chosen sampling rate."""
    valid = True
    try:
        sounddevice.check_input_settings(
            device=device_index,
            channels=None,
            dtype=None,
            extra_settings=None,
            samplerate=fs,
        )
    except ValueError as e:
        logger.debug(e)
        valid = False

    finally:
        return valid


@njit
def f2cent(f, standard_frequency=440.0):
    """Convert from Hz to Cents"""
    return 1200.0 * np.log2(np.abs(f) / standard_frequency + eps)


@njit
def gradient_filter(y, max_gradient):
    """Get index where the abs gradient of x, y is < max_gradient."""
    return np.where(np.abs(np.diff(f2cent(y))) < max_gradient)[0]


class RingBuffer:
    """Generic ring buffer for n-dimensional data"""

    def __init__(self, size, dtype):
        """Initialize buffer, size should be of format (n_frames, ..., n_channels)"""
        self.size = size
        self.buffer = np.zeros(size, dtype=dtype)
        self.write_head = 0
        self.read_head = 0

    def write(self, data):
        """Writes data to buffer"""
        if data.shape[0] > self.size[0]:
            logger.warning("Buffer overflow!")
        write_idcs = np.mod(self.write_head + np.arange(data.shape[0]), self.size[0])
        self.buffer[write_idcs, ...] = data
        self.write_head = np.mod(
            write_idcs[-1] + 1, self.size[0]
        )  # set write head to the next bin to write to

    def read_latest(self, n_frames):
        """Reads n_frames from buffer, starting from latest data"""
        if self.size[0] < n_frames:
            Exception("Cannot read more data than buffer length!")

        read_idcs = np.mod(
            self.size[0] + self.write_head - np.arange(n_frames) - 1, self.size[0]
        )[::-1]
        return self.buffer[read_idcs, ...]

    def read_next(self, n_frames, hop_frames=None):
        """Reads n_frames from buffer, starting from latest read"""

        if (
            np.mod(self.size[0] + self.write_head - self.read_head, self.size[0])
            < n_frames
        ):
            return np.array([])

        read_idcs = np.mod(
            self.size[0] + self.read_head + np.arange(n_frames), self.size[0]
        )[::-1]

        if hop_frames is None:
            hop_frames = n_frames

        self.read_head = np.mod(
            self.read_head + hop_frames, self.size[0]
        )  # advance read head

        return self.buffer[read_idcs, ...]

    def flush(self):
        self.buffer = np.zeros_like(self.buffer)
        self.write_head = 0
        self.read_head = 0


class AudioProcessor:
    """Class for recording and processing of multichannel audio"""

    def __init__(
        self,
        fs=8000,
        buf_len_sec=30.0,
        fft_len=512,
        channels=None,
        device_no=None,
        f0_algorithm="YIN",
        gui=None,
        out_path="",
    ):
        self.fs = fs
        self.buf_len_sec = buf_len_sec
        self.fft_len = fft_len
        self.hop_len = 2 ** int(np.log2(fs / 25))
        self.fft_freqs = np.fft.rfftfreq(self.fft_len, 1 / self.fs)
        self.fft_win = np.hanning(self.fft_len).reshape(-1, 1)
        self.channels = [0] if channels is None else channels
        self.device_no = device_no
        self.f0_algorithm = f0_algorithm
        self.out_path = out_path
        self.gui = gui
        self.f0_lvl_threshold = -70  # minimum level in dB to compute f0 estimates
        self.frame_rate = self.fs / self.hop_len
        self.stream = None
        self.is_running = False

        # initialize buffers
        buf_len_smp = int(
            np.ceil(self.buf_len_sec * self.fs / self.hop_len) * self.hop_len
        )
        self.audio_buf = RingBuffer(
            size=(buf_len_smp, len(self.channels)), dtype=np.float32
        )

        buf_len_frm = int(
            np.floor((self.buf_len_sec * self.fs - self.fft_len) / self.hop_len)
        )
        self.raw_lvl_buf = RingBuffer(
            size=(buf_len_frm, len(self.channels)), dtype=np.float32
        )
        self.raw_fft_buf = RingBuffer(
            size=(
                buf_len_frm,
                len(self.fft_freqs),
                len(self.channels),
            ),
            dtype=np.float32,
        )
        self.raw_f0_buf = RingBuffer(
            size=(buf_len_frm, len(self.channels)),
            dtype=np.float32,
        )
        self.raw_conf_buf = RingBuffer(
            size=(buf_len_frm, len(self.channels)),
            dtype=np.float32,
        )

        # initialize output files
        if out_path != "":
            start_t = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.audio_out_file = sf.SoundFile(
                out_path + f"/{start_t}.wav",
                samplerate=fs,
                channels=len(channels),
                subtype="PCM_16",
                format="WAV",
                mode="w",
            )
            self.traj_out_file = open(out_path + f"/{start_t}.csv", "x", newline="")
            writer = csv.writer(self.traj_out_file)
            writer.writerow(
                [f"F0 Channel {ch}" for ch in channels]
                + [f"Confidence Channel {ch}" for ch in channels]
            )

        # initialise output buffers that are read by GUI
        if gui is not None:
            self.new_gui_data_available = False
            self.proc_lvl = gui.lvl_cvals[0]
            self.proc_spec = np.zeros(
                (self.raw_fft_buf.buffer.shape[1], len(self.channels) + 1)
            )
            self.proc_stft = np.zeros(
                (
                    int(np.round(gui.disp_t_stft * self.frame_rate)),
                    len(self.fft_freqs),
                    len(self.channels) + 1,
                )
            )
            self.proc_inst_f0 = np.full((1, len(self.channels) + 1), np.nan)
            self.proc_f0 = np.zeros(
                (int(np.round(gui.disp_t_f0 * self.frame_rate)), len(self.channels))
            )
            self.proc_diff = np.zeros(
                (
                    self.proc_f0.shape[0],
                    (len(self.channels) * (len(self.channels) - 1)) // 2,
                )
            )

    def start_stream(self):
        """Start recording and processing"""
        if self.is_running:
            self.stop_stream()

        # initialize audio stream
        self.stream = sounddevice.InputStream(
            samplerate=self.fs,
            blocksize=self.hop_len,
            device=self.device_no,
            channels=np.max(self.channels) + 1,
            dtype=np.int16,
            latency=None,
            extra_settings=None,
            callback=self.recording_callback,
            finished_callback=None,
            clip_off=None,
            dither_off=None,
            never_drop_input=None,
            prime_output_buffers_using_stream_callback=None,
        )
        self.stream.start()
        self.is_running = True
        self.worker = threading.Thread(
            target=self.worker_thread
        )  # thread for computations
        self.worker.start()

    def stop_stream(self):
        """Stop recording and processing"""
        if self.is_running:
            self.is_running = False
            self.worker.join()
            self.stream.stop()

    def close_stream(self):
        """Close stream, processing thread and files"""
        if self.stream is not None:
            self.stream.close()
            self.stream = None
            if self.out_path != "":
                self.audio_out_file.close()
                self.traj_out_file.close()

    def worker_thread(self):
        """The thread that does all the audio processing"""
        while self.is_running:
            with _audio_lock:
                audio = self.audio_buf.read_next(
                    self.fft_len, self.hop_len
                )  # get audio

            if audio.size == 0:
                sleep(0.001)
                continue

            lvl = self.compute_level(audio)  # compute level
            fft = self.compute_fft(audio)  # compute fft
            f0, conf = self.compute_f0(audio, lvl)  # compute f0 & confidence

            with _feat_lock:
                self.raw_lvl_buf.write(lvl)
                self.raw_fft_buf.write(fft)
                self.raw_f0_buf.write(f0)
                self.raw_conf_buf.write(conf)

            # GUI pre-processing for faster updates
            if self.gui is not None:
                self.gui_preprocessing()
                self.new_gui_data_available = True

            # write trajectories to disk if configured
            if self.out_path != "":
                writer = csv.writer(self.traj_out_file)
                writer.writerow(np.concatenate((f0[0, :], conf[0, :])))

    def recording_callback(self, data, frames, time, status):
        """Receives and stores frames from soundcard, data is of shape (frames, channels)"""
        audio_conv = (
            data[:, self.channels].astype(np.float32, order="C") / 32768.0
        )  # convert int16 to float32

        with _audio_lock:
            self.audio_buf.write(audio_conv)

        if self.out_path != "":
            self.audio_out_file.write(audio_conv)

    @staticmethod
    def compute_level(audio):
        """Peak level in dB"""
        return 10 * np.log10(np.max(np.abs(audio + eps), axis=0)).reshape(1, -1)

    def compute_fft(self, audio):
        """FFT"""
        return np.abs(np.fft.rfft(audio * self.fft_win, self.fft_len, axis=0))[
            np.newaxis, :, :
        ]

    def compute_f0(self, audio, lvl):
        """Fundamental frequency estimation"""
        f0 = np.zeros((1, audio.shape[1]))
        conf = np.zeros((1, audio.shape[1]))

        for c in range(audio.shape[1]):
            if lvl[0, c] < self.f0_lvl_threshold:
                continue

            audio_tmp = np.concatenate(
                (audio[:, c][::-1], audio[:, c], audio[:, c][::-1])
            )
            if self.f0_algorithm == "YIN":
                f0_tmp, _, conf_tmp = libf0.yin(
                    audio_tmp,
                    Fs=self.fs,
                    N=self.fft_len,
                    H=self.fft_len,
                    F_min=80.0,
                    F_max=640.0,
                    threshold=0.15,
                    verbose=False,
                )
                f0[:, c] = np.mean(f0_tmp)  # take the center frame
                conf[:, c] = 1 - np.mean(conf_tmp)
            elif self.f0_algorithm == "SWIPE":
                # TODO: replace with real-time version when available
                f0_tmp, _, conf_tmp = libf0.swipe(
                    audio[:, c], Fs=self.fs, H=self.fft_len, F_min=80.0, F_max=640.0
                )
                f0[:, c] = np.mean(f0_tmp)
                conf[:, c] = 1 - np.mean(conf_tmp)
            else:
                f0[:, c] = np.zeros(f0.shape[0])
                conf[:, c] = np.zeros(f0.shape[0])

        return f0, conf

    def gui_preprocessing(self):
        """Prepares computed features for display in GUI which speeds up everything"""
        # get raw data
        lvl, spec, stft, f0, conf = self.read_latest_frames(
            self.gui.disp_t_lvl,
            self.gui.disp_t_spec,
            self.gui.disp_t_stft,
            self.gui.disp_t_f0,
            self.gui.disp_t_conf,
        )

        # compute max level and clip
        proc_lvl = np.clip(
            np.max(lvl, axis=0),
            a_min=self.gui.lvl_cvals[0],
            a_max=self.gui.lvl_cvals[-1],
        )

        # preprocess spectrum
        n_spec_frames = spec.shape[0]
        spec = np.mean(spec, axis=0)
        proc_spec = np.zeros((spec.shape[0], spec.shape[1] + 1))
        proc_spec[:, :-1] = spec
        proc_spec[:, -1] = np.prod(spec, axis=1)
        if self.gui.cur_spec_scale_type == "log":
            proc_spec = np.log(1 + 1 * proc_spec)
        max_values = np.abs(proc_spec).max(axis=0)
        proc_spec /= np.where(max_values != 0, max_values, 1)

        # preprocess stft
        proc_stft = np.zeros((stft.shape[0], stft.shape[1], stft.shape[2] + 1))
        proc_stft[:, :, :-1] = stft
        proc_stft[:, :, -1] = np.prod(stft, axis=2)
        if self.gui.cur_spec_scale_type == "log":
            proc_stft = np.log(1 + 1 * proc_stft)
        max_values = np.max(np.abs(proc_stft), axis=(0, 1), keepdims=True)
        proc_stft /= np.where(max_values != 0, max_values, 1)

        # preprocess f0
        median_len = self.gui.cur_smoothing_len
        if median_len > 0:
            idcs = np.argwhere(f0 > 0)
            f0[idcs] = median_filter(f0[idcs], size=median_len, axes=(0,))
            conf[idcs] = median_filter(conf[idcs], size=median_len, axes=(0,))

        inst_f0 = np.mean(f0[-n_spec_frames:, :], axis=0)
        inst_conf = np.mean(conf[-n_spec_frames:, :], axis=0)
        inst_f0[inst_conf < self.gui.cur_conf_threshold] = np.nan

        # compute reference frequency
        cur_ref_freq_mode = self.gui.cur_ref_freq_mode
        ref_freq = self.gui.cur_ref_freq
        if cur_ref_freq_mode == "fixed":
            cur_ref_freq = ref_freq
        elif cur_ref_freq_mode == "highest":
            cur_ref_freq = np.max(np.mean(f0, axis=0))
        elif cur_ref_freq_mode == "lowest":
            cur_ref_freq = np.min(np.mean(f0, axis=0))
        else:
            cur_ref_freq = f0[-1, int(cur_ref_freq_mode[-2:]) - 1]

        # threshold trajectories and compute intervals
        nan_val = 99999
        proc_f0, proc_diff = self.f0_diff_computations(
            f0,
            conf,
            self.gui.cur_conf_threshold,
            self.gui.cur_derivative_tol,
            cur_ref_freq,
            nan_val,
        )
        proc_f0[proc_f0 == nan_val] = np.nan
        proc_diff[proc_diff == nan_val] = np.nan

        with _gui_lock:
            self.proc_lvl = proc_lvl
            self.proc_spec[:] = proc_spec
            self.proc_stft[:] = proc_stft
            self.proc_f0[:] = proc_f0
            self.proc_inst_f0[:, :-1] = inst_f0
            self.proc_diff[:] = proc_diff

    @staticmethod
    @njit
    def f0_diff_computations(
        f0, conf, cur_conf_threshold, cur_derivative_tol, cur_ref_freq, nan_val
    ):
        """Computes pair-wise differences between F0-trajectories, speed-up using jit-compilation"""
        proc_f0 = np.ones_like(f0) * nan_val

        for i in range(f0.shape[1]):
            # filter f0 using confidence threshold and gradient filter
            index = np.where((conf[:, i] >= cur_conf_threshold) & (f0[:, i] > 0))[0]
            index_grad = gradient_filter(f0[:, i], cur_derivative_tol)
            index = np.intersect1d(index, index_grad)

            proc_f0[index, i] = f2cent(f0[index, i], cur_ref_freq)

        proc_diff = (
            np.ones((f0.shape[0], (f0.shape[1] * (f0.shape[1] - 1)) // 2)) * nan_val
        )
        if f0.shape[1] > 1:
            pair_num = 0
            for ch0 in range(f0.shape[1]):
                for ch1 in range(f0.shape[1]):
                    if ch0 >= ch1:
                        continue

                    index = np.where(
                        (proc_f0[:, ch0] != nan_val) & (proc_f0[:, ch1] != nan_val)
                    )[0]
                    proc_diff[index, pair_num] = (
                        proc_f0[index, ch0] - proc_f0[index, ch1]
                    )
                    pair_num += 1

        return proc_f0, proc_diff

    def read_latest_frames(self, t_lvl, t_spec, t_stft, t_f0, t_conf):
        """Reads latest t seconds from buffers"""

        with _feat_lock:
            lvl = self.raw_lvl_buf.read_latest(int(np.round(t_lvl * self.frame_rate)))
            spec = self.raw_fft_buf.read_latest(int(np.round(t_spec * self.frame_rate)))
            stft = self.raw_fft_buf.read_latest(int(np.round(t_stft * self.frame_rate)))
            f0 = self.raw_f0_buf.read_latest(int(np.round(t_f0 * self.frame_rate)))
            conf = self.raw_conf_buf.read_latest(
                int(np.round(t_conf * self.frame_rate))
            )

        return lvl, spec, stft, f0, conf

    def get_latest_gui_data(self):
        """Reads pre-processed data for GUI"""
        with _gui_lock:
            self.new_gui_data_available = False
            return (
                self.proc_lvl,
                self.proc_spec,
                self.proc_inst_f0,
                self.proc_stft,
                self.proc_f0,
                self.proc_diff,
            )
