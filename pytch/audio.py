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
from rtswipe import RTSwipe
from scipy.ndimage import median_filter
from datetime import datetime
import csv

logger = logging.getLogger("pytch.audio")

eps = np.finfo(float).eps


def get_input_devices():
    """Returns a list of input devices.

    Returns:
        List of available input devices.

    """
    input_devices = []
    for device_id, device in enumerate(sounddevice.query_devices()):
        if device["max_input_channels"] > 0:
            input_devices.append((device_id, device))
    return input_devices


def get_fs_options(device_idx):
    """Returns a dictionary of supported sampling rates for all devices.

    Args:
        device_idx: Device index.

    Returns:
        List of supported sampling rates.

    """
    candidates = [8000.0, 11025.0, 16000.0, 22050.0, 32000.0, 37800.0, 44100.0, 48000.0]
    supported_fs = []
    for c in candidates:
        if check_fs(device_idx, int(c)):
            supported_fs.append(c)

    return supported_fs


def check_fs(device_index, fs):
    """Validates chosen sampling rate.

    Args:
        device_index: Device index.
        fs: Sampling rate.

    Returns:
        True if sampling rate is supported, else False.

    """
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

    return valid


@njit
def f2cent(f, f_ref=440.0):
    """Convert frequency from Hz to Cents.

    Args:
        f: Frequency.
        f_ref: Reference frequency.

    Returns:
        Frequency in Cents.

    """
    return 1200.0 * np.log2(np.abs(f) / f_ref + eps)


@njit
def gradient_filter(y, max_gradient):
    """Gradient filter.

    Args:
        y: Signal.
        max_gradient: Upper boundary for absolute gradient.

    Returns:
        Indices where the absolute gradient of y is < max_gradient.

    """
    return np.where(np.abs(np.diff(f2cent(y))) < max_gradient)[0]


class RingBuffer:
    """Generic ring buffer for n-dimensional data"""

    def __init__(self, size, dtype):
        """Initialize buffer.

        Args:
            size: buffer size (n_frames, ..., n_channels)
            dtype: buffer dtype
        """
        self.size = size
        self.buffer = np.zeros(size, dtype=dtype)
        self.write_head = 0
        self.read_head = 0
        self.lock = threading.Lock()

    def write(self, data):
        """Writes data to buffer.

        Args:
            data: Data of shape (n_frames, ..., n_channels).

        """
        if data.shape[0] > self.size[0]:
            logger.warning("Buffer overflow!")
        with self.lock:
            write_idcs = np.mod(
                self.write_head + np.arange(data.shape[0]), self.size[0]
            )
            self.buffer[write_idcs, ...] = data
            self.write_head = np.mod(
                write_idcs[-1] + 1, self.size[0]
            )  # set write head to the next bin to write to

    def read_latest(self, n_frames):
        """Read latest n_frames frames from buffer, starting from write head.

        Args:
            n_frames: Number of frames to read.

        Returns:
            Read data.

        """
        if self.size[0] < n_frames:
            Exception("cannot read more data than buffer length!")

        with self.lock:
            read_idcs = np.mod(
                self.size[0] + self.write_head - np.arange(n_frames) - 1, self.size[0]
            )[::-1]
            return self.buffer[read_idcs, ...]

    def read_next(self, n_frames, hop_frames=None):
        """Read n_frames frames from buffer, starting from read head.

        Args:
            n_frames: Number of frames to read.
            hop_frames: Read head increment.

        Returns:
            Read data.

        """
        with self.lock:
            if (
                np.mod(self.size[0] + self.write_head - self.read_head, self.size[0])
                < n_frames
            ):
                # return empty array if not enough data available
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
        """Flush buffer."""
        self.buffer = np.zeros_like(self.buffer)
        self.write_head = 0
        self.read_head = 0


class AudioProcessor:
    """Class for recording and processing of multichannel audio."""

    def __init__(
        self,
        fs=8000,
        buf_len_sec=30.0,
        fft_len=512,
        channels=None,
        device_no=None,
        out_path="",
    ):
        """Initialize audio processing.

        Args:
            fs: Sampling rate.
            buf_len_sec: Buffer length in seconds.
            fft_len: FFT length in bins.
            channels: List of channels to record.
            device_no: Index of device to record from.
            out_path: Output directory for F0 trajectories.
        """
        self.fs = fs
        self.buf_len_sec = buf_len_sec
        self.fft_len = fft_len
        self.hop_len = self.fft_len // 2
        self.fft_freqs = np.fft.rfftfreq(self.fft_len, 1 / self.fs)
        self.fft_win = np.hanning(self.fft_len).reshape(-1, 1)
        self.channels = [0] if channels is None else channels
        self.device_no = device_no
        self.out_path = out_path
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

        # initialize real-time SWIPE
        self.rtswipe = RTSwipe(
            fs=self.fs,
            hop_len=self.fft_len,
            f_min=55.0,
            f_max=1760.0,
            num_channels=len(channels),
            delay=0.0,
        )

    def start_stream(self):
        """Start recording and processing"""
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
            self.stream.stop()
            self.worker.join()
            self.audio_buf.flush()
            self.raw_lvl_buf.flush()
            self.raw_fft_buf.flush()
            self.raw_f0_buf.flush()
            self.raw_conf_buf.flush()

    def close_stream(self):
        """Close stream, processing thread and files"""
        if self.stream is not None:
            self.stream.close()
            self.stream = None
            if self.out_path != "":
                self.audio_out_file.close()
                self.traj_out_file.close()

    def worker_thread(self):
        """The thread that does the audio processing"""
        while self.is_running:
            audio = self.audio_buf.read_next(self.fft_len, self.hop_len)  # get audio

            if audio.size == 0:
                sleep(0.001)
                continue

            start_t = time()
            lvl = self.compute_level(audio)  # compute level
            fft = self.compute_fft(audio)  # compute fft
            f0, conf = self.compute_f0(audio, lvl)  # compute f0 & confidence
            logger.debug(f"Processing took {time()-start_t:.4f}s.")

            self.raw_lvl_buf.write(lvl)
            self.raw_fft_buf.write(fft)
            self.raw_f0_buf.write(f0)
            self.raw_conf_buf.write(conf)

            # write trajectories to disk if configured
            if self.out_path != "":
                writer = csv.writer(self.traj_out_file)
                writer.writerow(np.concatenate((f0[0, :], conf[0, :])))

    def recording_callback(self, data, frames, time, status):
        """Receives frames from soundcard and stores them in buffer, data is of shape (frames, channels)"""
        audio_conv = (
            data[:, self.channels].astype(np.float32, order="C") / 32768.0
        )  # convert int16 to float32

        self.audio_buf.write(audio_conv)

        if self.out_path != "":
            self.audio_out_file.write(audio_conv)

    @staticmethod
    def compute_level(audio):
        """Computes peak level in dB"""
        return 10 * np.log10(np.max(np.abs(audio + eps), axis=0)).reshape(1, -1)

    def compute_fft(self, audio):
        """Computes the Fast Fourier Transform (FFT)"""
        return np.abs(np.fft.rfft(audio * self.fft_win, self.fft_len, axis=0))[
            np.newaxis, :, :
        ]

    def compute_f0(self, audio, lvl):
        """Fundamental frequency (F0) estimation.

        Args:
            audio: audio signal
            lvl: audio levels

        Returns:
            f0: F0 estimate.
            conf: Confidence.

        """
        if np.all(lvl > self.f0_lvl_threshold):
            f0, conf = self.rtswipe(audio)
        else:
            f0 = conf = np.zeros((1, len(self.channels)))

        return f0, conf

    @staticmethod
    @njit
    def f0_diff_computations(f0, conf, conf_threshold, gradient_tol, ref_freq, nan_val):
        """Computes pair-wise differences between F0-trajectories, speed-up using jit-compilation.

        Args:
            f0: Fundamental frequencies of all voices.
            conf: Confidences of all voices.
            conf_threshold: Confidence threshold.
            gradient_tol: Tolerance for gradient filter.
            ref_freq: Reference frequency.
            nan_val: Value that is used in replace for NaN.

        Returns:
            proc_f0: Thresholded and smoothed F0 trajectories in Cents.
            proc_diff: Harmonic differences between voices in Cents.

        """
        proc_f0 = np.ones_like(f0) * nan_val

        for i in range(f0.shape[1]):
            # filter f0 using confidence threshold and gradient filter
            index = np.where((conf[:, i] >= conf_threshold) & (f0[:, i] > 0))[0]
            index_grad = gradient_filter(f0[:, i], gradient_tol)
            index = np.intersect1d(index, index_grad)

            proc_f0[index, i] = f2cent(f0[index, i], ref_freq)

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

    def get_gui_data(
        self,
        disp_t_lvl,
        disp_t_spec,
        disp_t_stft,
        disp_t_f0,
        disp_t_conf,
        lvl_cvals,
        spec_scale_type,
        smoothing_len,
        conf_threshold,
        ref_freq_mode,
        ref_freq,
        gradient_tol,
    ):
        """Reads and prepares data for GUI.

        Args:
            disp_t_lvl: Time for level computation.
            disp_t_spec: Time for spectrum computation.
            disp_t_stft: Time for spectrogram computation.
            disp_t_f0: Time for F0 computation.
            disp_t_conf: Time for confidence computation.
            lvl_cvals: GUI level limits.
            spec_scale_type: Spectral scale type.
            smoothing_len: Smoothing filter length in frames.
            conf_threshold: Confidence threshold.
            ref_freq_mode: Reference frequency mode.
            ref_freq: Reference frequency.
            gradient_tol: Gradient filter tolerance.

        Returns:
            lvl: Levels for all channels.
            spec: Spectra for all channels & product.
            inst_f0: Instantaneous F0 for all channels & product.
            stft: Spectrograms for all channels & product.
            f0: F0 estimates for all channels.
            diff: Differential F0s (harmonic intervals) for all channels.

        """
        start_t = time()

        # read latest data from buffer
        # why not read_next()? -> we prioritize low latency over completeness of the visualized data.
        lvl = self.raw_lvl_buf.read_latest(int(np.round(disp_t_lvl * self.frame_rate)))
        spec_raw = self.raw_fft_buf.read_latest(
            int(np.round(disp_t_stft * self.frame_rate))
        )
        f0 = self.raw_f0_buf.read_latest(int(np.round(disp_t_f0 * self.frame_rate)))
        conf = self.raw_conf_buf.read_latest(
            int(np.round(disp_t_conf * self.frame_rate))
        )

        # compute max level and clip
        if len(lvl) > 0:
            lvl = np.clip(
                np.max(lvl, axis=0),
                a_min=lvl_cvals[0],
                a_max=lvl_cvals[-1],
            )

        # preprocess spectrum
        if len(spec_raw) > 0:
            n_spec_frames = int(np.round(spec_raw.shape[0] * disp_t_spec / disp_t_stft))
            spec = np.mean(spec_raw[-n_spec_frames:, :, :], axis=0)
            spec = np.concatenate((spec, np.prod(spec, axis=1).reshape(-1, 1)), axis=-1)
            if spec_scale_type == "log":
                spec = np.log(1 + 1 * spec)
            max_values = np.abs(spec).max(axis=0)
            spec /= np.where(max_values != 0, max_values, 1)
        else:
            spec = np.array([])

        # preprocess stft
        if len(spec_raw) > 0:
            stft = np.zeros(
                (spec_raw.shape[0], spec_raw.shape[1], spec_raw.shape[2] + 1)
            )
            stft[:, :, :-1] = spec_raw
            stft[:, :, -1] = np.prod(spec_raw, axis=2)
            if spec_scale_type == "log":
                stft = np.log(1 + 1 * stft)
            max_values = np.max(np.abs(stft), axis=(0, 1), keepdims=True)
            stft /= np.where(max_values != 0, max_values, 1)
        else:
            stft = np.array([])

        # preprocess f0
        if len(f0) > 0:
            median_len = smoothing_len
            if median_len > 0:
                idcs = np.argwhere(f0 > 0)
                f0[idcs] = median_filter(f0[idcs], size=median_len, axes=(0,))
                conf[idcs] = median_filter(conf[idcs], size=median_len, axes=(0,))

            n_spec_frames = int(np.round(spec_raw.shape[0] * disp_t_spec / disp_t_stft))
            inst_f0 = np.mean(f0[-n_spec_frames:, :], axis=0)
            inst_f0 = np.concatenate((inst_f0, [0]))
            inst_conf = np.mean(conf[-n_spec_frames:, :], axis=0)
            inst_conf = np.concatenate((inst_conf, [0]))
            inst_f0[inst_conf < conf_threshold] = np.nan

            # compute reference frequency
            ref_freq_mode = ref_freq_mode
            ref_freq = ref_freq
            if ref_freq_mode == "fixed":
                ref_freq = ref_freq
            elif ref_freq_mode == "highest":
                ref_freq = np.max(np.mean(f0, axis=0))
            elif ref_freq_mode == "lowest":
                ref_freq = np.min(np.mean(f0, axis=0))
            else:
                ref_freq = f0[-1, int(ref_freq_mode[-2:]) - 1]

            # threshold trajectories and compute intervals
            nan_val = 99999
            f0, diff = self.f0_diff_computations(
                f0,
                conf,
                conf_threshold,
                gradient_tol,
                ref_freq,
                nan_val,
            )
            f0[f0 == nan_val] = np.nan
            diff[diff == nan_val] = np.nan
        else:
            inst_f0 = np.array([])
            diff = np.array([])

        logger.debug(f"GUI pre-processing took {time()-start_t:.4f}s.")

        return lvl, spec, inst_f0, stft, f0, diff
