#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Audio Functions"""
import threading
import time
from time import sleep
import numpy as np
import logging
import sounddevice
import libf0


_audio_lock = threading.Lock()
_gui_lock = threading.Lock()
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


class RingBuffer:
    """
    Generic ring buffer for n-dimensional data
    """

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
    """
    Class for recording and processing of multichannel audio
    """

    def __init__(
        self,
        fs=8000,
        buf_len_sec=30.0,
        fft_len=512,
        channels=[0],
        device_no=None,
        f0_algorithm="YIN",
    ):
        self.fs = fs
        self.buf_len_sec = buf_len_sec
        self.fft_len = fft_len
        self.hop_len = fft_len // 2
        self.fft_freqs = np.fft.rfftfreq(self.fft_len, 1 / self.fs)
        self.fft_win = np.hanning(self.fft_len).reshape(-1, 1)
        self.channels = channels
        self.device_no = device_no
        self.f0_algorithm = f0_algorithm
        self.frame_rate = self.fs / self.hop_len
        self.stream = None
        self.is_running = False

        # initialize buffers
        buf_len_smp = int(
            np.ceil(self.buf_len_sec * self.fs / self.hop_len) * self.hop_len
        )
        self.audio_buf = RingBuffer(
            size=(buf_len_smp, len(self.channels)), dtype=np.float64
        )

        buf_len_frm = int(
            np.floor((self.buf_len_sec * self.fs - self.fft_len) / self.hop_len)
        )
        self.lvl_buf = RingBuffer(
            size=(buf_len_frm, len(self.channels)), dtype=np.float64
        )
        self.fft_buf = RingBuffer(
            size=(
                buf_len_frm,
                len(self.fft_freqs),
                len(self.channels),
            ),
            dtype=np.float64,
        )
        self.f0_buf = RingBuffer(
            size=(buf_len_frm, len(self.channels)),
            dtype=np.float64,
        )
        self.conf_buf = RingBuffer(
            size=(buf_len_frm, len(self.channels)),
            dtype=np.float64,
        )

    def start_stream(self):
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
        if self.is_running:
            self.stream.stop()
            self.is_running = False
            self.worker.join()

    def close_stream(self):
        if self.stream is not None:
            self.stream.close()
            self.stream = None

    def worker_thread(self):
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
            f0, conf = self.compute_f0(audio)  # compute f0 & confidence

            with _gui_lock:
                self.lvl_buf.write(lvl)
                self.fft_buf.write(fft)
                self.f0_buf.write(f0)
                self.conf_buf.write(conf)

    def recording_callback(self, data, frames, time, status):
        """receives frames from soundcard, data is of shape (frames, channels)"""
        # only stores audio, we do all the heavy lifting in a dedicated thread for performance reasons
        with _audio_lock:
            self.audio_buf.write(
                data[:, self.channels].astype(np.float64, order="C") / 32768.0
            )  # convert int16 to float64

    def compute_level(self, audio):
        return 10 * np.log10(np.max(np.abs(audio + eps), axis=0)).reshape(1, -1)

    def compute_fft(self, audio):
        return np.abs(np.fft.rfft(audio * self.fft_win, self.fft_len, axis=0))[
            np.newaxis, :, :
        ]

    def compute_f0(self, audio):
        f0 = np.zeros((1, audio.shape[1]))
        conf = np.zeros((1, audio.shape[1]))

        for c in range(audio.shape[1]):
            if self.f0_algorithm == "YIN":
                f0_tmp, _, conf_tmp = libf0.yin(
                    audio[:, c],
                    Fs=self.fs,
                    N=self.fft_len,
                    H=self.fft_len,
                    F_min=55.0,
                    F_max=1650.0,
                    threshold=0.15,
                    verbose=False,
                )
                f0[:, c] = f0_tmp[1]  # take the center frame
                conf[:, c] = 1 - conf_tmp[1]

            else:
                f0_tmp, _, conf_tmp = libf0.swipe(
                    audio[:, c],
                    Fs=self.fs,
                    H=self.hop_len,
                    F_min=55.0,
                    F_max=1650.0,
                    dlog2p=1 / 96,
                    derbs=0.1,
                    strength_threshold=0,
                )
                f0[:, c] = f0_tmp[1]  # take the center frame
                conf[:, c] = conf_tmp[1]

        return f0, conf

    def read_latest_frames(self, t_lvl, t_stft, t_f0, t_conf):
        """Reads latest t seconds from buffers"""

        with _gui_lock:
            lvl = self.lvl_buf.read_latest(int(np.round(t_lvl * self.frame_rate)))
            stft = self.fft_buf.read_latest(int(np.round(t_stft * self.frame_rate)))
            f0 = self.f0_buf.read_latest(int(np.round(t_f0 * self.frame_rate)))
            conf = self.conf_buf.read_latest(int(np.round(t_conf * self.frame_rate)))

        return lvl, stft, f0, conf
