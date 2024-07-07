#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Audio Functions"""
import threading
from time import sleep
import atexit
import numpy as np
from scipy.signal import ShortTimeFFT
import logging
import sounddevice
import libf0


_lock = threading.Lock()
logger = logging.getLogger("pytch.audio")

eps = np.finfo(float).eps


def get_input_devices():
    """Returns a list of devices."""
    return sounddevice.query_devices()


def get_sampling_rate_options(device_idx):
    """Returns a dictionary of supported sampling rates for all devices."""
    candidates = [8000.0, 11025.0, 16000.0, 22050.0, 32000.0, 37800.0, 44100.0, 48000.0]
    supported_sampling_rates = []
    for c in candidates:
        if check_sampling_rate(device_idx, int(c)):
            supported_sampling_rates.append(c)

    return supported_sampling_rates


def check_sampling_rate(device_index, sampling_rate):
    """Validates chosen sampling rate."""
    valid = True
    try:
        sounddevice.check_input_settings(
            device=device_index,
            channels=None,
            dtype=None,
            extra_settings=None,
            samplerate=sampling_rate,
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
        self.size = size
        self.buffer = np.zeros(size, dtype=dtype)
        self.write_head = 0
        self.read_head = 0

    def write(self, data):
        write_idcs = np.mod(self.write_head + np.arange(len(data)), self.size[0])
        self.buffer[write_idcs, ...] = data
        self.write_head = write_idcs[-1] + 1

    def read(self, blocksize):
        read_idcs = np.mod(self.read_head + np.arange(blocksize), self.size[0])
        self.read_head = read_idcs[-1] + 1
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
        buf_len_sec=10.0,
        fft_len=1024,
        hop_len=256,
        blocksize=4096,
        n_channels=1,
        device_no=0,
        f0_algorithm="YIN",
        f0_ref=220.0,
    ):
        self.fs = fs
        self.buf_len_sec = buf_len_sec
        self.fft_len = fft_len
        self.hop_len = hop_len
        self.n_channels = n_channels
        self.blocksize = blocksize
        self.device_no = device_no
        self.f0_algorithm = f0_algorithm
        self.f0_ref = f0_ref
        self.stft = ShortTimeFFT(
            np.hanning(self.fft_len),
            hop_len,
            fs,
            fft_mode="onesided",
            mfft=None,
            dual_win=None,
            scale_to=None,
            phase_shift=0,
        )

        # initialize buffers
        buf_len_smp = int(np.ceil(buf_len_sec * fs))
        buf_len_block = int(np.ceil(buf_len_smp / blocksize))
        buf_len_hop = int(np.ceil(buf_len_smp / hop_len))
        self.audio_buf = RingBuffer(size=(buf_len_smp, n_channels), dtype=np.float64)
        self.lvl_buf = RingBuffer(size=(buf_len_block, n_channels), dtype=np.float64)
        self.fft_buf = RingBuffer(
            size=(buf_len_hop, n_channels, 1 + self.fft_len // 2), dtype=np.float64
        )
        self.f0_buf = RingBuffer(size=(buf_len_hop, n_channels), dtype=np.float64)
        self.conf_buf = RingBuffer(size=(buf_len_hop, n_channels), dtype=np.float64)
        self.f0_diff_buf = RingBuffer(
            size=(buf_len_hop, np.max([1, n_channels - 1])), dtype=np.float64
        )

        # initialize audio stream
        self.stream = sounddevice.InputStream(
            samplerate=self.fs,
            blocksize=self.blocksize,
            device=self.device_no,
            channels=self.n_channels,
            dtype=np.int16,
            latency=None,
            extra_settings=None,
            callback=self.callback,
            finished_callback=None,
            clip_off=None,
            dither_off=None,
            never_drop_input=None,
            prime_output_buffers_using_stream_callback=None,
        )

        # initialize thread for computations
        self.worker = threading.Thread(target=self.worker_thread)
        self.running = False

        atexit.register(self.close_stream)

    def start_stream(self):
        self.stream.start()
        self.running = True
        self.worker.start()

    def stop_stream(self):
        self.stream.stop()
        self.running = False
        self.worker.join()

    def close_stream(self):
        self.stream.close()

    def worker_thread(self):
        while self.running:
            audio = None
            with _lock:
                if self.audio_buf.read_head != self.audio_buf.write_head:
                    audio = self.audio_buf.read(self.blocksize)  # get audio

            if audio is None:
                sleep(self.blocksize / 4 / self.fs)
                continue

            lvl = self.compute_level(audio)  # compute level
            fft = self.compute_stft(audio)  # compute fft
            f0, conf = self.compute_f0(audio)  # compute f0 & confidence

            with _lock:
                self.lvl_buf.write(lvl)
                self.fft_buf.write(fft)
                self.f0_buf.write(f0)
                self.conf_buf.write(conf)

                if self.n_channels > 1:
                    f0_diff = self.compute_f0_diff(f0)  # compute f0 diff
                    self.f0_diff_buf.write(f0_diff)

    def callback(self, data, frames, time, status):
        """receives frames from soundcard, data is of shape (frames, channels)"""
        # only stores audio, we do all the heavy lifting in a dedicated thread for performance reasons
        with _lock:
            self.audio_buf.write(
                data.astype(np.float64, order="C") / 32768.0
            )  # convert to float

    def compute_level(self, audio):
        return 10 * np.log10(np.max(np.abs(audio), axis=0)).reshape(-1, 1)

    def compute_stft(self, audio):
        return np.transpose(
            np.abs(self.stft.stft(audio, axis=0, padding="even")), (2, 1, 0)
        )

    def compute_f0(self, audio):
        f0 = np.zeros((1 + int(self.blocksize / self.hop_len), audio.shape[1]))
        conf = np.zeros((1 + int(self.blocksize / self.hop_len), audio.shape[1]))

        for c in range(audio.shape[1]):
            if self.f0_algorithm == "YIN":
                f0[:, c], _, conf[:, c] = libf0.yin(
                    audio[:, c],
                    Fs=self.fs,
                    N=self.fft_len,
                    H=self.hop_len,
                    F_min=55.0,
                    F_max=1760.0,
                    threshold=0.15,
                    verbose=False,
                )
            else:
                f0[:, c], _, conf[:, c] = libf0.swipe(
                    audio[:, c],
                    Fs=self.fs,
                    H=self.hop_len,
                    F_min=55.0,
                    F_max=1760.0,
                    dlog2p=1 / 96,
                    derbs=0.1,
                    strength_threshold=0,
                )

        f0 = 1200.0 * np.log2(f0 / self.f0_ref + eps)  # hz to cents

        return f0, conf

    def compute_f0_diff(self, f0):
        return f0[:, 1:] - f0[:, :-1]
