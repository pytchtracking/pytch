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


_audio_lock = threading.Lock()
_gui_lock = threading.Lock()
logger = logging.getLogger("pytch.audio")

eps = np.finfo(float).eps


def get_input_devices():
    """Returns a list of devices."""
    return sounddevice.query_devices()


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
        self.size = size
        self.buffer = np.zeros(size, dtype=dtype)
        self.write_head = 0
        self.available_frames = 0

    def write(self, data):
        if data.shape[0] > self.size[0]:
            logger.warning("Buffer overflow!")
        write_idcs = np.mod(self.write_head + np.arange(data.shape[0]), self.size[0])
        self.buffer[write_idcs, ...] = data
        self.write_head = np.mod(write_idcs[-1] + 1, self.size[0])
        if self.available_frames < self.size[0]:
            self.available_frames += data.shape[0]

    def read(self, blocksize):
        # make sure that we are not reading more than we are allowed to
        if self.available_frames < blocksize:
            read_frames = self.available_frames
            logger.warning("Buffer underflow!")
        else:
            read_frames = blocksize
        read_idcs = np.mod(
            self.size[0] + self.write_head - np.arange(read_frames) - 1, self.size[0]
        )[::-1]
        return self.buffer[read_idcs, ...]

    def flush(self):
        self.buffer = np.zeros_like(self.buffer)
        self.write_head = 0
        self.available_frames = 0


class AudioProcessor:
    """
    Class for recording and processing of multichannel audio
    """

    def __init__(
        self,
        fs=8000,
        buf_len_sec=10.0,
        fft_len=512,
        hop_len=256,
        blocksize=256,
        channels=[0],
        device_no=None,
        f0_algorithm="YIN",
    ):
        self.fs = fs
        self.buf_len_sec = buf_len_sec
        self.fft_len = fft_len
        self.hop_len = hop_len
        self.channels = channels
        self.blocksize = blocksize
        self.device_no = device_no
        self.f0_algorithm = f0_algorithm
        self.stream = None
        self.is_running = False
        self.worker = threading.Thread(
            target=self.worker_thread
        )  # thread for computations
        atexit.register(self.close_stream)

        self.init_buffers()

    def init_buffers(self):
        self.blocksize = self.fft_len
        self.hop_len = self.fft_len // 2
        self.stft = ShortTimeFFT(
            np.hanning(self.fft_len),
            self.hop_len,
            self.fs,
            fft_mode="onesided",
            mfft=None,
            dual_win=None,
            scale_to=None,
            phase_shift=0,
        )
        self.fft_freqs = self.stft.f

        # initialize buffers
        buf_len_smp = int(
            np.ceil(self.buf_len_sec * self.fs / self.blocksize) * self.blocksize
        )
        n_blocks = int(buf_len_smp / self.blocksize)

        self.frames_per_block_f0 = int(np.floor(self.blocksize / self.hop_len)) + 1
        self.frames_per_block_stft = len(self.stft.t(self.blocksize))
        self.audio_buf = RingBuffer(
            size=(buf_len_smp, len(self.channels)), dtype=np.float64
        )
        self.lvl_buf = RingBuffer(size=(n_blocks, len(self.channels)), dtype=np.float64)
        self.stft_buf = RingBuffer(
            size=(
                n_blocks * self.frames_per_block_stft,
                1 + self.fft_len // 2,
                len(self.channels),
            ),
            dtype=np.float64,
        )
        self.f0_buf = RingBuffer(
            size=(n_blocks * self.frames_per_block_f0, len(self.channels)),
            dtype=np.float64,
        )
        self.conf_buf = RingBuffer(
            size=(n_blocks * self.frames_per_block_f0, len(self.channels)),
            dtype=np.float64,
        )

    def start_stream(self):
        if self.is_running:
            self.stop_stream()

        # initialize audio stream
        self.stream = sounddevice.InputStream(
            samplerate=self.fs,
            blocksize=self.blocksize,
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
        self.worker.start()

    def stop_stream(self):
        if self.is_running:
            self.stream.stop()
            self.is_running = False
            self.worker.join()

    def close_stream(self):
        if self.is_running:
            self.stop_stream()
            self.stream.close()
            self.stream = None

    def worker_thread(self):
        while self.is_running:
            audio = None
            with _audio_lock:
                if self.audio_buf.available_frames >= self.fft_len + self.fft_len // 2:
                    audio = self.audio_buf.read(
                        self.fft_len + self.fft_len // 2
                    )  # get audio

            if audio is None:
                sleep(self.blocksize / self.fs)
                continue

            lvl = self.compute_level(audio[: self.fft_len])  # compute level
            stft = self.compute_stft(audio)  # compute stft
            f0, conf = self.compute_f0(audio[: self.fft_len])  # compute f0 & confidence

            with _gui_lock:
                self.lvl_buf.write(lvl)
                self.stft_buf.write(stft)
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
        return 10 * np.log10(np.max(np.abs(audio), axis=0)).reshape(-1, 1)

    def compute_stft(self, audio):
        return np.transpose(
            np.abs(self.stft.stft(audio, axis=0, padding="even")), (2, 0, 1)
        )[:-1, :]

    def compute_f0(self, audio):
        f0 = np.zeros((self.frames_per_block_f0, audio.shape[1]))
        conf = np.zeros((self.frames_per_block_f0, audio.shape[1]))

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

        return f0, conf

    def read_block_data(self, n_blocks=1):
        with _gui_lock:
            lvl = self.lvl_buf.read(n_blocks)
            stft = self.stft_buf.read(self.frames_per_block_stft * n_blocks)
            f0 = self.f0_buf.read(self.frames_per_block_f0 * n_blocks)
            conf = self.conf_buf.read(self.frames_per_block_f0 * n_blocks)

        return lvl, stft, f0, conf
