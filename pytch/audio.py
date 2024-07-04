#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Audio Functions"""
import threading
import atexit
import numpy as np
import logging
import sounddevice
import libf0

from collections import defaultdict
from .util import f2cent, cent2f


_lock = threading.Lock()
logger = logging.getLogger("pytch.data")


def get_input_devices():
    """
    Returns a list of devices.
    """
    return sounddevice.query_devices()


def get_sampling_rate_options(device_idx):
    """
    Returns a dictionary of supported sampling rates for all devices.
    """

    candidates = [8000.0, 11025.0, 16000.0, 22050.0, 32000.0, 37800.0, 44100.0, 48000.0]
    supported_sampling_rates = defaultdict(list)
    for device_no in range(len(sounddevice.query_devices(device=device_idx))):
        for c in candidates:
            if check_sampling_rate(device_no, int(c)):
                supported_sampling_rates[device_no].append(c)

    return supported_sampling_rates


def check_sampling_rate(device_index, sampling_rate):
    """
    Validates chosen sampling rate.
    """
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
    """Based on numpy"""

    def __init__(
        self,
        sampling_rate,
        buffer_length_seconds,
        dtype=np.float32,
        tmin=0,
        proxy=None,
    ):
        self.tmin = tmin
        self.tmax = self.tmin + buffer_length_seconds
        self.sampling_rate = sampling_rate
        self.data_len = int(buffer_length_seconds * sampling_rate)
        self.dtype = dtype
        self.empty()
        self.i_filled = 0
        self._x = np.arange(self.data_len, dtype=self.dtype) * self.delta() + self.tmin
        self.proxy = self._proxy if not proxy else proxy

    def empty(self):
        self.data = np.empty((int(self.data_len)), dtype=self.dtype)

    def _proxy(self, data):
        """subclass this method do extra work on data chunk."""
        return data

    def delta(self):
        return 1.0 / self.sampling_rate

    def append(self, d):
        """append new data d to buffer f"""
        n = d.size
        if n == 1:
            self.append_value(d)
            return

        i_filled_mod = self.i_filled % self.data_len
        istop = i_filled_mod + n
        if istop >= self.data_len:
            istop_wrap = istop - self.data_len
            iwrap = n - istop_wrap
            self.data[i_filled_mod:] = d[:iwrap]
            self.data[0:istop_wrap] = d[iwrap:]
        else:
            self.data[i_filled_mod:istop] = d

        self.i_filled += n

    def append_value(self, v):
        self.data[self.i_filled % self.data_len] = v
        self.i_filled += 1

    def latest_frame_data(self, n):
        """Return the latest n samples data from buffer as array."""
        return self.proxy(
            np.take(
                self.data,
                np.arange(self.i_filled - n, self.i_filled),
                mode="wrap",
                axis=0,
            )
        )

    def latest_frame(self, seconds, clip_min=False):
        """Return the latest *seconds* data from buffer as x and y data tuple."""
        istart, istop = self.latest_indices(seconds)
        n = int(seconds * self.sampling_rate) + 1
        x = self.i_filled / self.sampling_rate - self._x[:n][::-1]
        if clip_min:
            istart = np.where(x > 0)[0]
            if not len(istart):
                istart = 0
            else:
                istart = np.min(istart)
        else:
            istart = 0
        return (x[istart:], self.latest_frame_data(n - istart))

    def latest_indices(self, seconds):
        return (
            self.i_filled - int(min(seconds * self.sampling_rate, self.i_filled)),
            self.i_filled,
        )


class RingBuffer2D(RingBuffer):
    """2 dimensional ring buffer. E.g. used to buffer spectrogram data."""

    def __init__(self, ndimension2, *args, **kwargs):
        self.ndimension2 = ndimension2
        RingBuffer.__init__(self, *args, **kwargs)

    def empty(self):
        self.data = np.ones(
            (int(self.data_len), int(self.ndimension2)), dtype=self.dtype
        )

    def append(self, d):
        if len(d.shape) == 1:
            self.append_value(d)
            return

        n, n2 = d.shape
        if n2 != self.ndimension2:
            raise Exception("ndim2 wrong")

        istop = self.i_filled + n
        if istop >= self.data_len:
            istop %= self.data_len
            iwrap = n - istop
            self.data[self.i_filled :] = d[:iwrap]
            self.data[0:istop] = d[iwrap:]
        else:
            self.data[self.i_filled : istop] = d

        self.i_filled = istop

    def append_value(self, v):
        self.i_filled += 1
        self.i_filled %= self.data_len
        self.data[self.i_filled, :] = v


class Channel(RingBuffer):
    def __init__(self, sampling_rate, fftsize=2048):
        self.buffer_length_seconds = 40
        RingBuffer.__init__(self, sampling_rate, self.buffer_length_seconds)

        self.__algorithm = "YIN"
        self.name = ""
        self.fftsize = fftsize
        self.setup_buffers()
        self.standard_frequency = 220.0
        self.pitch_shift = 0.0

    def pitch_proxy(self, data):
        # TODO refactor to processing module
        return f2cent(data, self.standard_frequency) + self.pitch_shift

    def undo_pitch_proxy(self, data):
        # TODO refactor to processing module
        return cent2f(data - self.pitch_shift, self.standard_frequency)

    def setup_buffers(self):
        """Setup Buffers."""
        nfft = (int(self.fftsize), self.delta())
        self.freqs = np.fft.rfftfreq(*nfft)
        sr = int(1000.0 / 58.0)
        # TODO: 58=gui refresh rate. Nastily hard coded here for now
        self.fft = RingBuffer2D(
            ndimension2=self.fftsize / 2 + 1,
            # sampling_rate=self.sampling_rate/self.fftsize,   # Hop size
            sampling_rate=sr,
            buffer_length_seconds=self.buffer_length_seconds,
            dtype=np.float32,
        )
        self.fft_power = RingBuffer(
            sampling_rate=sr, buffer_length_seconds=self.buffer_length_seconds
        )
        self.pitch = RingBuffer(
            sampling_rate=sr,
            buffer_length_seconds=self.sampling_rate
            * self.buffer_length_seconds
            / self.fftsize,
            proxy=self.pitch_proxy,
        )
        self.pitch_confidence = RingBuffer(
            sampling_rate=sr,
            buffer_length_seconds=self.sampling_rate
            * self.buffer_length_seconds
            / self.fftsize,
        )

    def latest_confident_indices(self, n, threshold):
        return np.where(self.pitch_confidence.latest_frame_data(n) >= threshold)

    @property
    def fftsize(self):
        return self.__fftsize

    @fftsize.setter
    def fftsize(self, size):
        self.__fftsize = size
        self.setup_buffers()

    @property
    def pitch_algorithm(self):
        return self.__algorithm

    @pitch_algorithm.setter
    def pitch_algorithm(self, alg):
        self.__algorithm = alg
        self.setup_buffers()

    def get_latest_pitch(self):
        return self.pitch.latest_frame_data(1)

    def compute_pitch(self, x):
        if self.pitch_algorithm == "YIN":
            f0, t, conf = libf0.yin(
                x,
                Fs=self.sampling_rate,
                N=2048,
                H=256,
                F_min=55.0,
                F_max=1760.0,
                threshold=0.15,
                verbose=False,
            )
        else:
            f0, t, conf = libf0.swipe(
                x,
                Fs=self.sampling_rate,
                H=256,
                F_min=55.0,
                F_max=1760.0,
                dlog2p=1 / 96,
                derbs=0.1,
                strength_threshold=0,
            )

        return f0, t, conf


class MicrophoneRecorder:
    """Interfacing Sounddevice to record data from sound"""

    def __init__(
        self,
        chunksize=512,
        device_no=None,
        sampling_rate=None,
        fftsize=1024,
        selected_channels=None,
    ):
        self.frames = []
        atexit.register(self.terminate)

        selected_channels = selected_channels or []
        self.stream = None
        self.nchannels = max(selected_channels) + 1

        self.device_no = device_no
        self.sampling_rate = sampling_rate
        self.selected_channels = selected_channels

        self.channels = []
        for i in range(self.nchannels):
            self.channels.append(Channel(self.sampling_rate, fftsize=fftsize))

        self.channels = [self.channels[i] for i in self.selected_channels]

        self.chunksize = chunksize

    @property
    def fftsizes(self):
        """List of sampling rates of all channels registered by the input
        device"""
        return [c.fftsize for c in self.channels]

    @property
    def sampling_rate_options(self):
        """List of supported sampling rates."""
        return get_sampling_rate_options(self.device_no)

    def new_frame(self, data, frames, time, status):
        """Callback function called as soon as we have audio data available
        available data."""

        with _lock:
            self.frames.append(data.astype(np.float32, order="C") / 32768.0)
            if self._stop:
                return None

        self.flush()

        return None

    def get_frames(self):
        """Read frames and empty pre-buffer."""
        with _lock:
            frames = self.frames
            self.frames = []
        return frames

    def start(self):
        if self.stream is None:
            self.start_new_stream()

        self.stream.start()
        self._stop = False

    @property
    def sampling_rate(self):
        return self.__sampling_rate

    @sampling_rate.setter
    def sampling_rate(self, rate):
        check_sampling_rate(self.device_no, rate)
        self.__sampling_rate = rate

    def start_new_stream(self):
        """Start audio stream."""
        self.frames = []
        self.stream = sounddevice.InputStream(
            samplerate=self.sampling_rate,
            blocksize=self.chunksize,
            device=self.device_no,
            channels=self.nchannels,
            dtype=np.int16,
            latency=None,
            extra_settings=None,
            callback=self.new_frame,
            finished_callback=None,
            clip_off=None,
            dither_off=None,
            never_drop_input=None,
            prime_output_buffers_using_stream_callback=None,
        )
        self._stop = False
        logger.debug("starting new stream: %s" % self.stream)
        self.stream.start()

    def stop(self):
        with _lock:
            self._stop = True
        if self.stream is not None:
            self.stream.stop()

    def close(self):
        self.stop()
        self.stream.close()

    def terminate(self):
        if self.stream:
            self.close()
        logger.debug("terminated stream")

    @property
    def deltat(self):
        return 1.0 / self.sampling_rate

    def flush(self):
        """read data and put it into channels' track_data"""
        # make this entirely numpy:
        frames = np.array(self.get_frames())
        for frame in frames:
            r = np.reshape(frame, (self.chunksize, self.nchannels)).T
            for channel, i in zip(self.channels, self.selected_channels):
                channel.append(r[i])


class Worker:
    def __init__(self, channels):
        """
        The Worker does the signal processing in its' `process` method.

        :param channels: list of `pytch.data.Channel` instances"""

        super().__init__()
        self.channels = channels

    def process(self):
        """Process the channels' data and update the channel instances."""
        logger.debug("processing data")

        for ic, channel in enumerate(self.channels):
            frame_work = channel.latest_frame_data(channel.fftsize)
            win = np.hanning(channel.fftsize)
            amp_spec = np.abs(np.fft.rfft(frame_work * win)) ** 2 / channel.fftsize
            channel.fft.append(np.asarray(amp_spec, dtype=np.float32))

            f0, t, conf = channel.compute_pitch(frame_work)
            channel.pitch_confidence.append_value(np.mean(conf))

            channel.pitch.append_value(np.mean(f0))
