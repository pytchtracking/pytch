import os
import threading
import atexit
import numpy as num
import logging
import pyaudio
import PyQt5.QtCore as qc

from collections import defaultdict
from functools import lru_cache
from scipy.io import wavfile
from aubio import pitch
from pytch.util import f2cent, cent2f


_lock = threading.Lock()

# This module contains buffering and input devices
# class taken from the scipy 2015 vispy talk opening example
# see https://github.com/vispy/vispy/pull/928

logger = logging.getLogger("pytch.data")


def is_input_device(device):
    return device["maxInputChannels"] != 0


def get_input_devices():
    """returns a dict of device descriptions.
    If the device's `maxInputChannels` is 0 the device is skipped
    """
    p = pyaudio.PyAudio()
    devices = []
    for i in range(p.get_device_count()):
        device = p.get_device_info_by_index(i)
        devices.append(device)

    p.terminate()
    return devices


@lru_cache(maxsize=128)
def get_sampling_rate_options(audio=None):
    """dictionary of supported sampling rates for all devices."""

    if not audio:
        paudio = pyaudio.PyAudio()
    else:
        paudio = audio

    candidates = [8000.0, 11025.0, 16000.0, 22050.0, 32000.0, 37800.0, 44100.0, 48000.0]
    supported_sampling_rates = defaultdict(list)
    for device_no in range(paudio.get_device_count()):
        for c in candidates:
            if check_sampling_rate(device_no, int(c), audio=paudio):
                supported_sampling_rates[device_no].append(c)

    if not audio:
        paudio.terminate()

    return supported_sampling_rates


def check_sampling_rate(device_index, sampling_rate, audio=None):
    p = audio or pyaudio.PyAudio()
    devinfo = p.get_device_info_by_index(device_index)
    valid = True
    try:
        p.is_format_supported(
            sampling_rate,
            input_device=devinfo["index"],
            input_channels=devinfo["maxinputchannels"],
            input_format=pyaudio.paint16,
        )
    except ValueError as e:
        logger.debug(e)
        valid = False

    finally:
        if not audio:
            p.terminate()
        return valid


class Buffer:

    """data container

    new data is prepended, so that the latest data point is in self.data[0]"""

    def __init__(
        self,
        sampling_rate,
        buffer_length_seconds,
        dtype=num.float32,
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
        self._x = num.arange(self.data_len, dtype=self.dtype) * self.delta + self.tmin
        self.proxy = self._proxy if not proxy else proxy

    def _proxy(self, data):
        """subclass this method do do extra work on data chunk."""
        return data

    def empty(self):
        self.data = num.empty((int(self.data_len)), dtype=self.dtype)

    def save_as(self, fn, fmt="txt"):
        fn = fn + "." + fmt
        if fmt == "txt":
            num.savetxt(fn, num.vstack((self.xdata, self.ydata)).T)
        elif fmt == "mseed":
            fn = os.path.join(fn, "." + fmt)
            try:
                from pyrocko import trace, io
            except ImportError as e:
                logger.warn("%e \n no pyrocko installation found!" % e)
                return

            tr = trace.Trace(tmin=self.tmin, deltat=self.deltat, ydata=self.ydata)
            io.save([tr], fn)

        elif fmt == "wav":
            wavfile.write(
                fn, self.sampling_rate, num.asarray(self.ydata, dtype=num.int16)
            )
        logger.info("Saved file in %s" % fn)

    @property
    def t_filled(self):
        """the time to which the data buffer contains data."""
        return self.tmin + self.i_filled * self.delta

    @property
    def delta(self):
        return 1.0 / self.sampling_rate

    @property
    def ydata(self):
        return self.proxy(self.data[: self.i_filled])

    @property
    def xdata(self):
        return self._x[: self.i_filled]

    def index_at_time(self, t):
        """Get the index of the sample (closest) defined by *t*"""
        return int((t - self.tmin) * self.sampling_rate)

    def latest_indices(self, seconds):
        return (
            self.i_filled - int(min(seconds * self.sampling_rate, self.i_filled)),
            self.i_filled,
        )

    def latest_frame(self, seconds):
        """Return the latest *seconds* data from buffer as x and y data tuple."""
        istart, istop = self.latest_indices(seconds)
        return (self._x[istart:istop], self.proxy(self.data[istart:istop]))

    def latest_frame_data(self, n):
        """Return the latest n samples data from buffer as array."""
        return self.proxy(self.data[max(self.i_filled - n, 0) : self.i_filled])

    def append(self, d):
        """Append data frame *d* to Buffer"""
        n = d.shape[0]
        self.data[self.i_filled : self.i_filled + n] = d
        self.i_filled += n

    def append_value(self, v):
        self.data[self.i_filled + 1] = v
        self.i_filled += 1

    # def energy(self, nsamples_total, nsamples_sum=1):
    #    xi = num.arange(self.i_filled-nsamples_total, self.i_filled)
    #    y = self.data[xi].reshape((int(len(xi)/nsamples_sum), nsamples_sum))
    #    y = num.sum(y**2, axis=1)
    #    return self._x[xi[::nsamples_sum]], y


class RingBuffer(Buffer):
    """Based on numpy"""

    def __init__(self, *args, **kwargs):
        Buffer.__init__(self, *args, **kwargs)

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
            num.take(
                self.data,
                num.arange(self.i_filled - n, self.i_filled),
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
            istart = num.where(x > 0)[0]
            if not len(istart):
                istart = 0
            else:
                istart = num.min(istart)
        else:
            istart = 0
        return (x[istart:], self.latest_frame_data(n - istart))


class RingBuffer2D(RingBuffer):
    """2 dimensional ring buffer. E.g. used to buffer spectrogram data."""

    def __init__(self, ndimension2, *args, **kwargs):
        self.ndimension2 = ndimension2
        RingBuffer.__init__(self, *args, **kwargs)

    def empty(self):
        self.data = num.ones(
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
    def __init__(self, sampling_rate, fftsize=8192):
        self.buffer_length_seconds = 40
        RingBuffer.__init__(self, sampling_rate, self.buffer_length_seconds)

        self.__algorithm = "yinfast"
        self.name = ""
        self.pitch_o = None
        self.fftsize = fftsize
        self.setup_pitch()
        self.setup_buffers()

        # TODO refactor to processing module
        P = 0.0
        R = 0.01**2
        Q = 1e-6
        self.kalman_pitch_filter = Kalman(P, R, Q)
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
        nfft = (int(self.fftsize), self.delta)
        self.freqs = num.fft.rfftfreq(*nfft)
        sr = int(1000.0 / 58.0)
        # TODO: 58=gui refresh rate. Nastily hard coded here for now
        self.fft = RingBuffer2D(
            ndimension2=self.fftsize / 2 + 1,
            # sampling_rate=self.sampling_rate/self.fftsize,   # Hop size
            sampling_rate=sr,
            buffer_length_seconds=self.buffer_length_seconds,
            dtype=num.uint32,
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
        return num.where(self.pitch_confidence.latest_frame_data(n) >= threshold)

    def append_value_pitch(self, val, apply_kalman=False):
        """Append a new pitch value to pitch buffer. Apply Kalman filter
        before appending"""
        if apply_kalman:
            val = self.kalman_pitch_filter.evaluate(val)
        self.pitch.append_value(val)

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
        self.setup_pitch()

    def get_latest_pitch(self):
        return self.pitch.latest_frame_data(1)

    def setup_pitch(self):
        if self.pitch_o:
            self.pitch_o = None
        tolerance = 0.8
        win_s = self.fftsize

        # TODO check parameters
        self.pitch_o = pitch(self.pitch_algorithm, win_s, win_s, self.sampling_rate)
        self.pitch_o.set_unit("Hz")
        self.pitch_o.set_tolerance(tolerance)


class DataProvider:
    """Base class defining common interface for data input to Worker"""

    def __init__(self):
        self.frames = []
        atexit.register(self.terminate)

    def terminate(self):
        # cleanup
        pass


class MicrophoneRecorder(DataProvider):
    """Interfacing PyAudio to record data from sound"""

    def __init__(
        self,
        chunksize=512,
        device_no=None,
        sampling_rate=None,
        fftsize=1024,
        selected_channels=None,
    ):
        DataProvider.__init__(self)

        selected_channels = selected_channels or []
        self.stream = None
        self.paudio = pyaudio.PyAudio()
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
        return get_sampling_rate_options(self.device_no, audio=self.paudio)

    def new_frame(self, data, frame_count, time_info, status):
        """Callback function called as soon as pyaudio anounces new
        available data."""
        data = num.asarray(num.fromstring(data, "int16"), num.float32)

        with _lock:
            self.frames.append(data)
            if self._stop:
                return None, pyaudio.paComplete

        self.flush()

        return None, pyaudio.paContinue

    def get_frames(self):
        """Read frames and empty pre-buffer."""
        with _lock:
            frames = self.frames
            self.frames = []
        return frames

    def start(self):
        if self.stream is None:
            self.start_new_stream()

        self.stream.start_stream()
        self._stop = False

    @property
    def sampling_rate(self):
        return self.__sampling_rate

    @sampling_rate.setter
    def sampling_rate(self, rate):
        check_sampling_rate(self.device_no, rate, audio=self.paudio)
        self.__sampling_rate = rate

    def start_new_stream(self):
        """Start audio stream."""
        self.frames = []
        self.stream = self.paudio.open(
            format=pyaudio.paInt16,
            channels=self.nchannels,
            rate=self.sampling_rate,
            input=True,
            output=False,
            frames_per_buffer=self.chunksize,
            input_device_index=self.device_no,
            stream_callback=self.new_frame,
        )
        self._stop = False
        logger.debug("starting new stream: %s" % self.stream)
        self.stream.start_stream()

    def stop(self):
        with _lock:
            self._stop = True
        if self.stream is not None:
            self.stream.stop_stream()

    def close(self):
        self.stop()
        self.stream.close()

    def terminate(self):
        if self.stream:
            self.close()
        self.paudio.terminate()
        logger.debug("terminated stream")

    @property
    def deltat(self):
        return 1.0 / self.sampling_rate

    def flush(self):
        """read data and put it into channels' track_data"""
        # make this entirely numpy:
        frames = num.array(self.get_frames())
        for frame in frames:
            r = num.reshape(frame, (self.chunksize, self.nchannels)).T
            for channel, i in zip(self.channels, self.selected_channels):
                channel.append(r[i])


class Worker(qc.QObject):
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
            win = num.hanning(channel.fftsize)
            # slight pre-emphasis
            # frame_work[1:] -=  0.1 * frame_work[:-1]
            # frame_work[0] = frame_work[1]

            amp_spec = num.abs(num.fft.rfft(frame_work * win)) ** 2 / channel.fftsize
            channel.fft.append(num.asarray(amp_spec, dtype=num.uint32))

            channel.pitch_confidence.append_value(channel.pitch_o.get_confidence())

            channel.pitch.append_value(channel.pitch_o(frame_work)[0])


def cross_spectrum(spec1, spec2):
    """Returns cross spectrum and phase of *spec1* and *spec2*"""
    cross = spec1 * spec2.conjugate()
    return num.abs(cross), num.unwrap(num.arctan2(cross.imag, cross.real))


class Kalman:
    """A simple Kalman filter which can be applied recusively to continuous
    data.

    A Python implementation of the example given in pages 11-15 of "An
    Introduction to the Kalman Filter" by Greg Welch and Gary Bishop,
    University of North Carolina at Chapel Hill, Department of Computer
    Science, TR 95-041,
    http://www.cs.unc.edu/~welch/kalman/kalmanIntro.html

    by Andrew D. Straw
    """

    def __init__(self, P, R, Q):
        self.P = P
        self.R = R
        self.Q = Q

    def evaluate(self, new_sample, previous_estimate, weight=1.0, dt=None):
        """Calculate the next estimate, based on the
        *new_sample* and the *previous_sample*"""

        # time update
        xhatminus = previous_estimate
        Pminus = self.P + self.Q * dt * 100.0

        # measurement update
        K = Pminus / (Pminus + self.R) * weight
        self.P = (1 - K) * Pminus
        return xhatminus + K * (new_sample - xhatminus)

    def evaluate_array(self, array):
        xhat = num.zeros(array.shape)
        for k in range(1, len(array)):
            new_sample = array[k]  # grab a new sample from the data set

            # get a filtered new estimate:
            xhat[k] = self.evaluate(
                new_sample=new_sample, previous_estimate=xhat[k - 1]
            )

        return xhat
