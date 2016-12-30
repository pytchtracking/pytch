import threading
import pyaudio
import atexit
import numpy as num
import logging

from pytch.util import DummySignal

_lock = threading.Lock()

# class taken from the SciPy 2015 Vispy talk opening example
# see https://github.com/vispy/vispy/pull/928

logger = logging.getLogger(__name__)


def append_to_frame(f, d):
    ''' shift data in f and append new data d to buffer f'''
    i = d.shape[0]
    #f[:-i] = f[i:]
    num.roll(f, -i)
    f[-i:] = d.T


def prepend_to_frame(f, d):
    i = d.shape[0]
    num.roll(f, i)
    f[:i] = d.T


def getaudiodevices():
    ''' Returns a list of device descriptions'''
    p = pyaudio.PyAudio()
    devices = []
    for i in range(p.get_device_count()):
        devices.append(p.get_device_info_by_index(i).get('name'))
    p.terminate()
    return devices


def sampling_rate_options(device_no, audio=None):
    ''' List of supported sampling rates.'''
    candidates = [8000., 11.025, 123123123123., 16000., 22050., 32000., 37.800, 44100.,
                  48000.]
    print(candidates)
    supported_sampling_rates = []
    for c in candidates:
        print(c)
        if check_sampling_rate(device_no, int(c), audio=audio):
            print('GREAT')
            supported_sampling_rates.append(c)

    return supported_sampling_rates


def check_sampling_rate(device_index, sampling_rate, audio=None):
    p = audio or pyaudio.PyAudio()
    devinfo = p.get_device_info_by_index(device_index)
    valid = False
    try:
        p.is_format_supported(
            sampling_rate,
            input_device=devinfo['index'],
            input_channels=devinfo['maxInputChannels'],
            input_format=pyaudio.paInt16)
    except ValueError as e:
        print(e)
        logger.debug(e)
        valid = False

    finally:
        if not audio:
            p.terminate()
        return valid


class Buffer():

    ''' Data container

    New data is prepended, so that the latest data point is in self.data[0]'''
    def __init__(self, sampling_rate, buffer_length_seconds, dtype=num.float, tmin=0):
        self.tmin = tmin
        self.tmax = self.tmin + buffer_length_seconds
        self.sampling_rate = sampling_rate
        self.data_len = buffer_length_seconds * sampling_rate
        self.dtype = dtype
        self.empty()

        self.i_filled = 0

        self._x = num.arange(self.data_len, dtype=num.float64) *self.delta + self.tmin

    def empty(self):
        self.data = num.empty((int(self.data_len)),
                          dtype=self.dtype)

    def dump(self):
        pass

    @property
    def t_filled(self):
        ''' The time to which the data buffer contains data.'''
        return self.tmin + self.i_filled*self.delta

    @property
    def delta(self):
        return 1./self.sampling_rate

    @property
    def xdata(self):
        return self.tmin + num.arange(self.i_filled, dtype=num.float64) * self.delta

    @property
    def ydata(self):
        return self.data[:self.i_filled]

    def index_at_time(self, t):
        ''' Get the index of the sample (closest) defined by *t* '''
        return int((t-self.tmin) * self.sampling_rate)

    def latest_frame(self, seconds):
        ''' Return the latest *seconds* data from buffer as x and y data tuple.'''
        n = int(min(seconds * self.sampling_rate, self.i_filled))
        xi = num.arange(self.i_filled-n, self.i_filled) % self.data.size
        return (self._x[xi], self.data[xi])

    def latest_frame_data(self, n):
        ''' Return the latest *seconds* data from buffer as x and y data tuple.'''
        xi = num.arange(self.i_filled-n, self.i_filled) % self.data.size
        return self.data[xi]

    def append(self, d):
        ''' Append data frame *d* to Buffer'''
        n = d.shape[0]
        if self.i_filled + n > self.data.size:
            raise Exception('data overflow')
        self.data[self.i_filled:self.i_filled+n] = d
        self.i_filled += n

        data = num.fromstring(data, 'int16').reshape(self.chunksize,
                                                     self.nchannels).T

class RingBuffer(Buffer):
    def __init__(self, *args, **kwargs):
        Buffer.__init__(self, *args, **kwargs)

    def append(self, d):
        n = d.size
        xi = (self.i_filled + num.arange(n)) % self.data.size
        self.data[xi] = d
        self.i_filled += n


class DataProvider(object):
    ''' Base class defining common interface for data input to Worker'''
    def __init__(self):
        self.frames = []
        atexit.register(self.terminate)

    #def get_data(self):
    #    return self.frames

    def terminate(self):
        # cleanup
        pass


class SamplingRateException(Exception):
    pass


class MicrophoneRecorder(DataProvider):

    def __init__(self, chunksize=512, data_ready_signal=None):
        DataProvider.__init__(self)
        self.stream = None
        self.p = pyaudio.PyAudio()
        default = self.p.get_default_input_device_info()

        self.device_no = default['index']
        #self.__sampling_rate = int(default['defaultSampleRate'])
        self.__sampling_rate = int(16000)

        self.chunksize = chunksize
        self.data_ready_signal = data_ready_signal or DummySignal()
        self.nchannels = 2

    @property
    def sampling_rate_options(self):
        ''' List of supported sampling rates.'''
        return sampling_rate_options(self.device_no, audio=self.p)

    def new_frame(self, data, frame_count, time_info, status):
        #logger.debug('new data. frame count: %s, time_info:%s' % (frame_count,
        #                                                          time_info))
        data = num.fromstring(data, 'int16')
        with _lock:
            self.frames.append(data)
            if self._stop:
                return None, pyaudio.paComplete

        self.data_ready_signal.emit()

        return None, pyaudio.paContinue

    def get_frames(self):
        with _lock:
            frames = self.frames
            self.frames = []
        return frames

    def start(self):
        if self.stream is None:
            raise Exception('cannot start stream which is None')
        self.stream.start_stream()
        self._stop = False

    @property
    def sampling_rate(self):
        return self.__sampling_rate

    @sampling_rate.setter
    def sampling_rate(self, rate):
        check_sampling_rate(rate, self.device_no, audio=self.p)
        self.__sampling_rate = rate

    def start_new_stream(self):
        self.frames = []
        self.stream = self.p.open(format=pyaudio.paInt16,
                                  channels=self.nchannels,
                                  rate=self.sampling_rate,
                                  input=True,
                                  output=False,
                                  frames_per_buffer=self.chunksize,
                                  input_device_index=self.device_no,
                                  stream_callback=self.new_frame)
        self._stop = False
        logger.debug('starting new stream: %s' % self.stream)
        self.stream.start_stream()

    def stop(self):
        with _lock:
            self._stop = True
        self.stream.stop_stream()

    def close(self):
        self.stop()
        self.stream.close()

    def terminate(self):
        if self.stream:
            self.close()
        self.p.terminate()
        logger.debug('terminated stream')

    def set_device_no(self, i):
        self.close()
        self.device_no = i
        self.start_new_stream()

    @property
    def deltat(self):
        return 1./self.sampling_rate
