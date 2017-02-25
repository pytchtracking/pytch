import os
import threading
import atexit
import numpy as num
import logging
import pyaudio

from scipy.io import wavfile
from aubio import pitch
from pytch.kalman import Kalman


_lock = threading.Lock()

# class taken from the scipy 2015 vispy talk opening example
# see https://github.com/vispy/vispy/pull/928

logger = logging.getLogger(__name__)

pitch_algorithms = [
    'default', 'schmitt', 'fcomb', 'mcomb', 'specacf', 'yin', 'yinfft']


def getaudiodevices():
    ''' returns a list of device descriptions'''
    p = pyaudio.PyAudio()
    devices = []
    for i in range(p.get_device_count()):
        devices.append(p.get_device_info_by_index(i).get('name'))
    p.terminate()
    return devices


def sampling_rate_options(device_no, audio=None):
    ''' list of supported sampling rates.'''
    candidates = [8000., 11.025, 123123123123., 16000., 22050., 32000., 37.800,
                  44100., 48000.]
    supported_sampling_rates = []
    for c in candidates:
        if check_sampling_rate(device_no, int(c), audio=audio):
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
            input_channels=devinfo['maxinputchannels'],
            input_format=pyaudio.paint16)
    except ValueError as e:
        logger.debug(e)
        valid = False

    finally:
        if not audio:
            p.terminate()
        return valid


class Buffer():

    ''' data container

    new data is prepended, so that the latest data point is in self.data[0]'''
    def __init__(self, sampling_rate, buffer_length_seconds, dtype=num.float32,
                 tmin=0):
        self.tmin = tmin
        self.tmax = self.tmin + buffer_length_seconds
        self.sampling_rate = sampling_rate
        self.data_len = int(buffer_length_seconds * sampling_rate)
        self.dtype = dtype
        self.empty()

        self.i_filled = 0
        self._x = num.arange(self.data_len, dtype=self.dtype) * self.delta + self.tmin

    def empty(self):
        self.data = num.empty((int(self.data_len)),
                          dtype=self.dtype)

    def save_as(self, fn, fmt='txt'):
        fn = fn + '.' + fmt
        if fmt == 'txt':
            num.savetxt(fn, num.vstack((self.xdata, self.ydata)).T)
        elif fmt ==  'mseed':
            fn = os.path.join(fn, '.' + fmt)
            try:
                from pyrocko import trace, io
            except ImportError as e:
                logger.warn('no pyrocko installation found!')
                return

            tr = trace.Trace(tmin=self.tmin, deltat=self.deltat, ydata=self.ydata)
            io.save([tr], fn)

        elif fmt == 'wav':
            wavfile.write(fn, self.sampling_rate,
                          num.asarray(self.ydata, dtype=num.int16))
        logger.info('Saved file in %s' % fn)

    @property
    def t_filled(self):
        ''' the time to which the data buffer contains data.'''
        return self.tmin + self.i_filled*self.delta

    @property
    def delta(self):
        return 1./self.sampling_rate

    @property
    def ydata(self):
        return self.data[:self.i_filled]

    @property
    def xdata(self):
        return self._x[:self.i_filled]

    def index_at_time(self, t):
        ''' Get the index of the sample (closest) defined by *t* '''
        return int((t-self.tmin) * self.sampling_rate)

    def latest_indices(self, seconds):
        return self.i_filled-int(min(
            seconds * self.sampling_rate, self.i_filled)), self.i_filled

    def latest_frame(self, seconds):
        ''' Return the latest *seconds* data from buffer as x and y data tuple.'''
        istart, istop = self.latest_indices(seconds)
        return (self._x[istart: istop], self.data[istart: istop])

    def latest_frame_data(self, n):
        ''' Return the latest n samples data from buffer as array.'''
        return self.data[max(self.i_filled-n, 0): self.i_filled]

    def append(self, d):
        ''' Append data frame *d* to Buffer'''
        n = d.shape[0]
        self.data[self.i_filled:self.i_filled+n] = d
        self.i_filled += n

    def append_value(self, v):
        self.data[self.i_filled+1] = v
        self.i_filled += 1

    #def energy(self, nsamples_total, nsamples_sum=1):
    #    xi = num.arange(self.i_filled-nsamples_total, self.i_filled)
    #    y = self.data[xi].reshape((int(len(xi)/nsamples_sum), nsamples_sum))
    #    y = num.sum(y**2, axis=1)
    #    return self._x[xi[::nsamples_sum]], y



class RingBuffer(Buffer):
    ''' Based on numpy'''
    def __init__(self, *args, **kwargs):
        Buffer.__init__(self, *args, **kwargs)

    def append(self, d):
        '''append new data d to buffer f'''
        n = d.size
        if n == 1:
            self.append_value(d)
            return

        i_filled_mod = self.i_filled % self.data_len
        istop = i_filled_mod + n
        if istop >= self.data_len:
            istop_wrap = istop - self.data_len
            iwrap = n - istop_wrap
            self.data[i_filled_mod: ] = d[: iwrap]
            self.data[0: istop_wrap] = d[iwrap :]
        else:
            self.data[i_filled_mod: istop] = d

        self.i_filled += n

    def append_value(self, v):
        self.data[self.i_filled % self.data_len] = v
        self.i_filled += 1

    def latest_frame_data(self, n):
        ''' Return the latest n samples data from buffer as array.'''
        n %= self.data_len
        i_filled = self.i_filled % self.data_len
        if n > i_filled:
            return num.roll(self.data, -i_filled)[-n:]
        else:
            return self.data[i_filled-n: i_filled]

    def latest_frame(self, seconds, clip_min=False):
        ''' Return the latest *seconds* data from buffer as x and y data tuple.'''
        istart, istop = self.latest_indices(seconds)
        n = int(seconds*self.sampling_rate)+1
        x = self.i_filled/self.sampling_rate - self._x[:n][::-1]
        if clip_min:
            istart = num.min(num.where(x>0))
        else:
            istart = 0
        return (x[istart:], self.latest_frame_data(n-istart))

class RingBuffer2D(RingBuffer):
    def __init__(self, ndimension2, *args, **kwargs):
        self.ndimension2 = ndimension2
        RingBuffer.__init__(self, *args, **kwargs)

    def empty(self):
        self.data = num.empty((int(self.data_len), int(self.ndimension2)),
                          dtype=self.dtype)

    def append(self, d):
        if len(d.shape) == 1:
            self.append_value(d)
            return

        n, n2 = d.shape
        if n2 != self.ndimension2:
            raise Exception('ndim2 wrong')

        istop = (self.i_filled + n)
        if istop >= self.data_len:
            istop %= self.data_len
            iwrap = n - istop
            self.data[self.i_filled:] = d[: iwrap]
            self.data[0: istop] = d[iwrap :]
        else:
            self.data[self.i_filled: istop] = d

        self.i_filled = istop

    def append_value(self, v):
        self.i_filled += 1
        self.i_filled %= self.data_len
        self.data[self.i_filled, :] = v

    def latest_frame_data(self, n):
        ''' Return the latest n samples data from buffer as array.'''
        #return self.data[max(self.i_filled-n, 0): self.i_filled]
        return num.roll(self.data, -self.i_filled, 0)[-n:]


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


class Channel(RingBuffer):
    def __init__(self, sampling_rate, fftsize=8192):

        self.buffer_length_seconds = 14
        RingBuffer.__init__(self, sampling_rate, self.buffer_length_seconds)

        self.__pitch_algorithm = 'yinfft'
        self.name = ''
        self.pitch_o = None
        self.fftsize = fftsize
        self.setup_pitch()
        self.update()
        P = 0.
        R = 0.01**2
        Q = 1e-6
        self.kalman_pitch_filter = Kalman(P, R, Q)

    def update(self):
        nfft = (int(self.fftsize), self.delta)
        self.freqs = num.fft.rfftfreq(*nfft)
        sr = int(1000./58.)
        # TODO: 58=gui refresh rate. Nastily hard coded here for now
        self.fft = RingBuffer2D(
            ndimension2=self.fftsize/2+1,
            # sampling_rate=self.sampling_rate/self.fftsize,   # Hop size
            sampling_rate = sr,
            buffer_length_seconds=self.buffer_length_seconds)
        self.fft_power = RingBuffer(
            sampling_rate=sr,
            buffer_length_seconds=self.buffer_length_seconds)
        self.pitch = RingBuffer(
            sampling_rate=sr,
            buffer_length_seconds=self.sampling_rate*self.buffer_length_seconds/self.fftsize)

    def append_value_pitch(self, val):
        ''' Append a new pitch value to pitch buffer. Apply Kalman filter
        before appending'''
        self.pitch.append_value(
            self.kalman_pitch_filter.evaluate(val)
        )

    @property
    def fftsize(self):
        return self.__fftsize

    @fftsize.setter
    def fftsize(self, size):
        self.__fftsize = size
        self.update()

    @property
    def pitch_algorithm(self):
        return self.__pitch_algorithm

    @pitch_algorithm.setter
    def pitch_algorithm(self, alg):
        self.__algorithm = alg
        self.update()
        self.setup_pitch()

    def get_latest_pitch(self, standard_frequency):
        #return f2pitch(self.pitch.latest_frame_data(1), standard_frequency)
        return self.pitch.latest_frame_data(1)

    def setup_pitch(self):
        if self.pitch_o:
            self.pitch_o = None
        tolerance = 0.8
        win_s = self.fftsize
        self.pitch_o = pitch(self.pitch_algorithm,
          win_s, win_s, self.sampling_rate)
        self.pitch_o.set_unit("Hz")
        self.pitch_o.set_tolerance(tolerance)


class MicrophoneRecorder(DataProvider):

    def __init__(self, chunksize=512, device_no=None, sampling_rate=None, fftsize=1024,
                 nchannels=2):
        DataProvider.__init__(self)

        self.stream = None
        self.p = pyaudio.PyAudio()
        self.nchannels = nchannels
        default = self.p.get_default_input_device_info()

        self.device_no = device_no or default['index']
        self.sampling_rate = sampling_rate or int(default['defaultSampleRate'])

        self.channels = []
        for i in range(self.nchannels):
            c = Channel(self.sampling_rate, fftsize=fftsize)
            self.channels.append(c)

        self.chunksize = chunksize

    @property
    def fftsizes(self):
        ''' List of sampling rates of all channels registered by the input
        device'''
        return [c.fftsize for c in self.channels]

    @property
    def sampling_rate_options(self):
        ''' List of supported sampling rates.'''
        return sampling_rate_options(self.device_no, audio=self.p)

    def new_frame(self, data, frame_count, time_info, status):
        data = num.asarray(num.fromstring(data, 'int16'), num.float32)

        with _lock:
            self.frames.append(data)
            if self._stop:
                return None, pyaudio.paComplete

        return None, pyaudio.paContinue

    def get_frames(self):
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
        check_sampling_rate(self.device_no, rate, audio=self.p)
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
        if self.stream is not None:
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

    def flush(self):
        ''' read data and put it into channels' track_data'''
        # make this entirely numpy:
        frames = num.array(self.get_frames())
        for frame in frames:
            r = num.reshape(frame, (self.chunksize,
                                    self.nchannels)).T
            for i, channel in enumerate(self.channels):
                channel.append(r[i])

    def play_sound(self, freq):
        ''' keyboard output '''
        print('play frequency: ', freq)
