# -*- coding: utf-8 -*-
"""
Created on May 23 2014

@author: florian
"""
import copy
import time
import sys
import threading
import atexit
import pyaudio
import math
from aubio import pitch
import numpy as np
from PyQt5 import QtCore as qc
import matplotlib
matplotlib.use('Qt5Agg')

# import numpy as np
import mpl_toolkits.axisartist.angle_helper as angle_helper
from matplotlib.projections import PolarAxes
from mpl_toolkits.axisartist.grid_finder import (FixedLocator, MaxNLocator, DictFormatter)
#import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
#from gui import GaugeWidget

num = np

# number of samples of buffer
FFTSIZE = 512*4

RATE= 16384*4
#RATE= 48000
nchannels = 2
#pitch logs
global pitchlog1, pitchlog2
PITCHLOGLEN=20

# Pitch
tolerance = 0.8
downsample = 1
win_s = FFTSIZE // downsample # fft size
hop_s = FFTSIZE  // downsample # hop size

pitch_o = pitch("yin", win_s, hop_s, RATE)
pitch_o.set_unit("Hz")
pitch_o.set_tolerance(tolerance)

_lock = threading.Lock()

# class taken from the SciPy 2015 Vispy talk opening example
# see https://github.com/vispy/vispy/pull/928


def getaudiodevices():
    ''' Returns a list of device descriptions'''
    p = pyaudio.PyAudio()
    devices = []
    for i in range(p.get_device_count()):
        devices.append(p.get_device_info_by_index(i).get('name'))
    return devices

class MicrophoneRecorder(object):

    def __init__(self, rate=RATE, chunksize=FFTSIZE):
        getaudiodevices()
        self.rate = rate
        self.chunksize = chunksize
        self.p = pyaudio.PyAudio()
        self.device_no = 7
        self.frames = []
        self.data_ready_signal = None
        atexit.register(self.close)

    def new_frame(self, data, frame_count, time_info, status):
        data = np.fromstring(data, 'int16')
        with _lock:
            self.frames.append(data)
            if self._stop:
                return None, pyaudio.paComplete

        if self.data_ready_signal:
            self.data_ready_signal.emit()
        return None, pyaudio.paContinue

    def get_frames(self):
        with _lock:
            frames = self.frames
            self.frames = []
        return frames

    def start(self):
        self.stream = self.p.open(format=pyaudio.paInt16,
                                  channels=nchannels,
                                  rate=self.rate,
                                  input=True,
                                  frames_per_buffer=self.chunksize,
                                  input_device_index=self.device_no,
                                  stream_callback=self.new_frame)
        self.stream.start_stream()
        self._stop = False

    def stop(self):
        with _lock:
            self._stop = True
        #self.stream.stop_stream()
        self.stream.close()

    def close(self):
        self.p.terminate()


def append_to_frame(f, d):
    i = d.shape[0]
    f[:-i] = f[i:]
    f[-i:] = d.T


class Worker(qc.QObject):
    ''' Grabbing data, working on it and saving the results'''

    signalReady = qc.pyqtSignal()
    dataReady = qc.pyqtSignal()

    def __init__(self, gain=4999999, *args, **kwargs):
        self.ndata_scale = 8
        qc.QObject.__init__(self, *args, **kwargs)
        self.mic = MicrophoneRecorder()
        #self.mic.data_ready_signal = self.dataReady
        self.autogain_checkbox = False
        self.new_pitch1 = None
        self.new_pitch2 = None
        self.fftsize = FFTSIZE

        #nfft = (self.mic.chunksize* self.ndata_scale, 1./self.mic.rate)
        nfft = (self.fftsize, 1./self.mic.rate)
        self.freq_vect1 = self.freq_vect2 = np.fft.rfftfreq(*nfft)
        self.current_frame1 = num.ones(self.fftsize*self.ndata_scale, dtype=num.int)
        self.current_frame2 = num.ones(self.fftsize*self.ndata_scale, dtype=num.int)

        self.fft_frame1 = None
        self.fft_frame2 = None
        self.ivCents = None
        self.new_pitch1Cent = None
        self.new_pitch2Cent = None
        self.gain = gain
        self.mic.start()
        self.pitchlog1 = np.arange(PITCHLOGLEN)#, dtype=np.int)
        print 'pl', self.pitchlog1
        self.pitchlog2 = np.arange(PITCHLOGLEN)#, dtype=np.int)
        self.pitchlog_vect1 = np.arange(PITCHLOGLEN, dtype=np.int)
        self.pitchlog_vect2 = np.arange(PITCHLOGLEN, dtype=np.int)
        self.pitchlog_vect1[:] = num.nan
        self.pitchlog_vect2[:] = num.nan

        # keeps reference to mic

    def set_device_no(self, i):
        self.mic.device_no = i

    def start(self):
        #self.mic.data_ready.connect(self.work)
        ''' Start a loop '''
        self.mic.start()
        self.timer = qc.QTimer()
        self.timer.timeout.connect(self.work)
        self.timer.start(50)

    def stop(self):
        self.mic.stop = True
        #self.mic.close()

    def work(self):
        ''' Do the work'''
        frames = self.mic.get_frames()

        if len(frames) > 0:
            # keeps only the last frame (which contains two interleaved channels)
            buffer = frames[ -1]
            result = np.reshape(buffer, (self.fftsize, nchannels))

            i = result[:, 0].shape[0]
            append_to_frame(self.current_frame1, result[:, 0])
            append_to_frame(self.current_frame2, result[:, 1])

            self.fft_frame1 = np.fft.rfft(self.current_frame1[-self.fftsize:])

            signal1float = num.asarray(self.current_frame1, dtype=num.float32)

            self.fft_frame2 = np.fft.rfft(self.current_frame2[-self.fftsize:])

            signal1float = self.current_frame1.astype(np.float32)
            signal2float = self.current_frame2.astype(np.float32)
            self.new_pitch1 = pitch_o(signal1float[-self.fftsize:])[0]
            self.new_pitch2 = pitch_o(signal2float[-self.fftsize:])[0]
            print 'new pitch2', self.new_pitch2
            #pitch_confidence2 = pitch_o.get_confidence()

            self.new_pitch1Cent = 1200* math.log((self.new_pitch1+.1)/120.,2)
            self.new_pitch2Cent = 1200* math.log((self.new_pitch2+.1)/120.,2)
            self.pitchlog_vect1 = num.roll(self.pitchlog_vect1, 1)
            self.pitchlog_vect2 = num.roll(self.pitchlog_vect2, 1)

            self.pitchlog_vect1[-1] = self.new_pitch1Cent
            self.pitchlog_vect2[-1] = self.new_pitch2Cent
            #append_to_frame(self.pitchlog_vect1, self.new_pitch1Cent)
            #append_to_frame(self.pitchlog_vect2, self.new_pitch2Cent)

            #ivCents = abs(self.new_pitch2Cent - self.new_pitch1Cent)
            #if 0< ivCents <= 1200:
            #    plot gauge

            # plot self.new_pitcj1Cent - self.newptch2Cent und anders rum
            self.signalReady.emit()


def compute_pitch_hps(x, Fs, dF=None, Fmin=30., Fmax=900., H=5):
    # default value for dF frequency resolution
    if dF == None:
        dF = Fs / x.size

    # Hamming window apodization
    x = np.array(x, dtype=np.double, copy=True)
    x *= np.hamming(x.size)

    # number of points in FFT to reach the resolution wanted by the user
    n_fft = np.ceil(Fs / dF)

    # DFT computation
    X = np.abs(np.fft.fft(x, n=int(n_fft)))

    # limiting frequency R_max computation
    R = np.floor(1 + n_fft / 2. / H)

    # computing the indices for min and max frequency
    N_min = np.ceil(Fmin / Fs * n_fft)
    N_max = np.floor(Fmax / Fs * n_fft)
    N_max = min(N_max, R)

    # harmonic product spectrum computation
    indices = (np.arange(N_max)[:, np.newaxis] * np.arange(1, H+1)).astype(int)
    P = np.prod(X[indices.ravel()].reshape(N_max, H), axis=1)
    ix = np.argmax(P * ((np.arange(P.size) >= N_min) & (np.arange(P.size) <= N_max)))
    return dF * ix

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LiveFFTWidget()
    sys.exit(app.exec_())
