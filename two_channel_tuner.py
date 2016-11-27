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
FFTSIZE = 512*4
RATE= 16384*4
#RATE= 48000
DEVICENO=7
nchannels = 2
#pitch logs
global pitchlog1, pitchlog2
PITCHLOGLEN=20
pitchlog1 = np.arange(PITCHLOGLEN, dtype=np.float32)
pitchlog2 = np.arange(PITCHLOGLEN, dtype=np.float32)

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
class MicrophoneRecorder(object):
    def __init__(self, rate=RATE, chunksize=FFTSIZE):
        self.rate = rate
        self.chunksize = chunksize
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16,
                                  channels=nchannels,
                                  rate=self.rate,
                                  input=True,
                                  frames_per_buffer=self.chunksize,
                                  input_device_index=DEVICENO,
                                  stream_callback=self.new_frame)
        self.stop = False
        self.frames = []
        atexit.register(self.close)

    def new_frame(self, data, frame_count, time_info, status):
        data = np.fromstring(data, 'int16')
        with _lock:
            self.frames.append(data)
            if self.stop:
                return None, pyaudio.paComplete
        return None, pyaudio.paContinue

    def get_frames(self):
        with _lock:
            frames = self.frames
            self.frames = []
        return frames

    def start(self):
        self.stream.start_stream()

    def close(self):
        with _lock:
            self.stop = True
        self.stream.close()
        self.p.terminate()


def append_to_frame(f, d):
    i = d.shape[0]
    f[:-i] = f[i:]
    f[-i:] = d.T


class Worker(qc.QObject):
    ''' Grabbing data, working on it and saving the results'''

    signalReady = qc.pyqtSignal()

    def __init__(self, gain=4999999, *args, **kwargs):
        self.ndata_scale = 2
        qc.QObject.__init__(self, *args, **kwargs)
        self.mic = MicrophoneRecorder()

        self.autogain_checkbox = False
        self.new_pitch1 = None
        self.new_pitch2 = None

        nfft = (self.mic.chunksize* self.ndata_scale, 1./self.mic.rate)
        self.freq_vect1 = self.freq_vect2 = np.fft.rfftfreq(*nfft)
        self.current_frame1 = num.ones(FFTSIZE*self.ndata_scale, dtype=num.int)
        self.current_frame2 = num.ones(FFTSIZE*self.ndata_scale, dtype=num.int)
        print self.current_frame1.shape

        self.fft_frame1 = None
        self.fft_frame2 = None
        self.ivCents = None
        self.new_pitch1Cent = None
        self.new_pitch2Cent = None
        self.gain = gain
        self.mic.start()
        # keeps reference to mic
    def start(self):

        ''' Start a loop '''
        self.timer = qc.QTimer()
        self.timer.timeout.connect(self.work)
        self.timer.start(50)

    def work(self):
        ''' Do the work'''
        #frames = copy.deepcopy(self.mic.get_frames())
        frames = self.mic.get_frames()
        #tmp = num.empty(FFTSIZE)
        if len(frames) > 1:
            # keeps only the last frame (which contains two interleaved channels)
            buffer = frames[ -1]
            result = np.reshape(buffer, (FFTSIZE, nchannels))

            i = result[:, 0].shape[0]
            append_to_frame(self.current_frame1, result[:, 0])
            append_to_frame(self.current_frame2, result[:, 1])

            self.fft_frame1 = np.fft.rfft(self.current_frame1)

            #signal1float = num.zeros(self.current_frame1.shape,
            #                         dtype=num.float32)
            #print signal1float.dtype
            #signal1float.setflags(write=True)
            signal1float = num.asarray(self.current_frame1, dtype=num.float32)
            #signal1float = self.current_frame1.astype(num.float32)
            #signal1float = copy(self.current_frame1)
            #self.new_pitch1 = pitch_o(signal1float[-1024:])[0]
            self.fft_frame2 = np.fft.rfft(self.current_frame2)

            signal2float = self.current_frame2.astype(np.float32)
            #self.new_pitch2 = precise_pitch2 = pitch_o(signal2float[-len(signal2float)/self.ndata_scale:])[0]
            #pitch_confidence2 = pitch_o.get_confidence()

            #self.new_pitch1Cent = 1200* math.log((self.new_pitch1+.1)/120.,2)
            #self.new_pitch2Cent = 1200* math.log((self.new_pitch2+.1)/120.,2)

            #self.ivCents = abs(self.new_pitch2Cent - self.new_pitch1Cent)
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
