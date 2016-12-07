# -*- coding: utf-8 -*-

import time
import sys
import threading
import pyaudio
import math
from aubio import pitch
import numpy as num
from PyQt5 import QtCore as qc

from pytch.data import MicrophoneRecorder, Buffer
from pytch.util import DummySignal

import logging

logger = logging.getLogger(__name__)


class Worker():
    ''' Grabbing data, working on it and saving the results'''

    def __init__(self, fft_size, provider):
        self.ndata_scale = 16*2
        self.nchannels = 2
        self.processingFinished = DummySignal()
        self.provider = provider
        self.buffer_length = 3*60     # seconds
        self.fftsize = fft_size
        self.setup_buffers()

    def set_data_provider(self, provider):
        self.provider = provider

    def set_fft_length(self, n):
        self.fftsize = n
        self.setup_buffers()

    def setup_buffers(self):

        p = self.provider
        self.new_pitch1 = None
        self.new_pitch2 = None

        #nfft = (self.mic.chunksize* self.ndata_scale, 1./self.mic.rate)
        nfft = (self.fftsize, p.deltat)
        self.freqs  = num.fft.rfftfreq(*nfft)
        self.fft_frame1 = None
        self.fft_frame2 = None
        self.ivCents = None
        self.new_pitch1Cent = None
        self.new_pitch2Cent = None
        #self.current_frame1 = Buffer()
        #self.current_frame2 = Buffer()
        self.frames = [Buffer(p.sampling_rate,
                             self.buffer_length)] * self.nchannels

        self.ffts = num.ones((self.nchannels, len(self.freqs)))
        #self.pitches = num.zeros((self.nchannels, self.fftsize))
        #self.current_frame1 = num.ones(self.fftsize*self.ndata_scale, dtype=num.int)
        #self.current_frame2 = num.ones(self.fftsize*self.ndata_scale, dtype=num.int)

        #self.pitchlog1 = num.arange(PITCHLOGLEN)#, dtype=num.int)
        #self.pitchlog2 = num.arange(PITCHLOGLEN)#, dtype=num.int)
        #self.pitchlog_vect1 = num.ones(PITCHLOGLEN, dtype=num.int)
        #self.pitchlog_vect2 = num.ones(PITCHLOGLEN, dtype=num.int)

        # get sampling rate from refresh rate
        PITCHLOGLEN = 40
        self.pitchlogs = [Buffer(1., PITCHLOGLEN, dtype=num.int)] * self.nchannels

        # keeps reference to mic
        # Pitch
        tolerance = 0.8
        downsample = 1
        win_s = self.fftsize // downsample # fft size
        hop_s = self.fftsize  // downsample # hop size
        self.pitch_o = pitch("yin", win_s, hop_s, p.sampling_rate)
        self.pitch_o.set_unit("Hz")
        self.pitch_o.set_tolerance(tolerance)

    def set_nfft(self, nfft):
        self.fftsize = int(nfft)

    def process(self):
        ''' Do the work'''
        logger.debug('start processing')
        frames = self.provider.get_data()

        if len(frames) > 0:
            logger.debug('process')

            # change reshape
            result = num.reshape(frames[-1], (self.provider.chunksize,
                                          self.nchannels)).T
            for i in range(self.nchannels):
                self.frames[i].append(result[i].T)

            for i in range(self.nchannels):
                frame_work = self.frames[i].latest_frame_data(self.fftsize)
                self.ffts[i, :] = num.fft.rfft(frame_work)
                pitch = self.pitch_o(frame_work.astype(num.float32))[0]
                new_pitch_Cent = 1200.* math.log((pitch +.1)/120., 2)
                self.pitchlogs[i].append(num.array([new_pitch_Cent]))

            #if len(frames) > 0:


            #    # keeps only the last frame (which contains two interleaved channels)
            #    #buffer = frames[ -1]
            #    #result = num.reshape(buffer, (self.mic.chunksize, self.nchannels))
            #    #i = result[:, 0].shape[0]
            #    append_to_frame(self.current_frame1, result[:, 0])
            #    append_to_frame(self.current_frame2, result[:, 1])

            #    self.fft_frame1 = num.fft.rfft(self.current_frame1[-self.fftsize:])
            #    self.fft_frame2 = num.fft.rfft(self.current_frame2[-self.fftsize:])

            #    signal1float = self.current_frame1.astype(num.float32)
            #    signal2float = self.current_frame2.astype(num.float32)
            #    self.new_pitch1 = self.pitch_o(signal1float[-self.fftsize:])[0]
            #    self.new_pitch2 = self.pitch_o(signal2float[-self.fftsize:])[0]
            #    #pitch_confidence2 = pitch_o.get_confidence()

            #    self.pitchlog_vect1 = num.roll(self.pitchlog_vect1, -1)
            #    self.pitchlog_vect2 = num.roll(self.pitchlog_vect2, -1)

            #    self.new_pitch1Cent = 1200.* math.log((self.new_pitch1+.1)/120., 2)
            #    self.new_pitch2Cent = 1200.* math.log((self.new_pitch2+.1)/120., 2)

            #    self.pitchlog_vect1[-1] = self.new_pitch1Cent
            #    self.pitchlog_vect2[-1] = self.new_pitch2Cent

            #    #ivCents = abs(self.new_pitch2Cent - self.new_pitch1Cent)
            #    #if 0< ivCents <= 1200:
            #    #    plot gauge

            #    # plot self.new_pitcj1Cent - self.newptch2Cent und anders rum
            #    #self.signalReady.emit()
            self.processingFinished.emit()

            logger.debug('finished processing')

def compute_pitch_hps(x, Fs, dF=None, Fmin=30., Fmax=900., H=5):
    # default value for dF frequency resolution
    if dF == None:
        dF = Fs / x.size

    # Hamming window apodization
    x = num.array(x, dtype=num.double, copy=True)
    x *= num.hamming(x.size)

    # number of points in FFT to reach the resolution wanted by the user
    n_fft = num.ceil(Fs / dF)

    # DFT computation
    X = num.abs(num.fft.fft(x, n=int(n_fft)))

    # limiting frequency R_max computation
    R = num.floor(1 + n_fft / 2. / H)

    # computing the indices for min and max frequency
    N_min = num.ceil(Fmin / Fs * n_fft)
    N_max = num.floor(Fmax / Fs * n_fft)
    N_max = min(N_max, R)

    # harmonic product spectrum computation
    indices = (num.arange(N_max)[:, num.newaxis] * num.arange(1, H+1)).astype(int)
    P = num.prod(X[indices.ravel()].reshape(N_max, H), axis=1)
    ix = num.argmax(P * ((num.arange(P.size) >= N_min) & (num.arange(P.size) <= N_max)))
    return dF * ix

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LiveFFTWidget()
    sys.exit(app.exec_())
