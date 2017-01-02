# -*- coding: utf-8 -*-

import time
import sys
import threading
import pyaudio
import math
from aubio import pitch
import numpy as num

from pytch.data import MicrophoneRecorder, Buffer, RingBuffer
from pytch.util import DummySignal

import logging

logger = logging.getLogger(__name__)


class Worker():

    def __init__(self, fft_size, provider, buffer_length):
        ''' Grabbing data, working on it and saving the results

        :param buffer_length: in seconds'''


        self.ndata_scale = 16*2
        self.nchannels = 2
        self.processingFinished = DummySignal()
        self.provider = provider
        self.buffer_length = buffer_length     # seconds
        self.fftsize = fft_size
        #self.cross_spectra_combinations = [(0, 1), ]
        self.cross_spectra_combinations = []
        self.setup_buffers()
        self.pmin = 1000.

    def set_data_provider(self, provider):
        self.provider = provider

    def set_fft_length(self, n):
        self.fftsize = n
        self.setup_buffers()

    def setup_buffers(self):

        p = self.provider
        self.new_pitch1 = None
        self.new_pitch2 = None

        nfft = (self.fftsize, p.deltat)
        self.freqs  = num.fft.rfftfreq(*nfft)
        self.fft_frame1 = None
        self.fft_frame2 = None
        self.ivCents = None
        self.new_pitch1Cent = None
        self.new_pitch2Cent = None
        self.frames = [RingBuffer(p.sampling_rate, self.buffer_length),
                       RingBuffer(p.sampling_rate, self.buffer_length)]

        self.ffts = num.ones((self.nchannels, len(self.freqs)))

        #self.pitchlog_vect1 = num.ones(PITCHLOGLEN, dtype=num.int)
        #self.pitchlog_vect2 = num.ones(PITCHLOGLEN, dtype=num.int)

        # get sampling rate from refresh rate
        n_pitch = int(p.sampling_rate/self.fftsize)
        self.pitchlogs = []
        for i in range(self.nchannels):
            self.pitchlogs.append(Buffer(n_pitch, self.buffer_length*n_pitch))
        self.cross_spectra = [Buffer(n_pitch, self.buffer_length*n_pitch)] * len(self.cross_spectra_combinations)
        self.cross_phases = [Buffer(n_pitch, self.buffer_length*n_pitch)] * len(self.cross_spectra_combinations)

        # keeps reference to mic
        # Pitch
        tolerance = 0.8
        downsample = 1
        win_s = self.fftsize // downsample # fft size
        hop_s = self.fftsize  // downsample # hop size
        self.pitch_o = pitch("yin", win_s, hop_s, p.sampling_rate)
        self.pitch_o.set_unit("Hz")
        self.pitch_o.set_tolerance(tolerance)

    def set_pmin(self, v):
        self.pmin = v

    def set_nfft(self, nfft):
        self.fftsize = int(nfft)

    def process(self):
        ''' Do the work'''
        logger.debug('start processing')
        frames = self.provider.get_frames()

        if len(frames) > 0:
            logger.debug('process')

            # change reshape
            result = num.reshape(frames[-1], (self.provider.chunksize,
                                          self.nchannels)).T
            for i in range(self.nchannels):
                self.frames[i].append(result[i])

            for i in range(self.nchannels):
                frame_work = self.frames[i].latest_frame_data(self.fftsize)
                self.ffts[i, :] = num.abs(num.fft.rfft(frame_work))
                total_power = num.sum(self.ffts[i, :])/len(self.freqs)
                if total_power < self.pmin:
                    new_pitch_Cent = -9999999.
                else:
                    pitch = self.pitch_o(frame_work.astype(num.float32))[0]
                    new_pitch_Cent = 1200.* math.log((pitch +.1)/120., 2)
                #new_pitch_Cent = math.log((pitch +.1)/120., 2)
                self.pitchlogs[i].append(num.array([new_pitch_Cent]))
                #new_pitch_cent = pitch


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

def cross_spectrum(spec1, spec2):
    ''' Returns cross spectrum and phase of *spec1* and *spec2*'''
    cross = spec1 * spec2.conjugate()
    return num.abs(cross), num.unwrap(num.arctan2(cross.imag, cross.real))

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
