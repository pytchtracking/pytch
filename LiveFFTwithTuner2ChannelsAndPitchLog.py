# -*- coding: utf-8 -*-
"""
Created on May 23 2014

@author: florian
"""
import time
import sys
import threading
import atexit
import pyaudio
import math
from aubio import pitch
import numpy as np
import matplotlib.pyplot as plt
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QMenu, QHBoxLayout, QVBoxLayout, QSizePolicy, QMessageBox, QWidget, QLabel, QSlider, QCheckBox, QPushButton, QLCDNumber
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

DEVICENO=2

FFTSIZE=2048
RATE= 16384
MAXPLOTFREQ = 2000

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

pitch_o = pitch("yinfft", win_s, hop_s, RATE)
pitch_o.set_unit("Hz")
pitch_o.set_tolerance(tolerance)


# class taken from the SciPy 2015 Vispy talk opening example
# see https://github.com/vispy/vispy/pull/928
class MicrophoneRecorder(object):
    def __init__(self, rate=RATE, chunksize=FFTSIZE):
        self.rate = rate
        self.chunksize = chunksize
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16,
                                  channels=2,
                                  rate=self.rate,
                                  input=True,
                                  frames_per_buffer=self.chunksize,
                                  input_device_index=DEVICENO,
                                  stream_callback=self.new_frame)
        self.lock = threading.Lock()
        self.stop = False
        self.frames = []
        atexit.register(self.close)

    def new_frame(self, data, frame_count, time_info, status):
        data = np.fromstring(data, 'int16')
        with self.lock:
            self.frames.append(data)
            if self.stop:
                return None, pyaudio.paComplete
        return None, pyaudio.paContinue

    def get_frames(self):
        with self.lock:
            frames = self.frames
            self.frames = []
            return frames

    def start(self):
        self.stream.start_stream()

    def close(self):
        with self.lock:
            self.stop = True
        self.stream.close()
        self.p.terminate()


class MplFigure(object):
    def __init__(self, parent):
        self.figure = plt.figure(facecolor='white')
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, parent)

class LiveFFTWidget(QWidget):
    def __init__(self):
        QWidget.__init__(self)

        # customize the UI
        self.initUI()

        # init class data
        self.initData()

        # connect slots
        self.connectSlots()

        # init MPL widget
        self.initMplWidget()

    def initUI(self):

        hbox_gain = QHBoxLayout()
        autoGain = QLabel('Auto gain')
        autoGainCheckBox = QCheckBox(checked=True)
        hbox_gain.addWidget(autoGain)
        hbox_gain.addWidget(autoGainCheckBox)

        # reference to checkbox
        self.autoGainCheckBox = autoGainCheckBox

        hbox_fixedGain = QHBoxLayout()
        fixedGain = QLabel('Fixed gain level')
        fixedGainSlider = QSlider(QtCore.Qt.Horizontal)
        hbox_fixedGain.addWidget(fixedGain)
        hbox_fixedGain.addWidget(fixedGainSlider)

        self.fixedGainSlider = fixedGainSlider

        vbox = QVBoxLayout()

        vbox.addLayout(hbox_gain)
        vbox.addLayout(hbox_fixedGain)

        # mpl figure
        self.main_figure = MplFigure(self)
        vbox.addWidget(self.main_figure.toolbar)
        vbox.addWidget(self.main_figure.canvas)

        self.setLayout(vbox)

        self.setGeometry(300, 300, 350, 300)
        self.setWindowTitle('LiveFFTwithTuner2ChannelsAndPitchLog')
        self.show()
        # timer for calls, taken from:
        # http://ralsina.me/weblog/posts/BB974.html
        timer = QtCore.QTimer()
        timer.timeout.connect(self.handleNewData)
        #timer.start(100)
        timer.start(1)
        # keep reference to timer
        self.timer = timer


    def initData(self):
        mic = MicrophoneRecorder()
        mic.start()

        # keeps reference to mic
        self.mic = mic

        # computes the parameters that will be used during plotting
        self.freq_vect1 = np.fft.rfftfreq(mic.chunksize,
                                         1./mic.rate)
        self.time_vect1 = np.arange(mic.chunksize, dtype=np.float32) / mic.rate * 1000

        self.freq_vect2 = np.fft.rfftfreq(mic.chunksize,
                                              1./mic.rate)
        self.time_vect2 = np.arange(mic.chunksize, dtype=np.float32) / mic.rate * 1000
    
        self.pitchlog_vect1 = np.arange(PITCHLOGLEN, dtype=np.float32)
    
        self.pitchlog_vect2 = np.arange(PITCHLOGLEN, dtype=np.float32)
    
        self.intervallog_vect1 = np.arange(PITCHLOGLEN, dtype=np.float32)

    def connectSlots(self):
        pass

    def initMplWidget(self):
        """creates initial matplotlib plots in the main window and keeps
        references for further use"""
        # channel 1 time
        self.ax_ch1_time = self.main_figure.figure.add_subplot(611)
        self.ax_ch1_time.set_ylim(-32768, 32768)
        self.ax_ch1_time.set_xlim(0, self.time_vect1.max())
        self.ax_ch1_time.set_xlabel(u'time (ms)', fontsize=6)
        #pos = [pos1.x0, pos1.y0, pos1.width, pos1.height]
        #self.ax_ch1_time.set_position([0.1,0.9, 0.8, 0.08])
        self.ax_ch1_time.set_position([0.1, 0.8, 0.3, 0.15])
        
        # channel 1 spec
        self.ax_ch1_spec = self.main_figure.figure.add_subplot(612)
        #self.ax_ch1_spec.set_ylim(0, 1)
        self.ax_ch1_spec.set_ylim(-5, 0)
        #self.ax_ch1_spec.set_xlim(0, self.freq_vect1.max())
        self.ax_ch1_spec.set_xlim(0, MAXPLOTFREQ)
        #self.ax_ch1_spec.set_xlabel(u'frequency (Hz)\n', fontsize=6)
        #self.ax_ch1_spec.set_position([0.1,0.66, 0.8, 0.15])
        self.ax_ch1_spec.set_position([0.45,0.8, 0.4, 0.15])
        
        # channel 2 spec
        self.ax_ch2_spec = self.main_figure.figure.add_subplot(613)
        #self.ax_ch2_spec.set_ylim(0, 1)
        self.ax_ch2_spec.set_ylim(-5, 0)
        #self.ax_ch2_spec.set_xlim(0, self.freq_vect1.max())
        self.ax_ch2_spec.set_xlim(0, MAXPLOTFREQ)
        self.ax_ch2_spec.set_xlabel(u'frequency (Hz)\n', fontsize=6)
        #self.ax_ch2_spec.set_position([0.1,0.45, 0.8, 0.15])
        self.ax_ch2_spec.set_position([0.45,0.6, 0.4, 0.15])
        
        # channel 2 time
        self.ax_ch2_time = self.main_figure.figure.add_subplot(614)
        self.ax_ch2_time.set_ylim(-32768, 32768)
        self.ax_ch2_time.set_xlim(0, self.time_vect1.max())
        self.ax_ch2_time.set_xlabel(u'time (ms)', fontsize=6)
        #self.ax_ch2_time.set_position([0.1,0.31, 0.8, 0.08])
        self.ax_ch2_time.set_position([0.1,0.6, 0.3, 0.15])

        # pitch log plot
        self.ax_pitchlogs = self.main_figure.figure.add_subplot(615)
        self.ax_pitchlogs.set_ylim(-100,2400)
        self.ax_pitchlogs.set_xlim(0, self.pitchlog_vect1.max())
        self.ax_pitchlogs.set_xlabel(u'pitch index ', fontsize=6)
        self.ax_pitchlogs.set_position([0.1,0.4, 0.75, 0.15])

        # pitch interval log plot
        y_ticks = [0,100,200,300,400,500,700,800,900,1000,1100,1200]
        y_ticklabels=["U","2m","2M","3m","3M","4","5","6m","6M","7m","7M","8ve"]
        self.ax_intervallog = self.main_figure.figure.add_subplot(616)
        self.ax_intervallog.yaxis.tick_right()
        self.ax_intervallog.yaxis.set_label_position("right")
        self.ax_intervallog.set_yticks(y_ticks)
        self.ax_intervallog.set_yticklabels(y_ticklabels)
        self.ax_intervallog.set_ylim(-100,1200)
        self.ax_intervallog.set_xlim(0, self.pitchlog_vect1.max())
        self.ax_intervallog.set_xlabel(u'pitch index ', fontsize=6)
        self.ax_intervallog.set_position([0.1,0.1, 0.75, 0.25])
        
        # mark the 3rd, 4th and fifth
        self.ax_intervallog.plot(self.pitchlog_vect1,0 * np.ones_like(self.pitchlog_vect1),color='black',lw=2,ls='dotted')
        self.ax_intervallog.plot(self.pitchlog_vect1,100 * np.ones_like(self.pitchlog_vect1),color='blue',ls='dotted')
        self.ax_intervallog.plot(self.pitchlog_vect1,200 * np.ones_like(self.pitchlog_vect1),color='blue',ls='dashed')
        self.ax_intervallog.plot(self.pitchlog_vect1,300 * np.ones_like(self.pitchlog_vect1),color='green',ls='dotted')
        self.ax_intervallog.plot(self.pitchlog_vect1,400 * np.ones_like(self.pitchlog_vect1),color='green',ls='dashed')
        self.ax_intervallog.plot(self.pitchlog_vect1,500 * np.ones_like(self.pitchlog_vect1),color='black',lw=4,ls='dotted')
        self.ax_intervallog.plot(self.pitchlog_vect1,700 * np.ones_like(self.pitchlog_vect1),color='black',lw=4,ls='dotted')
        self.ax_intervallog.plot(self.pitchlog_vect1,800 * np.ones_like(self.pitchlog_vect1),color='green',ls='dotted')
        self.ax_intervallog.plot(self.pitchlog_vect1,900 * np.ones_like(self.pitchlog_vect1),color='green',ls='dashed')
        self.ax_intervallog.plot(self.pitchlog_vect1,1000 * np.ones_like(self.pitchlog_vect1),color='blue',ls='dotted')
        self.ax_intervallog.plot(self.pitchlog_vect1,1100 * np.ones_like(self.pitchlog_vect1),color='blue',ls='dashed')


        
        # line objects
        self.line_ch1_time, = self.ax_ch1_time.plot(self.time_vect1,
                                         np.ones_like(self.time_vect1),color='red')
        self.line_ch1_spec, = self.ax_ch1_spec.plot(self.freq_vect1,
                                               np.ones_like(self.freq_vect1),color='red')
        self.ch1_pitch_line, = self.ax_ch1_spec.plot((self.freq_vect1[self.freq_vect1.size / 2], self.freq_vect1[self.freq_vect1.size / 2]),
                                              self.ax_ch1_spec.get_ylim(),color='black', lw=2)

        self.line_ch2_time, = self.ax_ch2_time.plot(self.time_vect2,
                                            np.ones_like(self.time_vect2),color='green')
        self.line_ch2_spec, = self.ax_ch2_spec.plot(self.freq_vect2,
                                                np.ones_like(self.freq_vect2),color='green')
        self.ch2_pitch_line, = self.ax_ch2_spec.plot((self.freq_vect2[self.freq_vect2.size / 2], self.freq_vect2[self.freq_vect2.size / 2]),
                                                     self.ax_ch2_spec.get_ylim(), color='black',lw=2)

        # pitch log
        self.ch1_pitchlog, = self.ax_pitchlogs.plot(self.pitchlog_vect1,np.ones_like(self.pitchlog_vect1),color='red',lw=1,ls ='dotted',marker="o")
        self.ch2_pitchlog, = self.ax_pitchlogs.plot(self.pitchlog_vect2, np.ones_like(self.pitchlog_vect2),color='green',lw=1,ls ='dashed',marker="o")
    
        # interval log
        self.intervallog1, = self.ax_intervallog.plot(self.pitchlog_vect1,np.ones_like(self.pitchlog_vect1),color='blue',lw=2,marker="h")
    
        # tight layout
        #plt.tight_layout()

    def handleNewData(self):
        """ handles the asynchroneously collected sound chunks """
        # gets the latest frames
        frames = self.mic.get_frames()

        if len(frames) > 0:
            # keeps only the last frame (which contains two interleaved channels)
            t0g = time.time()
            buffer = frames[-1]
            result = np.reshape(buffer, (FFTSIZE, 2))
            current_frame1 = result[:, 0]
            current_frame2 = result[:, 1]
            time_str = "Time to load frame1 and frame2 in ms: %f" % ((time.time()-t0g)*1000)
            #print("{}".format(time_str))
            # channel 1
            # plots the time signal 1
            self.line_ch1_time.set_data(self.time_vect1, current_frame1)
            # computes and plots the fft signal
            fft_frame = np.fft.rfft(current_frame1)
            if self.autoGainCheckBox.checkState() == QtCore.Qt.Checked:
                fft_frame /= np.abs(fft_frame).max()
            else:
                fft_frame *= (1 + self.fixedGainSlider.value()) / 5000000.
                #print(np.abs(fft_frame).max())
            #self.line_ch1_spec.set_data(self.freq_vect1, np.abs(fft_frame))
            self.line_ch1_spec.set_data(self.freq_vect1, np.log(np.abs(fft_frame) +.001))
            
            #time_str = "Time to plot channel1 in ms: %f" % ((time.time()-t0g)*1000)
            #print("{}".format(time_str))
            #  pitch tracking algorithm
            
            t0g = time.time()
            #new_pitch1 = compute_pitch_hps(current_frame1, self.mic.rate,dF=1)
            #precise_pitch1 = compute_pitch_hps(current_frame1, self.mic.rate, dF=0.05, Fmin=new_pitch1 * 0.8, Fmax = new_pitch1 * 1.2)
            #time_str = "Time to pitch track channel1 in ms: %f" % ((time.time()-t0g)*1000)
            #print("{}".format(time_str))
            
            signal1float=current_frame1.astype(np.float32)
            new_pitch1=precise_pitch1=pitch_o(signal1float)[0]
            pitch_confidence1 = pitch_o.get_confidence()
            
            #self.ax_ch1_spec.set_title("pitch = {:.2f}  {:.3f} Hz".format(precise_pitch1,pitch_confidence1))
            self.ax_ch1_spec.set_title("pitch (Hz), conf = {:.2f} {:.3f}".format(precise_pitch1,pitch_confidence1))
            self.ch1_pitch_line.set_data((new_pitch1, new_pitch1),
                                     self.ax_ch1_spec.get_ylim())

            # channel 2
            # plots the time signal 2
            self.line_ch2_time.set_data(self.time_vect2, current_frame2)
            # computes and plots the fft signal
            fft_frame = np.fft.rfft(current_frame2)
            if self.autoGainCheckBox.checkState() == QtCore.Qt.Checked:
                fft_frame /= np.abs(fft_frame).max()
            else:
                fft_frame *= (1 + self.fixedGainSlider.value()) / 5000000.
                #print(np.abs(fft_frame).max())
            #self.line_ch2_spec.set_data(self.freq_vect2, np.abs(fft_frame))
            self.line_ch2_spec.set_data(self.freq_vect1, np.log(np.abs(fft_frame)+.001))

            #  pitch tracking algorithm
            #new_pitch2 = compute_pitch_hps(current_frame2, self.mic.rate,dF=1)
            #precise_pitch2 = compute_pitch_hps(current_frame2, self.mic.rate,dF=0.05, Fmin=new_pitch2 * 0.8, Fmax = new_pitch2 * 1.2)
                                       
            signal2float=current_frame2.astype(np.float32)
            new_pitch2=precise_pitch2=pitch_o(signal2float)[0]
            pitch_confidence2 = pitch_o.get_confidence()
                                       
            #self.ax_ch2_spec.set_title("pitch = {:.2f} Hz".format(precise_pitch2))
            self.ax_ch2_spec.set_title("pitch (Hz), conf = {:.2f} {:.3f}".format(precise_pitch2,pitch_confidence2))
            self.ch2_pitch_line.set_data((new_pitch2, new_pitch2),
                                         self.ax_ch2_spec.get_ylim())


            new_pitch1Cent = 1200* math.log((new_pitch1+.1)/120.,2)
            new_pitch2Cent = 1200* math.log((new_pitch2+.1)/120.,2)
            
            if pitch_confidence1 > 0.5:
                update_pitch_log1(abs(new_pitch1Cent))
                self.ch1_pitchlog.set_data(self.pitchlog_vect1, pitchlog1)

            if pitch_confidence2 > 0.5:
                update_pitch_log2(abs(new_pitch2Cent))
            
            self.ch2_pitchlog.set_data(self.pitchlog_vect2, pitchlog2)
            
            #interval log
            if pitch_confidence1 > 0.5 and pitch_confidence2 > 0.5:
                self.intervallog1.set_data(self.pitchlog_vect2, abs(pitchlog2-pitchlog1))
            
            # refreshes the plots
            self.main_figure.canvas.draw()


def update_pitch_log1(pitch):
    global pitchlog1
    current_pitchlog = pitchlog1
    shifted_pitchlog= np.concatenate((current_pitchlog[1:], current_pitchlog[:1]), axis=0)
    shifted_pitchlog[-1] = pitch
    pitchlog1 = shifted_pitchlog
    return

def update_pitch_log2(pitch):
    global pitchlog2
    current_pitchlog = pitchlog2
    shifted_pitchlog= np.concatenate((current_pitchlog[1:], current_pitchlog[:1]), axis=0)
    shifted_pitchlog[-1] = pitch
    pitchlog2 = shifted_pitchlog
    return

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
