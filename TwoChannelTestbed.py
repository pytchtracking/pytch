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

from matplotlib.transforms import Affine2D
import mpl_toolkits.axisartist.floating_axes as floating_axes
#import numpy as np
import mpl_toolkits.axisartist.angle_helper as angle_helper
from matplotlib.projections import PolarAxes
from mpl_toolkits.axisartist.grid_finder import (FixedLocator, MaxNLocator, DictFormatter)
#import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

FFTSIZE=2048
RATE= 16384
#RATE= 48000
DEVICENO=2

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
        self.setWindowTitle('2 Channel Testbed')
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
        self.ax_ch1_time.set_position([0.1, 0.8, 0.3, 0.15])
        
        # channel 2 time
        self.ax_ch2_time = self.main_figure.figure.add_subplot(614)
        self.ax_ch2_time.set_ylim(-32768, 32768)
        self.ax_ch2_time.set_xlim(0, self.time_vect1.max())
        self.ax_ch2_time.set_xlabel(u'time (ms)', fontsize=6)
        self.ax_ch2_time.set_position([0.1,0.6, 0.3, 0.15])
        
        # channel 1 spec
        self.ax_ch1_spec = self.main_figure.figure.add_subplot(612)
        #self.ax_ch1_spec.set_ylim(0, 1)
        self.ax_ch1_spec.set_ylim(-5,0)
        #self.ax_ch1_spec.set_xlim(0, self.freq_vect1.max())
        self.ax_ch1_spec.set_xlim(0, 1000)
        #self.ax_ch1_spec.set_xlabel(u'frequency (Hz)\n', fontsize=6)
        self.ax_ch1_spec.set_position([0.45,0.8, 0.4, 0.15])
        
        # channel 2 spec
        self.ax_ch2_spec = self.main_figure.figure.add_subplot(613)
        #self.ax_ch2_spec.set_ylim(0, 1)
        self.ax_ch2_spec.set_ylim(-5,0)
        #self.ax_ch2_spec.set_xlim(0, self.freq_vect1.max())
        self.ax_ch2_spec.set_xlim(0, 1000)
        self.ax_ch2_spec.set_xlabel(u'frequency (Hz)\n', fontsize=6)
        self.ax_ch2_spec.set_position([0.45,0.6, 0.4, 0.15])
        

        
        # gauges
        # PolarAxes.PolarTransform takes radian. However, we want our coordinate
        # system in degree
        tr = Affine2D().scale(np.pi/180., 1.) + PolarAxes.PolarTransform()
    
        # Find grid values appropriate for the coordinate (degree).
        # The argument is an approximate number of grids.
    
        grid_locator1 = angle_helper.LocatorD(2)
        # And also use an appropriate formatter:
        tick_formatter1 = angle_helper.FormatterDMS()
    
    
        angle_ticks = [(0*180/1200, "8ve"),(100*180/1200, "7M"),(200*180/1200, "7m"),(300*180/1200, "6M"),(400*180/1200, "6m"),(500*180/1200, "5"),
                   (700*180/1200, "4"),(800*180/1200, "3M"),(900*180/1200, "3m"),(1000*180/1200, "2M"),(1100*180/1200, "2m"),(1200*180/1200, "U")]

        grid_locator1 = FixedLocator([v for v, s in angle_ticks])
        tick_formatter1 = DictFormatter(dict(angle_ticks))
                   
                   
        # set up number of ticks for the r-axis
        grid_locator2 = MaxNLocator(1)
        radius_ticks = [(0, ""),(1, "")]
        tick_formatter2 = DictFormatter(dict(radius_ticks))
                   
        # the extremes are passed to the function
        grid_helper = floating_axes.GridHelperCurveLinear(tr,
                                                                     extremes=(0, 180, 0.55, 1.),
                                                                     grid_locator1=grid_locator1,
                                                                     grid_locator2=grid_locator2,
                                                                     tick_formatter1=tick_formatter1,
                                                                     tick_formatter2=tick_formatter2
                                                                     )
        
        
        #self.gauge1 = self.main_figure.figure.add_subplot(514)
        # self.gauge1_ax1 used to be called ax1
        self.gauge1_ax1 = floating_axes.FloatingSubplot(self.main_figure.figure,615, grid_helper=grid_helper)
        self.main_figure.figure.add_subplot(self.gauge1_ax1)
        self.gauge1_ax1.set_position([0.35,0.05, 0.3, 0.3])
        pllabel="Interval (cents)"
        
        # adjust axis
        self.gauge1_ax1.axis["left"].set_axis_direction("bottom")
        self.gauge1_ax1.axis["right"].set_axis_direction("top")
    
        self.gauge1_ax1.axis["bottom"].set_visible(True)
        self.gauge1_ax1.axis["bottom"].set_axis_direction("top")
        self.gauge1_ax1.axis["bottom"].major_ticklabels.set_axis_direction("bottom")
        self.gauge1_ax1.axis["bottom"].label.set_axis_direction("bottom")
    
        self.gauge1_ax1.axis["top"].set_axis_direction("bottom")
        self.gauge1_ax1.axis["top"].toggle(ticklabels=True, label=True)
        self.gauge1_ax1.axis["top"].major_ticklabels.set_axis_direction("top")
        self.gauge1_ax1.axis["top"].label.set_axis_direction("top")
    
        #ax1.axis["left"].label.set_text(r"cz [km$^{-1}$]")
        self.gauge1_ax1.axis["left"].label.set_text(r"")
        self.gauge1_ax1.axis["right"].label.set_text(r"")
        self.gauge1_ax1.axis["top"].label.set_text(pllabel)
        self.gauge1_ax1.axis["bottom"].label.set_text(" ")
    
        # create a parasite axes whose transData in RA, cz
        self.gauge1_aux_ax = self.gauge1_ax1.get_aux_axes(tr)
    
        self.gauge1_aux_ax.patch = self.gauge1_ax1.patch  # for aux_ax to have a clip path as in ax
        self.gauge1_ax1.patch.zorder = 0.9  # but this has a side effect that the patch is
        # drawn twice, and possibly over some other
        # artists. So, we decrease the zorder a bit to
        # prevent this.
        for an in [100.,200,300,400,500,700,800,900,1000,1100]:
            self.gauge1_aux_ax.plot([an*180/1200, an*180/1200], [0.5,1.], color='black',lw=1,ls='dotted')
        
        patches=[]
        #here I create the wedge to start
        self.gauge1_ax1.gauge1_wedge=Wedge((0,0),0.99,1,120,width=0.38,facecolor='#FF0000')
        self.gauge1_ax1.add_patch(self.gauge1_ax1.gauge1_wedge)
        
        # End gauge 1 -----------------------------------------------------------------------------------------------
        
        
        # pitch log plot
        self.ax_pitchlogs = self.main_figure.figure.add_subplot(616)
        self.ax_pitchlogs.set_ylim(190,1000)
        self.ax_pitchlogs.set_xlim(0, self.pitchlog_vect1.max())
        self.ax_pitchlogs.set_xlabel(u'pitch index ', fontsize=6)
        self.ax_pitchlogs.set_position([0.1,0.4, 0.75, 0.15])
        #self.ax_pitchlogs.set_position([0.1,0.1, 0.8, 0.25])
        
        # mark the 3rd, 4th and fifth
        self.ax_pitchlogs.plot(self.pitchlog_vect1,300 * np.ones_like(self.pitchlog_vect1),color='green',ls='dotted')
        self.ax_pitchlogs.plot(self.pitchlog_vect1,400 * np.ones_like(self.pitchlog_vect1),color='green',ls='dashed')
        self.ax_pitchlogs.plot(self.pitchlog_vect1,500 * np.ones_like(self.pitchlog_vect1),color='black',lw=2,ls='dotted')
        self.ax_pitchlogs.plot(self.pitchlog_vect1,700 * np.ones_like(self.pitchlog_vect1),color='black',lw=2,ls='dotted')
        self.ax_pitchlogs.plot(self.pitchlog_vect1,800 * np.ones_like(self.pitchlog_vect1),color='green',ls='dotted')
        self.ax_pitchlogs.plot(self.pitchlog_vect1,900 * np.ones_like(self.pitchlog_vect1),color='green',ls='dashed')
        
        #self.ax_pitchlogs.plot(self.pitchlog_vect1,-700 * np.ones_like(self.pitchlog_vect1),color='black',ls='dotted')
        
        # line objects
        self.line_ch1_time, = self.ax_ch1_time.plot(self.time_vect1,
                                         np.ones_like(self.time_vect1),color='black')
        self.line_ch1_spec, = self.ax_ch1_spec.plot(self.freq_vect1,
                                               np.ones_like(self.freq_vect1))
        self.ch1_pitch_line, = self.ax_ch1_spec.plot((self.freq_vect1[self.freq_vect1.size / 2], self.freq_vect1[self.freq_vect1.size / 2]),
                                              self.ax_ch1_spec.get_ylim(),color='green', lw=2)

        self.line_ch2_time, = self.ax_ch2_time.plot(self.time_vect2,np.ones_like(self.time_vect2),color='black')

        self.line_ch2_spec, = self.ax_ch2_spec.plot(self.freq_vect2,
                                                np.ones_like(self.freq_vect2))
        self.ch2_pitch_line, = self.ax_ch2_spec.plot((self.freq_vect2[self.freq_vect2.size / 2], self.freq_vect2[self.freq_vect2.size / 2]),
                                                     self.ax_ch2_spec.get_ylim(), color='red',lw=2)

        self.ch1_pitchlog, = self.ax_pitchlogs.plot(self.pitchlog_vect1,np.ones_like(self.pitchlog_vect1),color='green',lw=0,marker="o")

        # pitch log
        self.ch2_pitchlog, = self.ax_pitchlogs.plot(self.pitchlog_vect2, np.ones_like(self.pitchlog_vect2),color='red',lw=0,marker="o")
    
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
            
            #print("{} {}".format(min(np.log(np.abs(fft_frame))),max(np.log(np.abs(fft_frame)))))
            self.line_ch1_spec.set_data(self.freq_vect1, np.log(np.abs(fft_frame)))
            
            #time_str = "Time to plot channel1 in ms: %f" % ((time.time()-t0g)*1000)
            #print("{}".format(time_str))
            #  pitch tracking algorithm
            
            t0g = time.time()
            #new_pitch1 = compute_pitch_hps(current_frame1, self.mic.rate,dF=1)
            #precise_pitch1 = compute_pitch_hps(current_frame1, self.mic.rate, dF=0.05, Fmin=new_pitch1 * 0.8, Fmax = new_pitch1 * 1.2)
            signal1float=current_frame1.astype(np.float32)
            new_pitch1=precise_pitch1=pitch_o(signal1float)[0]
            pitch_confidence1 = pitch_o.get_confidence()
            #time_str = "Time to pitch track channel1 in ms: %f" % ((time.time()-t0g)*1000)
            #print("{}".format(time_str))
            
            
            self.ax_ch1_spec.set_title("pitch = {:.2f} Hz".format(precise_pitch1))
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
            self.line_ch2_spec.set_data(self.freq_vect1, np.log(np.abs(fft_frame)))
            #  pitch tracking algorithm
            #new_pitch2 = compute_pitch_hps(current_frame2, self.mic.rate,dF=1)
            #precise_pitch2 = compute_pitch_hps(current_frame2, self.mic.rate,dF=0.05, Fmin=new_pitch2 * 0.8, Fmax = new_pitch2 * 1.2)
            
            t0g = time.time()
            signal2float=current_frame2.astype(np.float32)
            new_pitch2=precise_pitch2=pitch_o(signal2float)[0]
            pitch_confidence2 = pitch_o.get_confidence()
            #time_str = "Time to aubio pitch track channel2 in ms: %f" % ((time.time()-t0g)*1000)
            #print("{}".format(time_str))

            self.ax_ch2_spec.set_title("pitch = {:.2f} Hz".format(precise_pitch2))
            self.ch2_pitch_line.set_data((new_pitch2, new_pitch2),self.ax_ch2_spec.get_ylim())


            new_pitch1Cent = 1200* math.log((new_pitch1+.1)/120.,2)
            new_pitch2Cent = 1200* math.log((new_pitch2+.1)/120.,2)
                
            ivCents=abs(new_pitch2Cent-new_pitch1Cent)
            
            if 0< ivCents <= 1200:
                self.gauge1_ax1.axis["top"].label.set_text(str(int(ivCents)))
                self.gauge1_ax1.gauge1_wedge.set_theta2(180.)
                self.gauge1_ax1.gauge1_wedge.set_theta1((1200 - ivCents)*180/1200)
                self.gauge1_ax1.add_patch(self.gauge1_ax1.gauge1_wedge)
                for an in [100.,200,300,400,500,700,800,900,1000,1100]:
                    self.gauge1_aux_ax.plot([an*180/1200, an*180/1200], [0.5,1.], color='black',lw=1,ls='dotted')
            

            update_pitch_log1(abs(new_pitch1Cent-new_pitch2Cent))
            self.ch1_pitchlog.set_data(self.pitchlog_vect1, pitchlog1)

            update_pitch_log2(abs(new_pitch2Cent-new_pitch1Cent))
            self.ch2_pitchlog.set_data(self.pitchlog_vect2, pitchlog2)
        
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
