import logging
import time
import numpy as num
from scipy import signal

logger = logging.getLogger("pytch.util")

keys = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def dummy():
    return


def f2cent(f, standard_frequency):
    return 1200.0 * num.log2((num.maximum(f, 0.0) + 0.1) / standard_frequency)


def cent2f(p, standard_frequency):
    return num.exp2(p / 1200.0) * standard_frequency - 0.1


relative_keys = dict(
    zip(
        list(range(100, 1500, 100)),
        [
            "2m",
            "2M",
            "3m",
            "3M",
            "4",
            "TT",
            "5",
            "6m",
            "6M",
            "7m",
            "7M",
            "P8",
            "9m",
            "9M",
        ],
    )
)


class DummySignal:
    """does nothing when emitted"""

    def __init__(self):
        pass

    def emit(self):
        logger.debug("DummySignal emitted")
        return

    def connect(self, *args, **kwargs):
        logger.debug("connected to DummySignal")


class Profiler:
    def __init__(self):
        self.times = []

    def mark(self, m):
        self.times.append((m, time.time()))

    def start(self):
        self.mark("")

    def __str__(self):
        tstart = self.times[0][1]
        s = ""
        for imark, mark in enumerate(self.times[1:]):
            s += "{}: {}\n".format(mark[0], mark[1] - self.times[imark][1])

        s += "total: %s" % (self.times[-1][1] - self.times[0][1])
        return s


def consecutive(arr):
    return num.split(arr, num.where(num.diff(arr) != 1)[0] + 1)


def index_gradient_filter(x, y, max_gradient):
    """Get index where the abs gradient of x, y is < max_gradient."""
    return num.where(num.abs(num.diff(y) / num.diff(x)) < max_gradient)[0]


def smooth(x, window_len=11, window="hanning"):
    """
    from http://scipy-cookbook.readthedocs.io/items/SignalSmooth.html

    smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    num.hanning, num.hamming, num.bartlett, num.blackman, num.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        return x
        # raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ["flat", "hanning", "hamming", "bartlett", "blackman"]:
        raise ValueError(
            "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
        )

    s = num.r_[x[window_len - 1 : 0 : -1], x, x[-1:-window_len:-1]]
    # print(len(s))
    if window == "flat":  # moving average
        w = num.ones(window_len, "d")
    else:
        w = eval("num." + window + "(window_len)")

    # y=num.convolve(w/w.sum(),s,mode='valid')
    # return num.convolve(w/w.sum(),s,mode='valid')
    return signal.fftconvolve(w / w.sum(), s, mode="valid")
