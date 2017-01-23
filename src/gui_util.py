import math
import numpy as num
import sys

import PyQt5.QtGui as qg


if sys.version_info < (3, 0):
    _buffer = buffer
else:
    _buffer = memoryview

def nice_value(x):
    '''Round x to nice value.'''

    exp = 1.0
    sign = 1
    if x < 0.0:
        x = -x
        sign = -1
    while x >= 1.0:
        x /= 10.0
        exp *= 10.0
    while x < 0.1:
        x *= 10.0
        exp /= 10.0

    if x >= 0.75:
        return sign * 1.0 * exp
    if x >= 0.35:
        return sign * 0.5 * exp
    if x >= 0.15:
        return sign * 0.2 * exp

    return sign * 0.1 * exp


def normalized_to01(d):
    ''' normalize data vector *d* between 0 and 1'''
    dmin = num.min(d)
    return (d-dmin)/(num.max(d)-dmin)



class AutoScaler():
    ''' taken from pyrocko.org'''
    def __init__(
            self,
            approx_ticks=7.0,
            mode='auto',
            exp=None,
            snap=False,
            inc=None,
            space=0.0,
            exp_factor=3,
            no_exp_interval=(-3, 5)):

        '''
        Create new AutoScaler instance.

        The parameters are described in the AutoScaler documentation.
        '''

        self.approx_ticks = approx_ticks
        self.mode = mode
        self.exp = exp
        self.snap = snap
        self.inc = inc
        self.space = space
        self.exp_factor = exp_factor
        self.no_exp_interval = no_exp_interval

    def make_scale(self, data_range, override_mode=None):

        '''
        Get nice minimum, maximum and increment for given data range.

        Returns ``(minimum, maximum, increment)`` or ``(maximum, minimum,
        -increment)``, depending on whether data_range is ``(data_min,
        data_max)`` or ``(data_max, data_min)``. If `override_mode` is defined,
        the mode attribute is temporarily overridden by the given value.
        '''

        data_min = min(data_range)
        data_max = max(data_range)

        is_reverse = (data_range[0] > data_range[1])

        a = self.mode
        if self.mode == 'auto':
            a = self.guess_autoscale_mode(data_min, data_max)

        if override_mode is not None:
            a = override_mode

        mi, ma = 0, 0
        if a == 'off':
            mi, ma = data_min, data_max
        elif a == '0-max':
            mi = 0.0
            if data_max > 0.0:
                ma = data_max
            else:
                ma = 1.0
        elif a == 'min-0':
            ma = 0.0
            if data_min < 0.0:
                mi = data_min
            else:
                mi = -1.0
        elif a == 'min-max':
            mi, ma = data_min, data_max

        elif a == 'symmetric':
            m = max(abs(data_min), abs(data_max))
            mi = -m
            ma = m

        nmi = mi
        if (mi != 0. or a == 'min-max') and a != 'off':
            nmi = mi - self.space*(ma-mi)

        nma = ma
        if (ma != 0. or a == 'min-max') and a != 'off':
            nma = ma + self.space*(ma-mi)

        mi, ma = nmi, nma

        if mi == ma and a != 'off':
            mi -= 1.0
            ma += 1.0

        # make nice tick increment
        if self.inc is not None:
            inc = self.inc
        else:
            if self.approx_ticks > 0.:
                inc = nice_value((ma-mi) / self.approx_ticks)
            else:
                inc = nice_value((ma-mi)*10.)

        if inc == 0.0:
            inc = 1.0

        # snap min and max to ticks if this is wanted
        if self.snap and a != 'off':
            ma = inc * math.ceil(ma/inc)
            mi = inc * math.floor(mi/inc)

        if is_reverse:
            return ma, mi, -inc
        else:
            return mi, ma, inc


    def make_exp(self, x):
        '''Get nice exponent for notation of `x`.

        For ax annotations, give tick increment as `x`.'''

        if self.exp is not None:
            return self.exp

        x = abs(x)
        if x == 0.0:
            return 0

        if 10**self.no_exp_interval[0] <= x <= 10**self.no_exp_interval[1]:
            return 0

        return math.floor(math.log10(x)/self.exp_factor)*self.exp_factor

    def guess_autoscale_mode(self, data_min, data_max):
        '''Guess mode of operation, based on data range.

        Used to map ``'auto'`` mode to ``'0-max'``, ``'min-0'``, ``'min-max'``
        or ``'symmetric'``.'''

        a = 'min-max'
        if data_min >= 0.0:
            if data_min < data_max/2.:
                a = '0-max'
            else:
                a = 'min-max'
        if data_max <= 0.0:
            if data_max > data_min/2.:
                a = 'min-0'
            else:
                a = 'min-max'
        if data_min < 0.0 and data_max > 0.0:
            if abs((abs(data_max)-abs(data_min)) /
                   (abs(data_max)+abs(data_min))) < 0.5:
                a = 'symmetric'
            else:
                a = 'min-max'
        return a



class Projection(object):
    def __init__(self):
        self.xr = 0., 1.
        self.ur = 0., 1.

    def set_in_range(self, xmin, xmax):
        if xmax == xmin:
            xmax = xmin + 1.

        self.xr = xmin, xmax

    def get_in_range(self):
        return self.xr

    def set_out_range(self, umin, umax):
        if umax == umin:
            umax = umin + 1.

        self.ur = umin, umax

    def get_out_range(self):
        return self.ur

    def __call__(self, x):
        umin, umax = self.ur
        xmin, xmax = self.xr
        return umin + (x-xmin)*((umax-umin)/(xmax-xmin))

    def clipped(self, x):
        umin, umax = self.ur
        xmin, xmax = self.xr
        return min(umax, max(umin, umin + (x-xmin)*((umax-umin)/(xmax-xmin))))

    def rev(self, u):
        umin, umax = self.ur
        xmin, xmax = self.xr
        return xmin + (u-umin)*((xmax-xmin)/(umax-umin))

    def copy(self):
        return copy.copy(self)


def make_QPolygonF(xdata, ydata):
    '''Create a :py:class:`qg.QPolygonF` instance from xdata and ydata, both
    numpy arrays.'''
    assert len(xdata) == len(ydata)

    nydata = len(ydata)
    qpoints = qg.QPolygonF(nydata)
    vptr = qpoints.data()
    vptr.setsize(int(nydata*8*2))
    aa = num.ndarray(
        shape=(nydata, 2),
        dtype=num.float64,
        buffer=_buffer(vptr))
    aa.setflags(write=True)
    aa[:, 0] = xdata
    aa[:, 1] = ydata

    return qpoints


def mean_decimation(d, ndecimate):
    ''' Decimate signal by factor (int) *ndecimate* using averaging.'''
    pad_size = int(math.ceil(float(d.size)/ndecimate)*ndecimate - d.size)
    d = num.append(d, num.zeros(pad_size)*num.nan)
    return num.nanmean(d.reshape(-1, ndecimate), axis=1)


