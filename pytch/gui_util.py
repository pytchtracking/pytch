import math
import numpy as num
import sys
import PyQt5.QtGui as qg
import PyQt5.QtCore as qc
import PyQt5.QtWidgets as qw


_color_names = [
    "butter1",
    "butter2",
    "butter3",
    "chameleon1",
    "chameleon2",
    "chameleon3",
    "orange1",
    "orange2",
    "orange3",
    "skyblue1",
    "skyblue2",
    "skyblue3",
    "plum1",
    "plum2",
    "plum3",
    "chocolate1",
    "chocolate2",
    "chocolate3",
    "scarletred1",
    "scarletred2",
    "scarletred3",
    "aluminium1",
    "aluminium2",
    "aluminium3",
    "aluminium4",
    "aluminium5",
    "aluminium6",
    "black",
    "grey",
    "white",
    "red",
    "green",
    "blue",
    "transparent",
]


_color_values = [
    (252, 233, 79),
    (237, 212, 0),
    (196, 160, 0),
    (138, 226, 52),
    (115, 210, 22),
    (78, 154, 6),
    (252, 175, 62),
    (245, 121, 0),
    (206, 92, 0),
    (114, 159, 207),
    (52, 101, 164),
    (32, 74, 135),
    (173, 127, 168),
    (117, 80, 123),
    (92, 53, 102),
    (233, 185, 110),
    (193, 125, 17),
    (143, 89, 2),
    (239, 41, 41),
    (204, 0, 0),
    (164, 0, 0),
    (238, 238, 236),
    (211, 215, 207),
    (186, 189, 182),
    (136, 138, 133),
    (85, 87, 83),
    (46, 52, 54),
    (0, 0, 0),
    (10, 10, 10),
    (255, 255, 255),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (0, 0, 0, 0),
]


_colors = dict(zip(_color_names, _color_values))


_pen_styles = {
    "solid": qc.Qt.SolidLine,
    "dashed": qc.Qt.DashLine,
    "dashdot": qc.Qt.DashDotLine,
    "dotted": qc.Qt.DotLine,
    "-": qc.Qt.SolidLine,
    "--": qc.Qt.DashLine,
    "-.": qc.Qt.DashDotLine,
    ":": qc.Qt.DotLine,
    "o": qc.Qt.SolidLine,
}


def add_action_group(options, menu, slot, exclusive=True):
    action_group = qw.QActionGroup(menu)
    action_group.setExclusive(exclusive)
    choices = []
    for option in options:
        action = qw.QAction(option, menu)
        action.triggered.connect(slot)
        action.setCheckable(True)
        choices.append(action)
        action_group.addAction(action)
        menu.addAction(action)
    return choices


if sys.version_info < (3, 0):
    _buffer = buffer
else:
    _buffer = memoryview


def nice_value(x):
    """Round x to nice value."""

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
    """ normalize data vector *d* between 0 and 1"""
    dmin = num.min(d)
    return (d - dmin) / (num.max(d) - dmin)


class AutoScaler(object):
    """ taken from pyrocko.org"""

    def __init__(
        self,
        approx_ticks=7.0,
        mode="auto",
        exp=None,
        snap=False,
        inc=None,
        space=0.0,
        exp_factor=3,
        no_exp_interval=(-3, 5),
    ):

        """
        Create new AutoScaler instance.

        The parameters are described in the AutoScaler documentation.
        """

        self.approx_ticks = approx_ticks
        self.mode = mode
        self.exp = exp
        self.snap = snap
        self.inc = inc
        self.space = space
        self.exp_factor = exp_factor
        self.no_exp_interval = no_exp_interval

    def make_scale(self, data_range, override_mode=None):

        """
        Get nice minimum, maximum and increment for given data range.

        Returns ``(minimum, maximum, increment)`` or ``(maximum, minimum,
        -increment)``, depending on whether data_range is ``(data_min,
        data_max)`` or ``(data_max, data_min)``. If `override_mode` is defined,
        the mode attribute is temporarily overridden by the given value.
        """

        data_min = min(data_range)
        data_max = max(data_range)

        is_reverse = data_range[0] > data_range[1]

        a = self.mode
        if self.mode == "auto":
            a = self.guess_autoscale_mode(data_min, data_max)

        if override_mode is not None:
            a = override_mode

        mi, ma = 0, 0
        if a == "off":
            mi, ma = data_min, data_max
        elif a == "0-max":
            mi = 0.0
            if data_max > 0.0:
                ma = data_max
            else:
                ma = 1.0
        elif a == "min-0":
            ma = 0.0
            if data_min < 0.0:
                mi = data_min
            else:
                mi = -1.0
        elif a == "min-max":
            mi, ma = data_min, data_max

        elif a == "symmetric":
            m = max(abs(data_min), abs(data_max))
            mi = -m
            ma = m

        elif a == "int":
            # m = num.array(data_range, dtype=num.int)
            # mi = min(m)
            mi = num.int(num.min(data_range))
            ma = num.int(num.max(data_range))
            self.inc = 1.0
            # ma = max(m)

        nmi = mi
        if (mi != 0.0 or a == "min-max") and a != "off":
            nmi = mi - self.space * (ma - mi)

        nma = ma
        if (ma != 0.0 or a == "min-max") and a != "off":
            nma = ma + self.space * (ma - mi)

        mi, ma = nmi, nma

        if mi == ma and a != "off":
            mi -= 1.0
            ma += 1.0

        # make nice tick increment
        if self.inc is not None:
            inc = self.inc
        else:
            if self.approx_ticks > 0.0:
                inc = nice_value((ma - mi) / self.approx_ticks)
            else:
                inc = nice_value((ma - mi) * 10.0)

        if inc == 0.0:
            inc = 1.0

        # snap min and max to ticks if this is wanted
        if self.snap and a != "off":
            ma = inc * math.ceil(ma / inc)
            mi = inc * math.floor(mi / inc)

        if is_reverse:
            return ma, mi, -inc
        else:
            return mi, ma, inc

    def make_exp(self, x):
        """Get nice exponent for notation of `x`.

        For ax annotations, give tick increment as `x`."""

        if self.exp is not None:
            return self.exp

        x = abs(x)
        if x == 0.0:
            return 0

        if 10 ** self.no_exp_interval[0] <= x <= 10 ** self.no_exp_interval[1]:
            return 0

        return math.floor(math.log10(x) / self.exp_factor) * self.exp_factor

    def guess_autoscale_mode(self, data_min, data_max):
        """Guess mode of operation, based on data range.

        Used to map ``'auto'`` mode to ``'0-max'``, ``'min-0'``, ``'min-max'``
        or ``'symmetric'``."""

        a = "min-max"
        if data_min >= 0.0:
            if data_min < data_max / 2.0:
                a = "0-max"
            else:
                a = "min-max"
        if data_max <= 0.0:
            if data_max > data_min / 2.0:
                a = "min-0"
            else:
                a = "min-max"
        if data_min < 0.0 and data_max > 0.0:
            if (
                abs((abs(data_max) - abs(data_min)) / (abs(data_max) + abs(data_min)))
                < 0.5
            ):
                a = "symmetric"
            else:
                a = "min-max"
        return a


class Projection(object):
    def __init__(self):
        self.xr = 0.0, 1.0
        self.ur = 0.0, 1.0
        self.update()

    def update(self):
        self.out_range = abs(self.ur[1] - self.ur[0])
        self.in_range = abs(self.xr[1] - self.xr[0])

    def set_in_range(self, xmin, xmax):
        if xmax == xmin:
            xmax = xmin + 1.0

        self.xr = xmin, xmax
        self.update()

    def get_in_range(self):
        return self.xr

    def set_out_range(self, umin, umax, flip=False):
        if umax == umin:
            umax = umin + 1.0

        if flip:
            self.ur = umax, umin
        else:
            self.ur = umin, umax
        self.update()

    def get_out_range(self):
        return self.ur

    def __call__(self, x):
        umin, umax = self.ur
        xmin, xmax = self.xr
        return umin + (x - xmin) * ((umax - umin) / (xmax - xmin))

    def clipped(self, x):
        umin, umax = self.ur
        xmin, xmax = self.xr
        return min(umax, max(umin, umin + (x - xmin) * ((umax - umin) / (xmax - xmin))))

    def rev(self, u):
        umin, umax = self.ur
        xmin, xmax = self.xr
        return xmin + (u - umin) * ((xmax - xmin) / (umax - umin))

    def copy(self):
        return copy.copy(self)


def make_QPolygonF(xdata, ydata):
    """Create a :py:class:`qg.QPolygonF` instance from xdata and ydata, both
    numpy arrays."""
    assert len(xdata) == len(ydata)

    nydata = len(ydata)
    qpoints = qg.QPolygonF(nydata)
    vptr = qpoints.data()
    vptr.setsize(int(nydata * 8 * 2))
    aa = num.ndarray(shape=(nydata, 2), dtype=num.float64, buffer=_buffer(vptr))
    aa.setflags(write=True)
    aa[:, 0] = xdata
    aa[:, 1] = ydata

    return qpoints


def mean_decimation(d, ndecimate):
    """ Decimate signal by factor (int) *ndecimate* using averaging."""
    pad_size = int(math.ceil(float(d.size) / ndecimate) * ndecimate - d.size)
    d = num.append(d, num.zeros(pad_size) * num.nan)
    return num.nanmean(d.reshape(-1, ndecimate), axis=1)


def minmax_decimation(d, ndecimate):
    pad_size = int(math.ceil(float(d.size) / ndecimate) * ndecimate - d.size)
    d = num.append(d, num.zeros(pad_size) * num.nan).reshape(-1, ndecimate)
    dout = num.nanmax(d, axis=1)
    dout[::2] = num.nanmin(d[::2], axis=1)
    return dout


class FloatQLineEdit(qw.QLineEdit):
    accepted_value = qc.pyqtSignal(float)

    def __init__(self, default=None, *args, **kwargs):
        qw.QLineEdit.__init__(self, *args, **kwargs)
        self.setValidator(qg.QDoubleValidator())
        self.setFocusPolicy(qc.Qt.ClickFocus | qc.Qt.TabFocus)
        self.returnPressed.connect(self.do_check)
        p = self.parent()
        if p:
            self.returnPressed.connect(p.setFocus)
        if default:
            self.setText(str(default))

    def do_check(self):
        text = self.text()
        val = float(text)
        if val >= 0.0:
            self.accepted_value.emit(val)


class PlotBase(object):
    def __init__(self, *args, **kwargs):
        self.wheel_pos = 0
        self.scroll_increment = 100
        self.setFocusPolicy(qc.Qt.StrongFocus)
        self.set_background_color("white")
        self.setContentsMargins(0, 0, 0, 0)
        # self.setSizePolicy(
        ##    #qw.QSizePolicy.Maximum, qw.QSizePolicy.Maximum)
        #    qw.QSizePolicy.Minimum,
        #    qw.QSizePolicy.Minimum)

    def sizeHint(self):
        return qc.QSize(200, 200)

    def canvas_rect(self):
        """ Rectangular containing the data visualization. """
        w, h = self.wh
        tl = qc.QPoint(self.left * w, (1.0 - self.top) * h)
        size = qc.QSize(w * (self.right - self.left), h * self.top - self.bottom)
        rect = self.rect()
        rect.setTopLeft(tl)
        rect.setSize(size)
        return rect

    @property
    def wh(self):
        return self.width(), self.height()

    def set_brush(self, color="black"):
        self.brush = qg.QBrush(qg.QColor(*_colors[color]))

    def set_pen_color(self, color="black"):
        self.pen.setColor(qg.QColor(*_colors[color]))

    def get_pen(self, color="black", line_width=1, style="solid"):
        """
        :param color: color name as string
        """
        if style == "o":
            self.draw_points = True

        return qg.QPen(qg.QColor(*_colors[color]), line_width, _pen_styles[style])

    def update_projections(self):
        w, h = self.width(), self.height()

        mi, ma = self.xproj.get_out_range()
        xzoom = self.xzoom * (ma - mi)
        self.xproj.set_in_range(self._xmin - xzoom, self._xmax)
        self.xproj.set_out_range(w * self.left, w * self.right)

        self.yproj.set_in_range(self._ymin, self._ymax)
        self.yproj.set_out_range(
            h * (1.0 - self.top), h * (1.0 - self.bottom), flip=True
        )

    @qc.pyqtSlot(qg.QMouseEvent)
    def wheelEvent(self, wheel_event):
        self.wheel_pos += wheel_event.angleDelta().y()
        n = self.wheel_pos / 120
        self.wheel_pos = self.wheel_pos % 120
        if n == 0:
            return

        modifier = qw.QApplication.keyboardModifiers()
        if modifier & qc.Qt.ControlModifier:
            self.set_ylim(
                self._ymin - self.scroll_increment * n,
                self._ymax + self.scroll_increment * n,
            )
        else:
            self.set_ylim(
                self._ymin - self.scroll_increment * n,
                self._ymax - self.scroll_increment * n,
            )

        self.update()

    def set_ylim(self, ymin, ymax):
        """ Set range of Gauge."""
        self.ymin = ymin
        self.ymax = ymax
        self._ymin = ymin
        self._ymax = ymax
        self.update_projections()

    @qc.pyqtSlot(qg.QKeyEvent)
    def keyPressEvent(self, key_event):
        """ react on keyboard keys when they are pressed."""
        key_text = key_event.text()
        if key_text == "+":
            self.set_ylim(
                self._ymin - self.scroll_increment, self._ymax + self.scroll_increment
            )
        elif key_text == "-":
            self.set_ylim(
                self._ymin + self.scroll_increment, self._ymax - self.scroll_increment
            )

    def set_background_color(self, color):
        """
        :param color: color as string
        """
        background_color = qg.QColor(*_colors[color])
        self.background_brush = qg.QBrush(background_color)
        pal = self.palette()
        pal.setBrush(qg.QPalette.Background, self.background_brush)
        self.setPalette(pal)


class LineEditWithLabel(qw.QWidget):
    def __init__(self, label, default=None, *args, **kwargs):
        qw.QWidget.__init__(self, *args, **kwargs)
        layout = qw.QHBoxLayout()
        layout.addWidget(qw.QLabel(label))
        self.setLayout(layout)

        self.edit = qw.QLineEdit()
        layout.addWidget(self.edit)

        if default:
            self.edit.setText(str(default))

    @property
    def value(self):
        return self.edit.text()


class AdjustableMainWindow(qw.QMainWindow):
    def sizeHint(self):
        return qc.QSize(1200, 500)

    @qc.pyqtSlot(qg.QKeyEvent)
    def keyPressEvent(self, key_event):
        """ react on keyboard keys when they are pressed."""
        key_text = key_event.text()
        if key_text == "q":
            self.close()
        elif key_text == "f":
            self.showMaximized
        super().keyPressEvent(key_event)
