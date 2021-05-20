import numpy as num
import math
import scipy.interpolate as interpolate
import logging

from pytch.gui_util import PlotBase
from pytch.gui_util import AutoScaler, Projection, minmax_decimation
from pytch.gui_util import make_QPolygonF, _colors, _pen_styles  # noqa
from . import viridis

from PyQt5 import QtCore as qc
from PyQt5 import QtGui as qg
from PyQt5 import QtWidgets as qw

logger = logging.getLogger("pytch.plot")

try:
    from pytch.gui_util_opengl import GLWidget
except AttributeError as e:
    logger.debug(e)
    GLWidget = qw.QWidget

d2r = num.pi / 180.0
logger = logging.getLogger(__name__)


def get_colortable(name, log=False):
    if name == "viridis":
        ctable = [qg.qRgb(*val) for val in viridis.get_rgb()]
        if log:
            raise Exception("Not implemented")

    elif name in ["bw", "wb"]:
        ctable = []
        if log:
            a = num.exp(num.linspace(num.log(255.0), 0.0, 256))
        else:
            a = num.linspace(0, 255, 256)

        if name == "bw":
            a = a[::-1]

        for i in a.astype(num.int):
            ctable.append(qg.qRgb(i, i, i))

    elif name == "matrix":
        for i in range(256):
            ctable.append(qg.qRgb(i / 4, i * 2, i / 2))

    else:
        raise Exception("No such colortable %s" % name)

    return ctable


class InterpolatedColormap(object):
    """ Continuously interpolating colormap """

    def __init__(self, name=""):
        self.name = name
        self.colors = num.array([_colors["red"], _colors["green"], _colors["blue"]])

        self.values = num.linspace(0, 255.0, len(self.colors))
        self.r_interp = interpolate.interp1d(self.values, self.colors.T[0])
        self.g_interp = interpolate.interp1d(self.values, self.colors.T[1])
        self.b_interp = interpolate.interp1d(self.values, self.colors.T[2])
        self.proj = Projection()
        self.proj.set_out_range(0, 255.0)

    def update(self):
        pass

    def _map(self, val):
        """Interpolate RGB colormap for *val*
        val can be a 1D array.

        Values which are out of range are clipped.
        """
        val = self.proj.clipped(val)
        return self.r_interp(val), self.g_interp(val), self.b_interp(val)

    def map(self, val):
        return self._map(val)

    def map_to_QColor(self, val):
        return qg.QColor(*self.map(val))

    def set_vlim(self, vmin, vmax):
        if (vmin, vmax) == self.proj.get_in_range():
            return
        else:
            self.proj.set_in_range(vmin, vmax)
            self.update()

    def get_incremented_values(self, n=40):
        """ has to be implemented by every subclass. Needed for plotting."""
        mi, ma = self.proj.get_in_range()
        return num.linspace(mi, ma, n)

    def get_visualization(self, callback=None):
        """get dict of values and colors for visualization.

        :param callback: method to retrieve colors from value range.
                        default: *map*
        """
        vals = self.get_incremented_values()

        if callback:
            colors = [callback(v) for v in vals]
        else:
            colors = [self.map(v) for v in vals]

        return vals, colors

    def __call__(self, val):
        return self._map(val)


class Colormap(InterpolatedColormap):
    """Like Colormap but with discrete resolution and precalculated.
    Can return tabulated QColors. Faster than Colormap"""

    def __init__(self, name="", n=20):
        InterpolatedColormap.__init__(self, name=name)
        self.n = n
        self.update()

    def set_vlim(self, vmin, vmax):
        if (vmin, vmax) == self.proj.get_in_range():
            return
        else:
            self.proj.set_in_range(vmin, vmax)
            self.update()

    def update(self):
        vals = self.get_incremented_values()
        self.colors_QPen = []
        self.colors_QColor = []
        self.colors_rgb = []
        for v in vals:
            rgb = self._map(v)
            self.colors_rgb.append(rgb)
            c = qg.QColor(*rgb)
            self.colors_QColor.append(c)
            self.colors_QPen.append(qg.QPen(c))

    def get_incremented_values(self):
        return num.linspace(*self.proj.xr, num=self.n + 1)

    def get_index(self, val):
        return int(self.proj.clipped(val) / self.proj.ur[1] * self.n)

    def map(self, val):
        return self.colors_rgb[self.get_index(val)]

    def map_to_QColor(self, val):
        return self.colors_QColor[self.get_index(val)]

    def map_to_QPen(self, val):
        i = self.get_index(val)
        return self.colors_QPen[i]


class ColormapWidget(qw.QWidget):
    def __init__(self, colormap, *args, **kwargs):
        qw.QWidget.__init__(self, *args, **kwargs)
        self.colormap = colormap
        self.yproj = Projection()
        # size_policy = qw.QSizePolicy()
        # size_policy.setHorizontalPolicy(qw.QSizePolicy.Maximum)
        # self.setSizePolicy(size_policy)
        self.set_background_color("white")
        self._update()

    def _update(self):
        _, rgb = self.colormap.get_visualization()
        self.vals, self.colors = self.colormap.get_visualization(
            callback=self.colormap.map_to_QColor
        )
        self.yproj.set_in_range(num.min(self.vals), num.max(self.vals))

    @qc.pyqtSlot(qg.QPaintEvent)
    def paintEvent(self, e):
        rect = self.rect()
        self.yproj.set_out_range((1.0 - rect.top()), rect.bottom(), flip=True)

        yvals = self.yproj(self.vals)
        painter = qg.QPainter(self)
        for i in range(len(self.vals) - 1):
            patch = qc.QRect(
                qc.QPoint(rect.left(), yvals[i]), qc.QPoint(rect.right(), yvals[i + 1])
            )
        painter.end()

    # def sizeHint(self):
    #    return qc.QSize(100, 400)

    def set_background_color(self, color):
        """
        :param color: color as string
        """
        background_color = qg.QColor(*_colors[color])
        self.background_brush = qg.QBrush(background_color)
        pal = self.palette()
        pal.setBrush(qg.QPalette.Background, self.background_brush)
        self.setPalette(pal)


def MakeGaugeWidget(gl=False):
    if gl:
        WidgetBase = GLWidget
    else:
        WidgetBase = qw.QWidget

    class _GaugeWidget(WidgetBase, PlotBase):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            self.color = qg.QColor(0, 100, 100)
            self._val = 0
            self.set_title("")
            self.proj = Projection()
            out_min = -130.0
            out_max = 130.0
            self.proj.set_out_range(out_min, out_max)
            self.set_ylim(0.0, 1500.0)

            self.scaler = AutoScaler(no_exp_interval=(-3, 2), approx_ticks=7, snap=True)

            self.xtick_increment = 20
            self.pen = qg.QPen(self.color, 20, qc.Qt.SolidLine)
            self.pen.setCapStyle(qc.Qt.FlatCap)
            self.wheel_pos = 0

        def set_ylim(self, ymin, ymax):
            """ Set range of Gauge."""
            self.ymin = ymin
            self.ymax = ymax
            self._ymin = ymin
            self._ymax = ymax
            self.proj.set_in_range(self.ymin, self.ymax)

        def get_ylim(self):
            return self.ymin, self.ymax

        @qc.pyqtSlot(qg.QPaintEvent)
        def paintEvent(self, e):
            """ This is executed when self.repaint() is called"""
            painter = qg.QPainter(self)
            self.side = min(self.width(), self.height()) / 1.05
            self.halfside = self.side / 2.0
            rect = qc.QRectF(-self.halfside, -self.halfside, self.side, self.side)
            painter.translate(self.width() / 2.0, self.height() / 2.0)
            painter.save()
            painter.setPen(self.pen)
            self.arc_start = -(self.proj(0) + 180) * 16
            pmin, pmax = self.proj.get_in_range()
            if self._val:
                span_angle = self.proj(self._val) * 16 + self.arc_start + 180 * 16.0
                painter.drawArc(rect, self.arc_start, -span_angle)
            painter.restore()
            self.draw_deco(painter)

        def draw_deco(self, painter):
            painter.save()
            self.draw_ticks(painter)
            self.draw_title(painter)
            painter.restore()

        def draw_title(self, painter):
            painter.drawText(-40.0, 2.0, self.title)

        def draw_ticks(self, painter):
            # needs some performance polishing !!!
            xmin, xmax = self.proj.get_in_range()

            ticks = num.arange(
                xmin, xmax + self.xtick_increment, self.xtick_increment, dtype=num.int
            )
            ticks_proj = self.proj(ticks) + 180

            line = qc.QLine(self.halfside * 0.9, 0, self.halfside, 0)
            subline = qc.QLine(self.halfside * 0.95, 0, self.halfside, 0)
            painter.save()
            font = painter.font()
            font.setPointSize(8)
            painter.setFont(font)
            text_box_with = 100
            for i, degree in enumerate(ticks_proj):
                painter.save()
                if i % 5 == 0:
                    rotate_rad = degree * d2r
                    x = self.halfside * 0.8 * num.cos(rotate_rad) - text_box_with / 2
                    y = self.halfside * 0.8 * num.sin(rotate_rad)
                    rect = qc.QRectF(x, y, text_box_with, text_box_with / 5)
                    painter.drawText(rect, qc.Qt.AlignCenter, str(ticks[i]))
                    painter.rotate(degree)
                    painter.drawLine(line)
                else:
                    painter.rotate(degree)
                    painter.drawLine(subline)
                painter.restore()
            painter.restore()

        def set_title(self, title):
            self.title = title

        def set_data(self, val):
            """
            Call this method to update the arc
            """
            self._val = val

        # def sizeHint(self):
        #     return qc.QSize(400, 400)

    return _GaugeWidget


class Grid:
    def __init__(self, horizontal=True, vertical=True, *args, **kwargs):
        self.vertical = vertical
        self.horizontal = horizontal

    def draw_grid(self, widget, painter):
        return


class AutoGrid:
    def __init__(
        self, pen_color="aluminium2", style=":", line_width=1, *args, **kwargs
    ):
        Grid.__init__(self, *args, **kwargs)

        self.data_lims_v = (None, None)
        self.data_lims_h = (None, None)
        self.lines_h = []
        self.lines_v = []

        self.set_pen(pen_color, style, line_width)

    def set_pen(self, pen_color, style, line_width):
        self.grid_pen = qg.QPen(
            qg.QColor(*_colors[pen_color]), line_width, _pen_styles[style]
        )

    def draw_grid(self, widget, painter):
        lines = []
        if self.horizontal:
            lines.extend(self.lines_horizontal(widget, painter))

        if self.vertical:
            lines.extend(self.lines_vertical(widget, painter))

        painter.save()
        painter.setPen(self.grid_pen)
        painter.drawLines(lines)
        painter.restore()

    def lines_horizontal(self, widget, painter):
        """ setup horizontal grid lines"""

        # if not (widget._ymin, widget._ymax) == self.data_lims_h:
        ymin, ymax, yinc = widget.yscaler.make_scale((widget._ymin, widget._ymax))
        ticks_proj = widget.yproj(num.arange(ymin, ymax, yinc))

        w, h = widget.wh
        self.lines_h = [
            qc.QLineF(w * widget.left, yval, w, yval) for yval in ticks_proj
        ]
        self.data_lims_h = (widget._ymin, widget._ymax)

        return self.lines_h

    def lines_vertical(self, widget, painter):
        """ setup vertical grid lines"""

        # if not (widget._xmin, widget._xmax) == self.data_lims_v:
        xmin, xmax, xinc = widget.xscaler.make_scale((widget._xmin, widget._xmax))
        ticks_proj = widget.xproj(num.arange(xmin, xmax, xinc))

        w, h = widget.wh
        self.lines_v = [
            qc.QLineF(xval, h * (1.0 - widget.top), xval, h) for xval in ticks_proj
        ]
        self.data_lims_v = (widget._xmin, widget._xmax)

        return self.lines_v


class FixGrid(AutoGrid):
    def __init__(self, delta, *args, **kwargs):
        self.delta = delta
        AutoGrid.__init__(self, *args, **kwargs)

    def lines_horizontal(self, widget, painter):
        """ setup horizontal grid lines"""

        # if not (widget._ymin, widget._ymax) == self.data_lims_h:
        ymin, ymax, yinc = widget.yscaler.make_scale((widget._ymin, widget._ymax))
        ticks_proj = widget.yproj(num.arange(ymin, ymax, self.delta))

        w, h = widget.wh
        self.lines_h = [
            qc.QLineF(w * widget.left, yval, w, yval) for yval in ticks_proj
        ]
        self.data_lims_h = (widget._ymin, widget._ymax)

        return self.lines_h

    def lines_vertical(self, widget, painter):
        """ setup vertical grid lines"""

        # if not (widget._xmin, widget._xmax) == self.data_lims_v:
        xmin, xmax, xinc = widget.xscaler.make_scale((widget._xmin, widget._xmax))
        ticks_proj = widget.xproj(num.arange(xmin, xmax, self.delta))

        w, h = widget.wh
        self.lines_v = [
            qc.QLineF(xval, h * (1.0 - widget.top), xval, h) for xval in ticks_proj
        ]
        self.data_lims_v = (widget._xmin, widget._xmax)

        return self.lines_v


class SceneItem:
    def __init__(self, x=None, y=None, pen=None):
        self.pen = pen
        self.x = x
        self.y = y


class AxHLine(SceneItem):
    def __init__(self, y, pen):
        SceneItem.__init__(self, y=y, pen=pen)

    def draw(self, painter, xproj, yproj, rect=None):
        xmin, xmax = xproj.get_out_range()
        y = yproj(self.y)

        painter.save()
        painter.setPen(self.pen)
        painter.drawLine(xmin, y, xmax, y)
        painter.restore()


class AxVLine(SceneItem):
    def __init__(self, x, pen):
        SceneItem.__init__(self, x=x, pen=pen)

    def draw(self, painter, xproj, yproj, rect=None):
        ymin, ymax = yproj.get_out_range()
        x = xproj(self.x)

        painter.save()
        painter.setPen(self.pen)
        painter.drawLine(x, ymin, x, ymax)
        painter.restore()


class Points(SceneItem):
    """ Holds and draws data projected to screen dimensions."""

    def __init__(self, x, y, pen, antialiasing=True):
        SceneItem.__init__(self, x=x, y=y, pen=pen)
        self.antialiasing = antialiasing

    def draw(self, painter, xproj, yproj, rect=None):
        qpoints = make_QPolygonF(xproj(self.x), yproj(self.y))
        painter.save()
        if self.antialiasing:
            painter.setRenderHint(qg.QPainter.Antialiasing, True)
        painter.setPen(self.pen)
        painter.drawPoints(qpoints)
        painter.restore()


class Polyline(SceneItem):
    """ Holds and draws data projected to screen dimensions."""

    def __init__(self, x, y, pen, antialiasing=True):
        SceneItem.__init__(self, x=x, y=y, pen=pen)
        self.antialiasing = antialiasing

    def draw(self, painter, xproj, yproj, rect=None):
        qpoints = make_QPolygonF(xproj(self.x), yproj(self.y))

        painter.save()
        painter.setPen(self.pen)
        if self.antialiasing:
            painter.setRenderHint(qg.QPainter.Antialiasing, True)
        painter.drawPolyline(qpoints)
        painter.restore()


class Text(SceneItem):
    def __init__(self, x, y, pen, text):
        SceneItem.__init__(self, x=x, y=y, pen=pen)
        self.text = qg.QStaticText(str(text))

    def draw(self, painter, xproj, yproj, rect=None):
        painter.drawStaticText(qc.QPoint(xproj(self.x) * 0.9, yproj(self.y)), self.text)


class PColormesh(qw.QWidget):
    """
    2D array. Currently, opengl is not supported.
    """

    colortable = "viridis"

    def __init__(self, img, x, y, *args, **kwargs):
        super(qw.QWidget, self).__init__(*args, **kwargs)
        self.x = x
        self.y = y
        self.vmax = 13000
        self.vmin = None
        self.rect = None
        self.img = img
        buff = self.img.bits()
        nx = len(x)
        ny = len(y)
        buff.setsize(nx * ny * 2 ** 8)
        self.img_data = num.ndarray(shape=(nx, ny), dtype=num.uint8, buffer=buff)
        self.color_table = "viridis"

    def draw(self, painter, xproj, yproj, rect=None):
        painter.save()
        painter.setRenderHint(qg.QPainter.SmoothPixmapTransform, True)
        painter.drawImage(rect, self.img)
        painter.restore()

    @classmethod
    def from_numpy_array(cls, x=None, y=None, z=None):
        """
        :param a: RGB array
        """
        if not z.dtype == num.uint8:
            z = num.asarray(z, dtype=num.uint8)
            z = num.ascontiguousarray(z)

        img = qg.QImage(z.data, z.shape[1], z.shape[0], qg.QImage.Format_Indexed8)

        img.setColorTable(get_colortable(cls.colortable))

        o = cls(img, x, y)
        o.set_data(z)
        return o

    def prescale(self, d):
        return num.clip(num.divide(d, self.vmax), 0, 254)

    def set_data(self, *args):
        """
        :param args: z(2d) or x, y, z(2d) as arrays
        """
        if len(args) == 3:
            x, y, z = args
        elif len(args) == 1:
            z = args
        else:
            raise Exception("Invalid number of arguments to *set_data*")
        self.img_data[:, :] = memoryview(self.prescale(z))

    def set_colortable(self, name):
        self.img.setColorTable(get_colortable(name))


def MakeAxis(gl=True):
    if gl:
        WidgetBase = GLWidget
    else:
        WidgetBase = qw.QWidget

    class _Axis(WidgetBase, PlotBase):
        """ a plotwidget displays data (x, y coordinates). """

        def __init__(self, *args, **kwargs):
            WidgetBase.__init__(self, *args, **kwargs)
            PlotBase.__init__(self, *args, **kwargs)

            self.scroll_increment = 0
            self.track_start = None

            self.ymin = None
            self.ymax = None
            self.xmin = None
            self.xmax = None

            self._ymin = 0.0
            self._ymax = 1.0
            self._yinc = None
            self._xmin = 0.0
            self._xmax = 1.0
            self._xinc = None

            self.left = 0.15
            self.right = 1.0
            self.bottom = 0.1
            self.top = 0.9

            self.xlabels = True
            self.ylabels = True
            self.yticks = True
            self.xticks = True

            self.ytick_formatter = "%s"
            self.xtick_formatter = "%s"
            self.xzoom = 0.0

            self.clear()
            self.grids = [AutoGrid()]
            self.__want_minor_grid = True

            self.yscaler = AutoScaler(
                no_exp_interval=(-3, 2), approx_ticks=5, snap=True
            )

            self.xscaler = AutoScaler(
                no_exp_interval=(-3, 2), approx_ticks=5, snap=True
            )
            self.draw_points = False
            self.set_brush()
            self._xvisible = num.empty(0)
            self._yvisible = num.empty(0)
            self.yproj = Projection()
            self.xproj = Projection()
            self.colormap = Colormap()
            self.setAutoFillBackground(True)

        def clear(self):
            self.scene_items = []

        def setup_annotation_boxes(self):
            """ left and top boxes containing labels, dashes, marks, etc."""
            w, h = self.wh
            l = self.left
            r = self.right
            b = self.bottom
            t = self.top

            tl = qc.QPoint((1.0 - b) * h, l * w)
            size = qc.QSize(w * (1.0 - (l + (1.0 - r))), h * (1.0 - ((1.0 - t) + b)))
            self.x_annotation_rect = qc.QRect(tl, size)

        def set_title(self, title):
            self.title = title

        def plot(
            self,
            xdata=None,
            ydata=None,
            ndecimate=0,
            style="solid",
            color="black",
            line_width=1,
            ignore_nan=False,
            antialiasing=True,
        ):
            """plot data

            :param *args:  ydata | xdata, ydata
            :param ignore_nan: skip values which are nan
            """
            if ydata is None:
                return
            if len(ydata) == 0:
                self.update_datalims([0], [0])
                return

            if xdata is None:
                xdata = num.arange(len(ydata))

            if ignore_nan:
                ydata = num.ma.masked_invalid(ydata)
                xdata = xdata[~ydata.mask]
                ydata = ydata[~ydata.mask]

            if ndecimate != 0:
                # self._xvisible = minmax_decimation(xdata, ndecimate)
                xvisible = xdata[::ndecimate]
                yvisible = minmax_decimation(ydata, ndecimate)
                # self._yvisible = smooth(ydata, window_len=ndecimate*2)[::ndecimate]
                # index = num.arange(0, len(self._xvisible), ndecimate)
            else:
                xvisible = xdata
                yvisible = ydata

            pen = self.get_pen(color, line_width, style)
            if style == "o":
                self.scene_items.append(
                    Points(x=xvisible, y=yvisible, pen=pen, antialiasing=antialiasing)
                )
            else:
                self.scene_items.append(
                    Polyline(x=xvisible, y=yvisible, pen=pen, antialiasing=antialiasing)
                )

            self.update_datalims(xvisible, yvisible)
            self.update()

        def axvline(self, x, **pen_args):
            pen = self.get_pen(**pen_args)
            self.scene_items.append(AxVLine(x=x, pen=pen))

        def axhline(self, y, **pen_args):
            pen = self.get_pen(**pen_args)
            self.scene_items.append(AxHLine(y=y, pen=pen))

        def text(self, x, y, text, **pen_args):
            pen = self.get_pen(**pen_args)
            self.scene_items.append(Text(x=x, y=y, pen=pen, text=text))

        def colormesh(self, x=None, y=None, z=None, **pen_args):

            nx, ny = z.shape
            if not y:
                y = num.arange(ny)

            if not x:
                x = num.arange(nx)

            spec = PColormesh.from_numpy_array(x=x, y=y, z=z)
            self.scene_items.append(spec)
            self.update_datalims(x, y)

            return spec

        def fill_between(self, xdata, ydata1, ydata2, *args, **kwargs):
            x = num.hstack((xdata, xdata[::-1]))
            y = num.hstack((ydata1, ydata2[::-1]))
            self.update_datalims(x, y)

        def plotlog(self, xdata=None, ydata=None, ndecimate=0, **style_kwargs):
            try:
                self.plot(xdata, num.ma.log(ydata), ndecimate=ndecimate, **style_kwargs)
            except ValueError as e:
                logger.info(e)

        def update_datalims(self, xvisible, yvisible):

            if self.ymin is None:
                self._ymin = num.min(yvisible)
            else:
                self._ymin = self.ymin
            if not self.ymax:
                self._ymax = num.max(yvisible)
            else:
                self._ymax = self.ymax

            self.colormap.set_vlim(self._ymin, self._ymax)

            if not self.xmin:
                self._xmin = num.min(xvisible)
            else:
                self._xmin = self.xmin

            if not self.xmax:
                self._xmax = num.max(xvisible)
            else:
                self._xmax = self.xmax

            self.update_projections()

        @property
        def xlim(self):
            return (self._xmin, self._xmax)

        def set_xlim(self, xmin, xmax):
            """ Set x data range. If unset scale to min|max of ydata range """
            self.xmin = xmin
            self.xmax = xmax
            self.xproj.set_in_range(self.xmin, self.xmax)

        @property
        def xrange_visible(self):
            return self.xproj.in_range

        @qc.pyqtSlot(qg.QPaintEvent)
        def paintEvent(self, e):
            """this is executed e.g. when self.repaint() is called. Draws the
            underlying data and scales the content to fit into the widget."""
            painter = qg.QPainter(self)
            self.draw_deco(painter)
            rect = self.canvas_rect()
            for item in self.scene_items:
                item.draw(painter, self.xproj, self.yproj, rect=rect)

        def draw_deco(self, painter):
            painter.save()
            painter.setRenderHint(qg.QPainter.Antialiasing)
            self.draw_axes(painter)

            self.draw_y_ticks(painter)
            self.draw_x_ticks(painter)

            for grid in self.grids:
                grid.draw_grid(self, painter)

            painter.restore()

        def draw_axes(self, painter):
            """ draw x and y axis"""
            w, h = self.wh
            points = [
                qc.QPoint(w * self.left, h * (self.bottom)),
                qc.QPoint(w * self.left, h * (1.0 - self.top)),
                qc.QPoint(w * self.right, h * (1.0 - self.top)),
            ]
            painter.drawPoints(qg.QPolygon(points))

        def set_xtick_increment(self, increment):
            self._xinc = increment

        def set_ytick_increment(self, increment):
            self._yinc = increment

        def draw_x_ticks(self, painter):
            w, h = self.wh
            xmin, xmax, xinc = self.xscaler.make_scale((self._xmin, self._xmax))
            _xinc = self._xinc or xinc
            if self.xticks:
                ticks = num.arange(xmin, xmax, _xinc)
                ticks_proj = self.xproj(ticks)
                tick_anchor = (1.0 - self.top) * h
                lines = [
                    qc.QLineF(xval, tick_anchor * 0.8, xval, tick_anchor)
                    for xval in ticks_proj
                ]

                painter.drawLines(lines)
                if self.xlabels:
                    formatter = self.xtick_formatter
                    for i, xval in enumerate(ticks):
                        painter.drawText(
                            qc.QPointF(ticks_proj[i], tick_anchor * 0.75),
                            formatter % xval,
                        )

        def draw_y_ticks(self, painter):
            w, h = self.wh
            ymin, ymax, yinc = self.yscaler.make_scale((self._ymin, self._ymax))
            #            if self.scroll_increment == 0:
            #                self.scroll_increment = yinc / 4

            _yinc = self._yinc or yinc
            if self.yticks:
                ticks = num.arange(ymin, ymax, _yinc)
                ticks_proj = self.yproj(ticks)
                lines = [
                    qc.QLineF(w * self.left * 0.8, yval, w * self.left, yval)
                    for yval in ticks_proj
                ]
                painter.drawLines(lines)
                if self.ylabels:
                    formatter = self.ytick_formatter
                    for i, yval in enumerate(ticks):
                        painter.drawText(
                            qc.QPointF(0, ticks_proj[i]), formatter % (yval)
                        )

        def draw_labels(self, painter):
            self.setup_annotation_boxes()
            painter.drawText(self.x_annotation_rect, qc.Qt.AlignCenter, "Time")

        @qc.pyqtSlot(qw.QAction)
        def on_tick_increment_select(self, action):
            action_text = action.text()

            if isinstance(action, qw.QWidgetAction):
                return

            if action_text == "Save pitches":
                return

            if action_text == "Minor ticks":
                self.want_minor_grid = action.isChecked()
            else:
                val = int(action.text())
                self.set_grids(val)

        @property
        def want_minor_grid(self):
            return self.__want_minor_grid

        @want_minor_grid.setter
        def want_minor_grid(self, _bool):
            self.__want_minor_grid = _bool
            if _bool:
                self.grids = [self.grid_major, self.grid_minor]
            else:
                self.grids = [self.grid_major]

        def set_grids(self, minor_value):
            self.grid_major = FixGrid(
                minor_value * 5, pen_color="aluminium2", style="solid"
            )
            self.grid_minor = FixGrid(minor_value)
            self.set_ytick_increment(minor_value * 5)

            if self.want_minor_grid:
                self.grids = [self.grid_major, self.grid_minor]
            else:
                self.grids = [self.grid_major]

    return _Axis


Axis = MakeAxis(gl=False)
GLAxis = MakeAxis(gl=True)
GaugeWidget = MakeGaugeWidget(gl=True)


class MikadoWidget(Axis):
    def __init__(self, *args, **kwargs):
        super(MikadoWidget, self).__init__(*args, **kwargs)

    def fill_between(self, xdata1, ydata1, xdata2, ydata2, *args, **kwargs):
        """
        plot only data points which are in both x arrays

        :param xdata1, xdata2: xdata arrays
        :param ydata1, ydata2: ydata arrays
        :param colors: either single color or rgb array
                of length(intersect(xdata1, xdata2))
        """
        indxdata1 = num.in1d(xdata1, xdata2)
        indxdata2 = num.in1d(xdata2, xdata1)

        # this is usually done by *set_data*:
        self._xvisible = num.vstack((xdata1[indxdata1], xdata2[indxdata2]))
        self._yvisible = num.vstack((ydata1[indxdata1], ydata2[indxdata2]))

        self.update_datalims(self._xvisible, self._yvisible)

    @qc.pyqtSlot(qg.QPaintEvent)
    def paintEvent(self, e):
        """this is executed e.g. when self.repaint() is called. Draws the
        underlying data and scales the content to fit into the widget."""

        if len(self._xvisible) == 0:
            return
        painter = qg.QPainter(self)
        lines = []
        pens = []
        dy = num.abs(self._yvisible[0] - self._yvisible[1])

        x = self.xproj(self._xvisible)
        y = self.yproj(self._yvisible)

        for i in range(len(self._xvisible[0])):
            lines.append(qc.QLineF(x[0][i], y[0][i], x[1][i], y[1][i]))
            pens.append(self.colormap.map_to_QPen(dy[i]))

        painter.setRenderHint(qg.QPainter.Antialiasing)
        for iline, line in enumerate(lines):
            painter.save()
            painter.setPen(pens[iline])
            painter.drawLine(line)
            painter.restore()
        self.draw_deco(painter)
