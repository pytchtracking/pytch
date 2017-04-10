import numpy as num
import math
import scipy.interpolate as interpolate
import logging

from pytch.gui_util import AutoScaler, Projection, minmax_decimation
from pytch.gui_util import make_QPolygonF, _color_names, _colors, _pen_styles    # noqa

from PyQt5 import QtCore as qc
from PyQt5 import QtGui as qg
from PyQt5.QtWidgets import QWidget, QSizePolicy, QApplication


d2r = num.pi/180.
logger = logging.getLogger(__name__)

#try:
#    from pytch.gui_util_opengl import GLWidget
#    __PlotSuperClass = GLWidget
#except ImportError:
logger.warn('no opengl support')

class PlotWidgetBase(QWidget):

    def paintEvent(self, e):
        painter = qg.QPainter(self)

        self.do_draw(painter)

    def do_draw(self, painter):
        raise Exception('to be implemented in subclass')

__PlotSuperClass = PlotWidgetBase


class PlotBase(__PlotSuperClass):
    def __init__(self, *args, **kwargs):
        super(PlotBase, self).__init__(*args, **kwargs)
        self.wheel_pos = 0

    @qc.pyqtSlot(qg.QMouseEvent)
    def wheelEvent(self, wheel_event):
        self.wheel_pos += wheel_event.angleDelta().y()
        n = self.wheel_pos / 120
        self.wheel_pos = self.wheel_pos % 120
        if n == 0:
            return

        modifier = QApplication.keyboardModifiers()
        if modifier & qc.Qt.ShiftModifier:
            self.set_ylim(self.ymin-self.scroll_increment*n,
                          self.ymax+self.scroll_increment*n)
            logger.info('SHIFT')
        elif modifier & qc.Qt.AltModifier:
            self.set_ylim(self.ymin-self.scroll_increment*n,
                          self.ymax+self.scroll_increment*n)
            logger.info('ALT')
        elif modifier & qc.Qt.ControlModifier:
            self.set_ylim(self.ymin-self.scroll_increment*n,
                          self.ymax+self.scroll_increment*n)
            logger.info('CONTROL')
        else:
            self.set_ylim(self.ymin-self.scroll_increment*n,
                          self.ymax-self.scroll_increment*n)
        self.repaint()


class InterpolatedColormap:
    ''' Continuously interpolating colormap '''
    def __init__(self, name=''):
        self.name = name
        self.colors = num.array([
                _colors['red'], _colors['green'], _colors['blue']
            ])

        self.values = num.linspace(0, 255., len(self.colors))
        self.r_interp = interpolate.interp1d(self.values, self.colors.T[0])
        self.g_interp = interpolate.interp1d(self.values, self.colors.T[1])
        self.b_interp = interpolate.interp1d(self.values, self.colors.T[2])
        self.proj = Projection()
        self.proj.set_out_range(0, 255.)

    def update(self):
        pass

    def _map(self, val):
        ''' Interpolate RGB colormap for *val*
        val can be a 1D array.

        Values which are out of range are clipped.
        '''
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
        ''' has to be implemented by every subclass. Needed for plotting.'''
        mi, ma = self.proj.get_in_range()
        return num.linspace(mi, ma, n)

    def get_visualization(self, callback=None):
        '''get dict of values and colors for visualization.

        :param callback: method to retrieve colors from value range.
                        default: *map*
        '''
        vals = self.get_incremented_values()

        if callback:
            colors = [callback(v) for v in vals]
        else:
            colors = [self.map(v) for v in vals]

        return vals, colors

    def __call__(self, val):
        return self._map(val)


class Colormap(InterpolatedColormap):
    ''' Like Colormap but with discrete resolution and precalculated.
    Can return tabulated QColors. Faster than Colormap'''

    def __init__(self, name='', n=20):
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
        return num.linspace(*self.proj.xr, num=self.n+1)

    def get_index(self, val):
        return int(self.proj.clipped(val)/self.proj.ur[1] * self.n)

    def map(self, val):
        return self.colors_rgb[self.get_index(val)]

    def map_to_QColor(self, val):
        return self.colors_QColor[self.get_index(val)]

    def map_to_QPen(self, val):
        i = self.get_index(val)
        return self.colors_QPen[i]


class ColormapWidget(QWidget):
    def __init__(self, colormap, *args, **kwargs):
        QWidget.__init__(self, *args, **kwargs)
        self.colormap = colormap
        self.yproj = Projection()
        size_policy = QSizePolicy()
        size_policy.setHorizontalPolicy(QSizePolicy.Maximum)
        self.setSizePolicy(size_policy)

        self.update()

    def update(self):
        _, rgb = self.colormap.get_visualization()
        self.vals, self.colors = self.colormap.get_visualization(
            callback=self.colormap.map_to_QColor)
        self.yproj.set_in_range(num.min(self.vals), num.max(self.vals))

    def paintEvent(self, e):
        rect = self.rect()
        self.yproj.set_out_range(rect.top(), rect.bottom())

        yvals = self.yproj(self.vals)
        painter = qg.QPainter(self)
        for i in range(len(self.vals)-1):
            patch = qc.QRect(qc.QPoint(rect.left(), yvals[i]),
                             qc.QPoint(rect.right(), yvals[i+1]))
            painter.save()
            painter.fillRect(patch, qg.QBrush(self.colors[i]))
            painter.restore()

    def sizeHint(self):
        return qc.QSize(100, 400)



class GaugeWidget(PlotBase):
    def __init__(self, *args, **kwargs):
        super(GaugeWidget, self).__init__(*args, **kwargs)

        self.color = qg.QColor(0, 100, 100)
        self._val = 0
        self.set_title('')
        self.proj = Projection()
        out_min = -130.
        out_max = 130.
        self.proj.set_out_range(out_min, out_max)
        self.set_ylim(0., 1500.)
        self.scroll_increment = 100

        self.scaler = AutoScaler(
            no_exp_interval=(-3, 2), approx_ticks=7,
            snap=True
        )

        size_policy = QSizePolicy()
        size_policy.setHorizontalPolicy(QSizePolicy.Minimum)
        self.setSizePolicy(size_policy)
        self.xtick_increment = 20
        self.pen = qg.QPen(self.color, 20, qc.Qt.SolidLine)
        self.pen.setCapStyle(qc.Qt.FlatCap)
        self.wheel_pos = 0

    def set_ylim(self, ymin, ymax):
        ''' Set range of Gauge.'''
        self.ymin = ymin
        self.ymax = ymax
        self.proj.set_in_range(self.ymin, self.ymax)

    def get_ylim(self):
        return self.ymin, self.ymax

    def do_draw(self, painter):
        ''' This is executed when self.repaint() is called'''
        painter.save()
        self.side = min(self.width(), self.height())/1.05
        self.halfside = self.side/2.
        rect = qc.QRectF(-self.halfside, -self.halfside, self.side, self.side)
        painter.translate(self.width()/2., self.height()/2.)
        painter.save()
        painter.setPen(self.pen)
        self.arc_start = -(self.proj(0)+180) * 16
        pmin, pmax = self.proj.get_in_range()
        if self._val:
            span_angle = self.proj(self._val)*16 + self.arc_start+180*16.
            painter.drawArc(rect, self.arc_start, -span_angle)
        painter.restore()

        self.draw_deco(painter)
        painter.restore()

    def draw_deco(self, painter):
        self.draw_ticks(painter)
        self.draw_title(painter)

    def draw_title(self, painter):
        painter.drawText(-40., 2., self.title)

    def draw_ticks(self, painter):
        # needs some performance polishing !!!
        xmin, xmax = self.proj.get_in_range()

        ticks = num.arange(xmin, xmax+self.xtick_increment,
                           self.xtick_increment, dtype=num.int)
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
                rotate_rad = degree*d2r
                x = self.halfside*0.8*num.cos(rotate_rad) - text_box_with/2
                y = self.halfside*0.8*num.sin(rotate_rad)
                rect = qc.QRectF(x, y, text_box_with, text_box_with/5)
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
        '''
        Call this method to update the arc
        '''
        self._val = val

    def sizeHint(self):
        return qc.QSize(400, 400)


class Grid():
    def __init__(self, horizontal=True, vertical=True, *args, **kwargs):
        self.vertical = vertical
        self.horizontal = horizontal

    def draw_grid(self, widget, painter):
        return


class AutoGrid():
    def __init__(self, pen_color='aluminium2', style=':', line_width=1, *args, **kwargs):
        Grid.__init__(self, *args, **kwargs)

        self.data_lims_v = (None, None)
        self.data_lims_h = (None, None)
        self.lines_h = []
        self.lines_v = []

        self.set_pen(pen_color, style, line_width)

    def set_pen(self, pen_color, style, line_width):
        self.grid_pen = qg.QPen(
            qg.QColor(*_colors[pen_color]), line_width, _pen_styles[style])

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
        ''' setup horizontal grid lines'''

        #if not (widget._ymin, widget._ymax) == self.data_lims_h:
        ymin, ymax, yinc = widget.yscaler.make_scale(
            (widget._ymin, widget._ymax)
        )
        ticks_proj = widget.yproj(num.arange(ymin, ymax, yinc))

        w, h = widget.wh
        self.lines_h = [qc.QLineF(w * widget.left, yval, w, yval)
                 for yval in ticks_proj]
        self.data_lims_h = (widget._ymin, widget._ymax)

        return self.lines_h

    def lines_vertical(self, widget, painter):
        ''' setup vertical grid lines'''

        #if not (widget._xmin, widget._xmax) == self.data_lims_v:
        xmin, xmax, xinc = widget.xscaler.make_scale(
            (widget._xmin, widget._xmax)
        )
        ticks_proj = widget.xproj(num.arange(xmin, xmax, xinc))

        w, h = widget.wh
        self.lines_v = [qc.QLineF(xval, h * widget.top, xval, h)
                 for xval in ticks_proj]
        self.data_lims_v = (widget._xmin, widget._xmax)

        return self.lines_v


class FixGrid(AutoGrid):
    def __init__(self, delta, *args, **kwargs):
        self.delta = delta
        AutoGrid.__init__(self, *args, **kwargs)

    def lines_horizontal(self, widget, painter):
        ''' setup horizontal grid lines'''

        #if not (widget._ymin, widget._ymax) == self.data_lims_h:
        ymin, ymax, yinc = widget.yscaler.make_scale(
            (widget._ymin, widget._ymax)
        )
        ticks_proj = widget.yproj(num.arange(ymin, ymax, self.delta))

        w, h = widget.wh
        self.lines_h = [qc.QLineF(w * widget.left, yval, w, yval)
                 for yval in ticks_proj]
        self.data_lims_h = (widget._ymin, widget._ymax)

        return self.lines_h

    def lines_vertical(self, widget, painter):
        ''' setup vertical grid lines'''

        #if not (widget._xmin, widget._xmax) == self.data_lims_v:
        xmin, xmax, xinc = widget.xscaler.make_scale(
            (widget._xmin, widget._xmax)
        )
        ticks_proj = widget.xproj(num.arange(xmin, xmax, self.delta))

        w, h = widget.wh
        self.lines_v = [qc.QLineF(xval, h * widget.top, xval, h)
                 for xval in ticks_proj]
        self.data_lims_v = (widget._xmin, widget._xmax)

        return self.lines_v



class AxHLine():
    def __init__(self, y, pen):
        self.y = y
        self.pen = pen

    def draw(self, painter, xproj, yproj, rect=None):
        xmin, xmax = xproj.get_out_range()
        y = yproj(self.y)

        painter.save()
        painter.setPen(self.pen)
        painter.drawLine(xmin, y, xmax, y)
        painter.restore()


class AxVLine():
    def __init__(self, x, pen):
        self.x = x
        self.pen = pen

    def draw(self, painter, xproj, yproj, rect=None):
        ymin, ymax = yproj.get_out_range()
        x = xproj(self.x)

        painter.save()
        painter.setPen(self.pen)
        painter.drawLine(x, ymin, x, ymax)
        painter.restore()


class Points():
    ''' Holds and draws data projected to screen dimensions.'''
    def __init__(self, x, y, pen):
        self.x = x
        self.y = y
        self.pen = pen

    def draw(self, painter, xproj, yproj, rect=None):
        qpoints = make_QPolygonF(xproj(self.x), yproj(self.y))
        painter.save()
        painter.setPen(self.pen)
        painter.drawPoints(qpoints)
        painter.restore()



class Polyline():
    ''' Holds and draws data projected to screen dimensions.'''
    def __init__(self, x, y, pen):
        self.x = x
        self.y = y
        self.pen = pen

    def draw(self, painter, xproj, yproj, rect=None):
        qpoints = make_QPolygonF(xproj(self.x), yproj(self.y))

        painter.save()
        painter.setPen(self.pen)
        painter.drawPolyline(qpoints)
        painter.restore()


class Spectrogram(PlotBase):
    def __init__(self, img, x, y, *args, **kwargs):
        super(Spectrogram, self).__init__(*args, **kwargs)
        self.img = img
        self.x = x
        self.y = y
        self.rect = None

    def draw(self, painter, xproj, yproj, rect=None):
        painter.drawImage(rect, self.img)

    @classmethod
    def from_numpy_array(cls, x, y, z):
        '''
        :param a: RGB array
        '''
        img = qg.QImage(
            z.ctypes.data, z.shape[1], z.shape[0], qg.QImage.Format_RGB32)
        return cls(img, x, y)


class PlotWidget(PlotBase):
    ''' a plotwidget displays data (x, y coordinates). '''

    def __init__(self, *args, **kwargs):
        super(PlotWidget, self).__init__(*args, **kwargs)

        self.setContentsMargins(0, 0, 0, 0)
        self.scroll_increment = 0
        self.track_start = None

        self.ymin = None
        self.ymax = None
        self.xmin = None
        self.xmax = None

        self._ymin = 0.
        self._ymax = 1.
        self._yinc = None
        self._xmin = 0.
        self._xmax = 1.
        self._xinc = None

        self.left = 0.15
        self.right = 1.
        self.bottom = 0.1
        self.top = 0.05

        self.xlabels = True
        self.ylabels = True
        self.yticks = True
        self.xticks = True
        self.xzoom = 0.

        self.clear()
        self.grids = [AutoGrid()]
        self.yscaler = AutoScaler(
            no_exp_interval=(-3, 2), approx_ticks=5,
            snap=True
        )

        self.xscaler = AutoScaler(
            no_exp_interval=(-3, 2), approx_ticks=5,
            snap=True
        )
        self.draw_fill = False
        self.draw_points = False
        self.set_brush()
        self._xvisible = num.empty(0)
        self._yvisible = num.empty(0)
        self.yproj = Projection()
        self.xproj = Projection()
        self.colormap = Colormap()
        self.set_background_color('white')

    def clear(self):
        self.scene_items = []

    def setup_annotation_boxes(self):
        ''' left and top boxes containing labels, dashes, marks, etc.'''
        w, h = self.wh
        l = self.left
        r = self.right
        t = self.bottom
        b = self.top

        tl = qc.QPoint((1.-b) * h, l * w)
        size = qc.QSize(w * (1. - (l + (1.-r))),
                        h * (1. - ((1.-t)+b)))
        self.x_annotation_rect = qc.QRect(tl, size)

    @property
    def wh(self):
        return self.width(), self.height()

    def keyPressEvent(self, key_event):
        ''' react on keyboard keys when they are pressed.'''
        key_text = key_event.text()
        if key_text == 'q':
            self.close()

        elif key_text == 'f':
            self.showMaximized()

        QWidget.keyPressEvent(self, key_event)

    def set_brush(self, color='black'):
        self.brush = qg.QBrush(qg.QColor(*_colors[color]))

    def set_pen_color(self, color='black'):
        self.pen.setColor(qg.QColor(*_colors[color]))

    def get_pen(self, color='black', line_width=1, style='solid'):
        '''
        :param color: color name as string
        '''
        if style == 'o':
            self.draw_points = True

        return qg.QPen(qg.QColor(*_colors[color]),
                      line_width, _pen_styles[style])

    def set_title(self, title):
        self.title = title

    def plot(self, xdata=None, ydata=None, ndecimate=0,
             style='solid', color='black', line_width=1, ignore_nan=False):
        ''' plot data

        :param *args:  ydata | xdata, ydata
        :param ignore_nan: skip values which are nan
        '''
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
            #self._xvisible = minmax_decimation(xdata, ndecimate)
            xvisible = xdata[::ndecimate]
            yvisible = minmax_decimation(ydata, ndecimate)
            #self._yvisible = smooth(ydata, window_len=ndecimate*2)[::ndecimate]
            #index = num.arange(0, len(self._xvisible), ndecimate)
        else:
            xvisible = xdata
            yvisible = ydata

        pen = self.get_pen(color, line_width, style)
        if style == 'o':
            self.scene_items.append(Points(x=xvisible, y=yvisible, pen=pen))
        else:
            self.scene_items.append(Polyline(x=xvisible, y=yvisible, pen=pen))

        self.update_datalims(xvisible, yvisible)

    def axvline(self, x, **pen_args):
        pen = self.get_pen(**pen_args)
        self.scene_items.append(AxVLine(x=x, pen=pen))

    def axhline(self, y, **pen_args):
        pen = self.get_pen(**pen_args)
        self.scene_items.append(AxHLine(y=y, pen=pen))

    def colormesh(self, x, y, z, **pen_args):
        spec = Spectrogram.from_numpy_array(x=x, y=y, z=z)
        self.scene_items.append(spec)

    def fill_between(self, xdata, ydata1, ydata2, *args, **kwargs):
        x = num.hstack((xdata, xdata[::-1]))
        y = num.hstack((ydata1, ydata2[::-1]))
        self.draw_fill = True
        self.set_data(x, y)

    def plotlog(self, xdata=None, ydata=None, ndecimate=0, **style_kwargs):
        try:
            self.plot(xdata, num.log(ydata), ndecimate=ndecimate, **style_kwargs)
        except ValueError as e:
            logger.info(e)

    def update_datalims(self, xvisible, yvisible):

        xvisible_max = num.max(xvisible)

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
            self._xmax = xvisible_max
        else:
            self._xmax = self.xmax

        w, h = self.wh

        mi, ma = self.xproj.get_out_range()
        drange = ma-mi
        xzoom = self.xzoom * drange
        self.xproj.set_in_range(self._xmin - xzoom, self._xmax)
        self.xproj.set_out_range(w * self.left, w * self.right)

        self.yproj.set_in_range(self._ymin, self._ymax)
        self.yproj.set_out_range(
            h*(1-self.bottom),
            h*self.top,)

    def set_background_color(self, color):
        '''
        :param color: color as string
        '''
        background_color = qg.QColor(*_colors[color])
        self.background_brush = qg.QBrush(background_color)

    @property
    def xlim(self):
        return (self._xmin, self._xmax)

    def set_xlim(self, xmin, xmax):
        ''' Set x data range. If unset scale to min|max of ydata range '''
        self.xmin = xmin
        self.xmax = xmax
        self.xproj.set_in_range(self.xmin, self.xmax)

    def set_ylim(self, ymin, ymax):
        ''' Set x data range. If unset scales to min|max of ydata range'''
        self.ymin = ymin
        self.ymax = ymax
        self.yproj.set_in_range(self.ymin, self.ymax)

    @property
    def xrange_visible(self):
        return self.xproj.in_range

    def do_draw(self, painter):
        ''' this is executed e.g. when self.repaint() is called. Draws the
        underlying data and scales the content to fit into the widget.'''
        self.draw_background(painter)
        self.draw_deco(painter)
        rect = self.rect()
        for item in self.scene_items:
            item.draw(painter, self.xproj, self.yproj, rect=rect)

    def draw_deco(self, painter):
        painter.save()
        painter.setRenderHint(qg.QPainter.Antialiasing)
        self.draw_axes(painter)

        if self.yticks:
            self.draw_y_ticks(painter)
        if self.xticks:
            self.draw_x_ticks(painter)

        for grid in self.grids:
            grid.draw_grid(self, painter)

        painter.restore()

    def draw_axes(self, painter):
        ''' draw x and y axis'''
        w, h = self.wh
        points = [qc.QPoint(w*self.left, h*(1.-self.bottom)),
                  qc.QPoint(w*self.left, h*(1.-self.top)),
                  qc.QPoint(w*self.right, h*(1.-self.top))]
        painter.drawPoints(qg.QPolygon(points))

    def set_xtick_increment(self, increment):
        self._xinc = increment

    def set_ytick_increment(self, increment):
        self._yinc = increment

    def draw_x_ticks(self, painter):
        w, h = self.wh
        xmin, xmax, xinc = self.xscaler.make_scale((self._xmin, self._xmax))
        _xinc = self._xinc or xinc
        ticks = num.arange(xmin, xmax, _xinc)
        ticks_proj = self.xproj(ticks)
        tick_anchor = self.top*h
        lines = [qc.QLineF(xval, tick_anchor * 0.8, xval, tick_anchor)
                 for xval in ticks_proj]

        painter.drawLines(lines)
        if self.xlabels:
            for i, xval in enumerate(ticks):
                painter.drawText(qc.QPointF(ticks_proj[i], tick_anchor), str(xval))

    def draw_y_ticks(self, painter):
        w, h = self.wh
        ymin, ymax, yinc = self.yscaler.make_scale(
            (self._ymin, self._ymax)
        )
        if self.scroll_increment == 0:
            self.scroll_increment = yinc/4

        _yinc = self._yinc or yinc
        ticks = num.arange(ymin, ymax, _yinc)
        ticks_proj = self.yproj(ticks)
        lines = [qc.QLineF(w * self.left * 0.8, yval, w*self.left, yval)
                 for yval in ticks_proj]
        painter.drawLines(lines)
        if self.ylabels:
            for i, yval in enumerate(ticks):
                painter.drawText(qc.QPointF(0, ticks_proj[i]), str(yval))

    def draw_labels(self, painter):
        self.setup_annotation_boxes()
        painter.drawText(self.x_annotation_rect, qc.Qt.AlignCenter, 'Time')

    def draw_background(self, painter):
        painter.fillRect(self.rect(), self.background_brush)

    #def mousePressEvent(self, mouse_ev):
    #    point = self.mapFromGlobal(mouse_ev.globalPos())
    #    if mouse_ev.button() == qc.Qt.LeftButton:
    #        self.track_start = (point.x(), point.y())
    #        self.last_y = point.y()
    #    else:
    #        super(PlotWidget, self).mousePressEvent(mouse_ev)

    #def mouseReleaseEvent(self, mouse_event):
    #    self.track_start = None

    #def mouseMoveEvent(self, mouse_ev):
    #    ''' from pyrocko's pile viewer'''
    #    point = self.mapFromGlobal(mouse_ev.globalPos())

    #    if self.track_start is not None:
    #        x0, y0 = self.track_start
    #        dx = (point.x()- x0)/float(self.width())
    #        dy = (point.y() - y0)/float(self.height())
    #        #if self.ypart(y0) == 1:
    #        #dy = 0

    #        tmin0, tmax0 = self.xlim

    #        scale = math.exp(-dy*5.)
    #        dtr = scale*(tmax0-tmin0) - (tmax0-tmin0)
    #        frac = x0/ float(self.width())
    #        dt = dx*(tmax0-tmin0)*scale

    #        #self.interrupt_following()
    #        self.set_xlim(
    #            tmin0 - dt - dtr*frac,
    #            tmax0 - dt + dtr*(1.-frac))
    #        self.update()


class MikadoWidget(PlotWidget):
    def __init__(self, *args, **kwargs):
        super(MikadoWidget, self).__init__(*args, **kwargs)

    def fill_between(self, xdata1, ydata1, xdata2, ydata2, *args, **kwargs):
        '''
        plot only data points which are in both x arrays

        :param xdata1, xdata2: xdata arrays
        :param ydata1, ydata2: ydata arrays
        :param colors: either single color or rgb array
                of length(intersect(xdata1, xdata2))
        '''
        indxdata1 = num.in1d(xdata1, xdata2)
        indxdata2 = num.in1d(xdata2, xdata1)

        # this is usually done by *set_data*:
        self._xvisible = num.vstack((xdata1[indxdata1], xdata2[indxdata2]))
        self._yvisible = num.vstack((ydata1[indxdata1], ydata2[indxdata2]))

        self.update_datalims(self._xvisible, self._yvisible)

    def do_draw(self, painter):
        ''' this is executed e.g. when self.repaint() is called. Draws the
        underlying data and scales the content to fit into the widget.'''

        if len(self._xvisible) == 0:
            return

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
