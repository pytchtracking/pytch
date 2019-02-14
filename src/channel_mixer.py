import  PyQt5.QtCore as qc
import  PyQt5.QtGui as qg
import  PyQt5.QtWidgets as qw
import sys
import logging
import numpy as num
from .gui_util import AdjustableMainWindow


logger = logging.getLogger('pytch.level_meter')

class LevelMeter(qw.QWidget):
    '''Level meter for a single channel'''

    def __init__(self, channel, update_interval=0.1):
        super(LevelMeter, self).__init__()
        self.update_interval = update_interval
        layout = qw.QVBoxLayout(self)

        self.channel = channel
        self.level = 0.5

        self.setStyleSheet('''QWidget {
                background-color: None
            }
        ''')
        self.bar_color = qg.QColor('green')
        title = qw.QLabel('Level')
        layout.addWidget(title)

        self.refresh_timer = qc.QTimer()
        self.refresh_timer.timeout.connect(self.update)
        self.refresh_timer.start(1000*self.update_interval)

    def normalization(self):
        '''normalization factor i.e. max bit depth'''
        nbit = 16
        return 2 ** nbit / 2.

    def get_level(self):
        v = num.mean(num.abs(self.channel.latest_frame(self.update_interval)))
        v /= self.normalization()
        logger.debug('new level: %s' % v)
        return v

    def paintEvent(self, event):
        v = self.get_level()
        painter = qg.QPainter(self)
        bar = self.rect()
        padding = 10
        bar.setCoords(
            bar.left() + padding, bar.top() - padding,
            bar.right() - padding, bar.bottom() + padding,)

        level = bar.height() * v
        bar.setTop(bar.height() - level)

        painter.fillRect(bar, self.bar_color)
        painter.end()


class ChannelMixer(qw.QWidget):

    def __init__(self, channels=None):
        super(ChannelMixer, self).__init__()
        self.setLayout(qw.QHBoxLayout())
        self.set_channels(channels or [])

    def set_channels(self, channels):
        layout = self.layout()
        for channel in channels:
            logger.debug('Adding channel %s to channel mixer' % channel)
            layout.addWidget(LevelMeter(channel=channel))


class DummyChannel():
    def __init__(self, v):
        self.v = v

    def latest_frame(self, n):
        return self.v


def app_add_widget():
    '''Convenient for prototyping.

    Create an app, add the `widget` and run the app.
    '''

    app = qw.QApplication(sys.argv)

    channels = [DummyChannel(0.5 * 2**16), DummyChannel(0.3 * 2**16), DummyChannel(0.)]

    widget = ChannelMixer(channels=channels)
    win = AdjustableMainWindow()
    win.setCentralWidget(widget)
    print(win)
    win.show()
    app.exec_()


if __name__ == '__main__':
    app_add_widget()
