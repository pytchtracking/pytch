import  PyQt5.QtCore as qc
import  PyQt5.QtGui as qg
import  PyQt5.QtWidgets as qw
import sys
import logging
import numpy as num
from .gui_util import AdjustableMainWindow


logger = logging.getLogger('pytch.level_meter')


class LevelBar(qw.QWidget):
    '''Visual representation of the channel level as a vertical bar.'''
    def __init__(self, enabled=True):
        super(qw.QWidget, self).__init__()
        self.clip_value = 0.5
        self.enabled = enabled

        self.color = qg.QColor('green')
        self.clipped_color = qg.QColor('red')
        self.disabled_color = qg.QColor('grey')

        self.v = 0.

    def get_color(self):
        '''returns normal and clipped color'''
        if self.enabled:
            if not self.clipped():
                return self.color
            else:
                return self.clipped_color
        else:
            return self.disabled_color

    def clipped(self):
        return self.v > self.clip_value

    def update_level(self, v):
        self.v = v

    def set_enabled(self, enabled):
        self.enabled = enabled

    @qc.pyqtSlot()
    def paintEvent(self, event):
        painter = qg.QPainter(self)
        bar = self.rect()
        bar.setTop(bar.height() - bar.height() * self.v)

        painter.fillRect(bar, self.get_color())
        painter.end()


class LevelMeter(qw.QWidget):
    '''Level meter for a single channel'''

    state_changed = qc.pyqtSignal(bool)

    def __init__(self, channel, enabled=True, update_interval=0.1):
        super(LevelMeter, self).__init__()
        self.update_interval = update_interval
        layout = qw.QVBoxLayout(self)

        self.channel = channel
        self.enabled = enabled

        self.setStyleSheet('''QWidget {
                background-color: None
            }
        ''')
        self.title = qw.QLabel('')
        self.level_bar = LevelBar(enabled=enabled)
        self.toggle_button = qw.QPushButton()
        self.toggle_button.released.connect(self.toggle_enabled)
        self.set_button_text()

        layout.addWidget(self.toggle_button)
        layout.addWidget(self.level_bar)
        layout.addWidget(self.title)

        self.refresh_timer = qc.QTimer()
        self.refresh_timer.timeout.connect(self.update_content)
        self.refresh_timer.start(1000*self.update_interval)

        self.setSizePolicy(qw.QSizePolicy.Fixed, qw.QSizePolicy.Preferred)

    def toggle_enabled(self):
        self.enabled = not self.enabled
        self.level_bar.set_enabled(self.enabled)
        self.set_button_text()
        self.state_changed.emit(self.enabled)

    def set_button_text(self):
        self.toggle_button.setText('Disable' if self.enabled else 'Enable')

    def update_content(self):
        v = self.get_level()
        self.title.setText('%1.2f' % v)
        self.level_bar.update_level(v)
        self.level_bar.update()
        self.update()

    def normalization(self):
        '''normalization factor i.e. max bit depth'''
        nbit = 16
        return 2 ** nbit / 2.

    def get_level(self):
        v = num.mean(num.abs(self.channel.latest_frame(self.update_interval)))
        v /= self.normalization()
        logger.debug('new level: %s' % v)
        return v

    def sizeHint(self):
        return qc.QSize(100, 400)


class ChannelMixer(qw.QWidget):

    def __init__(self, channels=None):
        super(ChannelMixer, self).__init__()
        self.setLayout(qw.QHBoxLayout())
        self.set_channels(channels or [])

    def set_channels(self, channels):
        layout = self.layout()
        layout.setAlignment(qc.Qt.AlignLeft)
        for channel in channels:
            logger.debug('Adding channel %s to channel mixer' % channel)
            layout.addWidget(LevelMeter(channel=channel, enabled=False))


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
