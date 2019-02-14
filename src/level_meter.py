import  PyQt5.QtCore as qc
import  PyQt5.QtGui as qg
import  PyQt5.QtWidgets as qw
import sys

from .gui_util import AdjustableMainWindow

class LevelMeter(qw.QWidget):
    '''Level meter for a single channel'''

    def __init__(self, channel, update_interval=0.1):
        super(LevelMeter, self).__init__()
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
        self.refresh_timer.timeout.connect(1000*self.update)
        self.refresh_timer.start(self.update_interval)

    def get_level(self):
        return num.mean(self.channel.latest_frame(self.update_interval))

    def paintEvent(self, event):
        v = self.get_level()
        logger.debug('new level: %s' % v)
        painter = qg.QPainter(self)
        bar = self.rect()
        padding = 10
        level = bar.height() - 2*padding - bar.height() * v
        bar.setCoords(
            bar.left() + padding, bar.bottom() - padding - level,
            bar.right() - padding, bar.top() + padding,)
        painter.fillRect(bar, self.bar_color)
        painter.end()


class ChannelMixer(qw.QWidget):

    def __init__(self, channels):
        super(ChannelMixer, self).__init__()
        layout = self.qw.QHBoxLayout()
        for channel in channels:
            layout.addWidget(LevelMeter(channel=channel))


def app_add_widget():
    '''Convenient for prototyping.

    Create an app, add the `widget` and run the app.
    '''

    app = qw.QApplication(sys.argv)
    widget = LevelMeter()
    win = AdjustableMainWindow()
    win.setCentralWidget(widget)
    print(win)
    win.show()
    app.exec_()


if __name__ == '__main__':
    app_add_widget()
