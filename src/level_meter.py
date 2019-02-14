import  PyQt5.QtCore as qc
import  PyQt5.QtGui as qg
import  PyQt5.QtWidgets as qw
import sys

from .gui_util import AdjustableMainWindow

class LevelMeter(qw.QWidget):

    def __init__(self):
        super(LevelMeter, self).__init__()
        layout = qw.QVBoxLayout(self)

        self.level = 0.5

        self.setStyleSheet('''QWidget {
                background-color: None
            }
        ''')
        self.bar_color = qg.QColor('green')
        title = qw.QLabel('Level')
        layout.addWidget(title)

    def paintEvent(self, event):
        painter = qg.QPainter(self)
        bar = self.rect()
        padding = 10
        level = bar.height() - 2*padding - bar.height() * self.level
        bar.setCoords(
            bar.left() + padding, bar.bottom() - padding - level,
            bar.right() - padding, bar.top() + padding,)
        painter.fillRect(bar, self.bar_color)
        painter.end()

    @qc.pyqtSlot(float)
    def levelChaned(self, v):
        '''Set the level scaled between 0. and 1.'''
        logger.debug('new level: %s' % v)

        self.level = v
        self.update()


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
