import unittest
import numpy as num
from PyQt5.QtWidgets import QApplication, QMainWindow
#from PyQt4.QtGui import QApplication, QMainWindow
import sys
from pytch import gui_qt4 as gui
#from pytch import gui
import time


#class GUITestCase(unittest.TestCase):
class MainWindowQClose(QMainWindow):
    def __init__(self, *args, **kwargs):
        QMainWindow.__init__(self, *args, **kwargs)

    def keyPressEvent(self, key_event):
        ''' react on keyboard keys when they are pressed.'''
        if key_event.text() == 'q':
            self.close()
        QMainWindow.keyPressEvent(self, key_event)


class GUITestCase():

    def test_scaling(self):
        app = QApplication(sys.argv)
        main_window = MainWindowQClose()

        plot_widget = gui.PlotWidget()
        plot_widget.plot(num.arange(199), num.arange(199))
        main_window.setCentralWidget(plot_widget)
        #plot_widget.set_xlim(0, 20)
        #plot_widget.set_ylim(-50, 150)

        main_window.show()
        main_window.repaint()
        sys.exit(app.exec_())

    def test_PitchWidget(self):
        app = QApplication(sys.argv)
        main_window = MainWindowQClose()
        plot_widget = gui.PlotPitchWidget()
        #plot_widget.set_xlim(0, 18)
        #plot_widget.set_ylim(0, 1)
        n = 200


        #x1 = num.array([0,1,2,3,6,7,9,10,14,16])
        #x2 = num.array([0,1,3,4,5,6,8,11,16,17])
        #y1 = num.random.random(len(x1))*1000
        #y2 = num.random.random(len(x1))*1000

        x1 = num.linspace(0, 10., n)
        x2 = num.linspace(0, 10., n)
        y1 = num.random.random(len(x1))*100
        y2 = num.random.random(len(x1))*100


        plot_widget.fill_between(x1, y1, x2, y2)
        plot_widget.colormap.set_vlim(0, 100)
        main_window.setCentralWidget(plot_widget)
        main_window.show()
        main_window.repaint()

        #main_window.close()
        sys.exit(app.exec_())

    def test_ColormapWidget(self):
        app = QApplication(sys.argv)
        main_window = MainWindowQClose()
        cmap = gui.InterpolatedColormap()
        plot_widget = gui.ColormapWidget(cmap)

        main_window.setCentralWidget(plot_widget)
        main_window.show()
        main_window.repaint()
        sys.exit(app.exec_())

if __name__=='__main__':
    #unittest.main()
    t = GUITestCase()
    #t.test_scaling()
    #t.test_PitchWidget()
    t.test_ColormapWidget()


