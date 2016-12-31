import unittest
import numpy as num
from PyQt5.QtWidgets import QApplication, QMainWindow
#from PyQt4.QtGui import QApplication, QMainWindow
import sys
from pytch import gui_qt4 as gui
#from pytch import gui
import time


#class GUITestCase(unittest.TestCase):
class GUITestCase():

    def test_scaling(self):
        app = QApplication(sys.argv)
        main_window = gui.MainWindow()

        plot_widget = gui.PlotWidget()
        plot_widget.plot(num.arange(199), num.arange(199))
        main_window.setCentralWidget(plot_widget)
        #plot_widget.set_xlim(0, 20)
        #plot_widget.set_ylim(-50, 150)

        main_window.show()
        main_window.repaint()
        sys.exit(app.exec_())
        print('run')

if __name__=='__main__':
    #unittest.main()
    t = GUITestCase()
    t.test_scaling()


