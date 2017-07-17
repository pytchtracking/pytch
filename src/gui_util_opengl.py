#!/usr/bin/env python


#############################################################################
##
## Copyright (C) 2015 Riverbank Computing Limited.
## Copyright (C) 2010 Nokia Corporation and/or its subsidiary(-ies).
## All rights reserved.
##
## This file is part of the examples of PyQt. (Modified)
##
## $QT_BEGIN_LICENSE:BSD$
## You may use this file under the terms of the BSD license as follows:
##
## "Redistribution and use in source and binary forms, with or without
## modification, are permitted provided that the following conditions are
## met:
##   * Redistributions of source code must retain the above copyright
##     notice, this list of conditions and the following disclaimer.
##   * Redistributions in binary form must reproduce the above copyright
##     notice, this list of conditions and the following disclaimer in
##     the documentation and/or other materials provided with the
##     distribution.
##   * Neither the name of Nokia Corporation and its Subsidiary(-ies) nor
##     the names of its contributors may be used to endorse or promote
##     products derived from this software without specific prior written
##     permission.
##
## THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
## "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
## LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
## A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
## OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
## SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
## LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
## DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
## THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
## (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
## OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."
## $QT_END_LICENSE$
##
#############################################################################

import sys

import PyQt5.QtCore as qc
import PyQt5.QtGui as qg
import PyQt5.QtWidgets as qw
from pytch.gui_util import make_QPolygonF


class GLWidget(qw.QOpenGLWidget):
    def __init__(self, *args, **kwargs):
        super(GLWidget, self).__init__(*args, **kwargs)

        self.canvas = False

        self.setAutoFillBackground(True)
        self.setMinimumSize(200, 200)
        self.setWindowTitle("Overpainting a Scene")

    def initializeGL(self):
        self.gl = self.context().versionFunctions()
        self.gl.initializeOpenGLFunctions()

    @qc.pyqtSlot(qg.QPaintEvent)
    def paintEvent(self, event):
        self.makeCurrent()

        self.setupViewport(self.width(), self.height())

        painter = qg.QPainter(self)
        painter.setRenderHint(qg.QPainter.Antialiasing)

        #if self.canvas:
        #self.do_draw(painter)

        painter.end()

    def resizeGL(self, width, height):
        self.setupViewport(width, height)

    def showEvent(self, event):
        self.canvas = True

    def sizeHint(self):
        return qc.QSize(400, 400)

    def setupViewport(self, width, height):
        side = min(width, height)
        self.gl.glViewport((width - side) // 2, (height - side) // 2, side,
                side)

        self.gl.glMatrixMode(self.gl.GL_PROJECTION)
        self.gl.glLoadIdentity()
        self.gl.glOrtho(-0.5, +0.5, +0.5, -0.5, 4.0, 15.0)
        self.gl.glMatrixMode(self.gl.GL_MODELVIEW)


if __name__ == '__main__':

    app = qw.QApplication(sys.argv)

    fmt = qg.QSurfaceFormat()
    fmt.setSamples(4)
    qg.QSurfaceFormat.setDefaultFormat(fmt)

    glwindow = GLWidget()
    animationTimer = qc.QTimer()
    animationTimer.setSingleShot(False)
    animationTimer.timeout.connect(glwindow.repaint)
    animationTimer.start(25)

    #window.show()
    #window = QMainWindow()
    window = qw.QWidget()
    layout = qw.QHBoxLayout()
    layout.addWidget(glwindow)
    window.setLayout(layout)
    #window.setCentralWidget(glwindow)
    window.show()
    sys.exit(app.exec_())
