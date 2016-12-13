from pytch.two_channel_tuner import Worker
from pytch.data import MicrophoneRecorder
from pytch.util import dummy

from PyQt5 import QtCore as qc


class Core(qc.QObject):
    dataReady = qc.pyqtSignal()

    def __init__(self, *args, **kwargs):
        qc.QObject.__init__(self, *args, **kwargs)
        buffer_length = 3 * 60.
        fft_size = 512
        self.data_input = MicrophoneRecorder(data_ready_signal=self.dataReady)
        self.worker = Worker(fft_size, self.data_input, buffer_length)

        self.data_input.data_ready_signal.connect(self.worker.process)

    def set_device_no(self, i):
        self.data_input.set_device_no(i)

    def start(self):
        self.data_input.start_new_stream()


if __name__=='__main__':
    c = Core()
    c.start()
