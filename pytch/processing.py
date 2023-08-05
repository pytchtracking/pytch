import PyQt5.QtCore as qc
import numpy as num
import logging

logger = logging.getLogger("pytch.processing")


class Worker(qc.QObject):
    def __init__(self, channels):
        """
        The Worker does the signal processing in its' `process` method.

        :param channels: list of `pytch.data.Channel` instances"""

        super().__init__()
        self.channels = channels

    def process(self):
        """Process the channels' data and update the channel instances."""
        logger.debug("processing data")

        for ic, channel in enumerate(self.channels):
            frame_work = channel.latest_frame_data(channel.fftsize)
            win = num.hanning(channel.fftsize)
            # slight pre-emphasis
            # frame_work[1:] -=  0.1 * frame_work[:-1]
            # frame_work[0] = frame_work[1]

            amp_spec = num.abs(num.fft.rfft(frame_work * win)) ** 2 / channel.fftsize
            channel.fft.append(num.asarray(amp_spec, dtype=num.uint32))

            channel.pitch_confidence.append_value(channel.pitch_o.get_confidence())

            channel.pitch.append_value(channel.pitch_o(frame_work)[0])


def cross_spectrum(spec1, spec2):
    """Returns cross spectrum and phase of *spec1* and *spec2*"""
    cross = spec1 * spec2.conjugate()
    return num.abs(cross), num.unwrap(num.arctan2(cross.imag, cross.real))
