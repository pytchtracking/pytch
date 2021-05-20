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
    """ Returns cross spectrum and phase of *spec1* and *spec2*"""
    cross = spec1 * spec2.conjugate()
    return num.abs(cross), num.unwrap(num.arctan2(cross.imag, cross.real))


def _i(pow_frames):
    """ based on  http://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html"""
    low_freq_mel = 0
    high_freq_mel = 2595 * numpy.log10(1 + (sample_rate / 2) / 700)  # Convert Hz to Mel
    mel_points = numpy.linspace(
        low_freq_mel, high_freq_mel, nfilt + 2
    )  # Equally spaced in Mel scale
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)  # Convert Mel to Hz
    bin = numpy.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = numpy.zeros((nfilt, int(numpy.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])  # left
        f_m = int(bin[m])  # center
        f_m_plus = int(bin[m + 1])  # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = numpy.dot(pow_frames, fbank.T)
    filter_banks = numpy.where(
        filter_banks == 0, numpy.finfo(float).eps, filter_banks
    )  # Numerical Stability
    filter_banks = 20 * numpy.log10(filter_banks)  # dB
