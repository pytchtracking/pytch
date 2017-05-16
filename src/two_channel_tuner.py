# -*- coding: utf-8 -*-

import numpy as num

#from pytch.util import f2pitch

import logging

#_standard_frequency = 220.
logger = logging.getLogger(__name__)


class Worker():

    def __init__(self, channels):
        ''' Grabbing data, working on it and saving the results

        :param buffer_length: in seconds'''

        self.ndata_scale = 16*2
        self.channels = channels
        self.nchannels = len(self.channels)
        self.cross_spectra_combinations = []

    def process(self):
        ''' Do the work'''
        logger.debug('start processing')

        for ic, channel in enumerate(self.channels):
            frame_work = channel.latest_frame_data(channel.fftsize)

            #amp_spec = num.clip(num.log(num.abs(num.fft.rfft(frame_work)))*3000, 0, 2**32)
            amp_spec = num.clip(num.abs(num.fft.rfft(frame_work)), 0, 2**32)
            channel.fft.append(amp_spec)
            #channel.fft_power.append_value(num.sum(amp_spec)/channel.sampling_rate)

            channel.pitch_confidence.append_value(
                channel.pitch_o.get_confidence())
            #channel.pitch.append_value(f2pitch(channel.pitch_o(
            channel.pitch.append_value(channel.pitch_o(
                frame_work)[0])

        logger.debug('finished processing')


def cross_spectrum(spec1, spec2):
    ''' Returns cross spectrum and phase of *spec1* and *spec2*'''
    cross = spec1 * spec2.conjugate()
    return num.abs(cross), num.unwrap(num.arctan2(cross.imag, cross.real))
