import logging
import time

logger = logging.getLogger(__name__)


def dummy():
    return


class DummySignal():
    ''' does nothing when emitted'''
    def __init__(self):
        pass

    def emit(self):
        logger.debug('DummySignal emitted')
        return

    def connect(self, *args, **kwargs):
        logger.debug('connected to DummySignal')


class Profiler():
    def __init__(self):
        self.times = []

    def mark(self, m):
        self.times.append((m, time.time()))

    def __str__(self):
        tstart = self.times[0][1]
        s = ''
        for markt in self.times:
            s += '%s: %s\n' % (markt[0], markt[1]-tstart)

        s += 'total: %s' % (self.times[-1][1]-self.times[0][1])
        return s
