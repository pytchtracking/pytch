import logging

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

