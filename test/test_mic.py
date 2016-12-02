from unittest import TestCase
from pytch import two_channel_tuner

class MicTestCase(TestCase):

    def testMicInit(self):

        mic = two_channel_tuner.MicrophoneRecorder()
        mic.start()
        mic.stop()
        mic.terminate()
