import unittest
from subprocess import call
from pytch.data import MicrophoneRecorder
import time

class MicTestCase(unittest.TestCase):

    def test_micInit(self):

        # kill pulseaudio to check proper startup
        call(['pulseaudio', '--kill'])

        mic = MicrophoneRecorder()

        mic.start_new_stream()
        mic.terminate()



if __name__=='__main__':
    unittest.main()
