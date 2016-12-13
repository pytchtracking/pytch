import unittest
from subprocess import call
from pytch.data import MicrophoneRecorder
import time

class MicTestCase(unittest.TestCase):

    def test_micInit(self):

        # kill pulseaudio to check proper startup
        try:
            call(['pulseaudio', '--kill'])
        except OSError as e:
            if e.errno==2:
                return
            else:
                raise e
        mic = MicrophoneRecorder()
            #mic.device_no = 7
        print(mic.device_no)

        mic.start_new_stream()
        print('OPTIONS', mic.sampling_rate_options)
        mic.terminate()



if __name__=='__main__':
    unittest.main()
