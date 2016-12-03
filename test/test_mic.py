import unittest
from subprocess import call
from pytch import two_channel_tuner
import time

class MicTestCase(unittest.TestCase):

    def test_micInit(self):

        # kill pulseaudio to check proper startup
        #call(['pulseaudio', '--kill'])
        #time.sleep(1)
        print 'asdf'
        #mic = two_channel_tuner.MicrophoneRecorder()
        w = two_channel_tuner.Worker()
        #w.set_device_no(7)
        #w.mic.start_new_stream()
        #w.start()
        #time.sleep(1)


if __name__=='__main__':
    unittest.main()
