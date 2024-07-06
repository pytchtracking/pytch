from subprocess import call
from pytch.audio import AudioProcessor
import pytest


@pytest.mark.skip(reason="API of MicrophoneRecorder changed")
def test_mic_init():
    # kill pulseaudio to check proper startup
    try:
        call(["pulseaudio", "--kill"])
    except OSError as e:
        if e.errno == 2:
            pass
        else:
            raise e
    mic = AudioProcessor()
    mic.start_new_stream()
    mic.terminate()
