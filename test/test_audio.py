import numpy as np

from pytch.audio import (
    RingBuffer,
    AudioProcessor,
    get_input_devices,
    get_fs_options,
)

print(get_input_devices())
print(get_fs_options(1))


buf = RingBuffer(size=(20, 1), dtype=np.float64)
buf.write(np.arange(20).reshape(-1, 1))
test = buf.read(20)

#
ap = AudioProcessor(
    fs=8000,
    buf_len_sec=10.0,
    fft_len=1024,
    hop_len=256,
    blocksize=4096,
    channels=[0],
    device_no=1,
    f0_algorithm="YIN",
)
ap.start_stream()