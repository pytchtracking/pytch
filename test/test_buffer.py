import numpy as num
import unittest
from subprocess import call
from pytch.data import Buffer
import time

class MicTestCase(unittest.TestCase):

    def test_BufferIndex(self):

        b = Buffer(sampling_rate=10, buffer_length_seconds=10)

        for x in range(5):
            b.append(num.array([x*2, x*2+1]))

        for t in range(10):
            self.assertEqual(b.index_at_time(t), t*10)

    def test_Buffer(self):

        b = Buffer(sampling_rate=1, buffer_length_seconds=10)

        for x in range(5):
            b.append(num.array([x*2, x*2+1]))

        num.testing.assert_array_almost_equal(b.ydata, num.arange(10))
        num.testing.assert_array_almost_equal(b.xdata, num.arange(10))

    def test_benchmark_fill(self):
        iall = 1000
        chunk_length = 1000
        b = Buffer(sampling_rate=1, buffer_length_seconds=chunk_length*iall)

        for i in range(iall):
            t = time.time()
            b.append(num.arange(chunk_length))

        num.testing.assert_array_almost_equal(b.xdata,
                                              num.arange(iall*chunk_length))

        num.testing.assert_array_almost_equal(b.ydata,
                                              num.tile(num.arange(chunk_length),
                                                       iall))

    def test_get_latest_frame(self):
        sampling_rate = 10.
        dt = 1./sampling_rate
        b = Buffer(sampling_rate=sampling_rate, buffer_length_seconds=10)
        b.append(num.arange(20))
        x, y = b.latest_frame(2)
        num.testing.assert_array_almost_equal(y, num.arange(20))
        num.testing.assert_array_almost_equal(x, num.arange(2*sampling_rate, dtype=num.float)*dt)

        b.append(num.arange(20))
        x, y = b.latest_frame(1)
        num.testing.assert_array_almost_equal(y, num.arange(10, 20))
        num.testing.assert_array_almost_equal(x, num.arange(3*sampling_rate,
                                                            4*sampling_rate,
                                                            dtype=num.float)*dt)

if __name__=='__main__':
    unittest.main()
