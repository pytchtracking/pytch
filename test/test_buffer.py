import numpy as num
import unittest
from pytch.data import Buffer, RingBuffer
import time


class BufferTestCase(unittest.TestCase):
    def test_BufferIndex(self):

        b = Buffer(sampling_rate=10, buffer_length_seconds=10)

        for x in range(5):
            b.append(num.array([x * 2, x * 2 + 1]))

        for t in range(10):
            self.assertEqual(b.index_at_time(t), t * 10)

    def test_Buffer(self):

        b = Buffer(sampling_rate=1, buffer_length_seconds=10)

        for x in range(5):
            b.append(num.array([x * 2, x * 2 + 1]))

        num.testing.assert_array_almost_equal(b.ydata, num.arange(10))
        # num.testing.assert_array_almost_equal(b.xdata, num.arange(10))

    def test_benchmark_fill(self):
        iall = 100
        sampling_rate = 44100
        blength = 3 * 60.0
        chunk_length = 0.025 * sampling_rate
        b = Buffer(sampling_rate=sampling_rate, buffer_length_seconds=blength)

        for i in range(iall):
            b.append(num.arange(i * chunk_length, (i + 1) * chunk_length))
            (
                x,
                y,
            ) = b.latest_frame(5)

        # num.testing.assert_array_almost_equal(b.xdata,
        #                                      num.arange(iall*chunk_length))

        # num.testing.assert_array_almost_equal(b.ydata/sampling_rate,
        #                                      num.tile(num.arange(chunk_length),
        # Jiall))

    def test_get_latest_frame(self):
        sampling_rate = 10.0
        dt = 1.0 / sampling_rate
        b = Buffer(sampling_rate=sampling_rate, buffer_length_seconds=60 * 3)
        b.append(num.arange(20))
        x, y = b.latest_frame(2)
        num.testing.assert_array_almost_equal(y, num.arange(20))
        num.testing.assert_array_almost_equal(
            x, num.arange(2 * sampling_rate, dtype=num.float) * dt
        )

        b.append(num.arange(20))
        x, y = b.latest_frame(1)
        num.testing.assert_array_almost_equal(y, num.arange(10, 20))
        num.testing.assert_array_almost_equal(
            x, num.arange(3 * sampling_rate, 4 * sampling_rate, dtype=num.float) * dt
        )

    def test_ringbuffer_array(self):
        r = RingBuffer(1, 10)
        d = num.arange(4)
        num.testing.assert_equal(len(r.latest_frame_data(5)), 5)
        r.append(d)
        num.testing.assert_array_equal(
            r.latest_frame_data(5), num.array([9.0, 0.0, 1.0, 2.0, 3.0])
        )

        r.append(d)
        num.testing.assert_array_equal(
            r.latest_frame_data(5), num.array([3.0, 0.0, 1.0, 2.0, 3.0])
        )

        r.append(d)
        num.testing.assert_array_equal(
            r.latest_frame_data(5), num.array([3.0, 0.0, 1.0, 2.0, 3.0])
        )

        num.testing.assert_array_equal(
            r.data, num.array([2.0, 3.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0])
        )

    def test_ringbuffer_value(self):
        r = RingBuffer(1, 3)
        for i in range(4):
            r.append_value(i)
        num.testing.assert_array_equal(r.data, num.array([3.0, 1.0, 2.0]))
        num.testing.assert_array_equal(r.latest_frame_data(2), num.array([2.0, 3.0]))

    def test_ringbuffer_array_retrieve_by_time(self):
        sampling_rate = 10  # Herz
        buffer_length_seconds = 10
        r = RingBuffer(
            sampling_rate=sampling_rate, buffer_length_seconds=buffer_length_seconds
        )

        d = num.arange(20)  # 2 seconds data at 10 Hz
        r.append(d)

        # check last 5 samples
        x, y = r.latest_frame(3, clip_min=True)
        self.assertEqual(len(x), len(y))
        self.assertTrue(min(x) >= 0.0)

        # check last 5 samples
        y = r.latest_frame_data(5)
        num.testing.assert_equal(y, d[-5:])

        x, y = r.latest_frame(2)
        num.testing.assert_equal(y[1:], d)
        num.testing.assert_array_almost_equal(
            num.asarray(x[:-1], num.float64), d / float(sampling_rate)
        )

        d = num.arange(100)  # 5 seconds data at 10 Hz
        r.append(d)
        x, y = r.latest_frame(3, clip_min=True)
        num.testing.assert_array_almost_equal(
            num.asarray(x[:-1], num.float64), num.arange(90, 120) / sampling_rate
        )


if __name__ == "__main__":
    unittest.main()
