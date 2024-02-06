import numpy as num
import pytest
from pytch.audio import Buffer, RingBuffer


def test_BufferIndex():
    b = Buffer(sampling_rate=10, buffer_length_seconds=10)

    for x in range(5):
        b.append(num.array([x * 2, x * 2 + 1]))

    for t in range(10):
        assert b.index_at_time(t) == t * 10


def test_Buffer():
    b = Buffer(sampling_rate=1, buffer_length_seconds=10)

    for x in range(5):
        b.append(num.array([x * 2, x * 2 + 1]))

    num.testing.assert_array_almost_equal(b.ydata, num.arange(10))
    # num.testing.assert_array_almost_equal(b.xdata, num.arange(10))


def test_benchmark_fill():
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


def test_get_latest_frame():
    sampling_rate = 10.0
    dt = 1.0 / sampling_rate
    b = Buffer(sampling_rate=sampling_rate, buffer_length_seconds=60 * 3)
    b.append(num.arange(20))
    x, y = b.latest_frame(2)
    num.testing.assert_array_almost_equal(y, num.arange(20))
    num.testing.assert_array_almost_equal(
        x, num.arange(2 * sampling_rate, dtype=num.float32) * dt
    )

    b.append(num.arange(20))
    x, y = b.latest_frame(1)
    num.testing.assert_array_almost_equal(y, num.arange(10, 20))
    num.testing.assert_array_almost_equal(
        x, num.arange(3 * sampling_rate, 4 * sampling_rate, dtype=num.float32) * dt
    )


@pytest.mark.skip(reason="This test needs to be fixed")
def test_ringbuffer_array():
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


def test_ringbuffer_value():
    r = RingBuffer(1, 3)
    for i in range(4):
        r.append_value(i)
    num.testing.assert_array_equal(r.data, num.array([3.0, 1.0, 2.0]))
    num.testing.assert_array_equal(r.latest_frame_data(2), num.array([2.0, 3.0]))


def test_ringbuffer_array_retrieve_by_time():
    sampling_rate = 10  # Hertz
    buffer_length_seconds = 10
    r = RingBuffer(
        sampling_rate=sampling_rate, buffer_length_seconds=buffer_length_seconds
    )

    d = num.arange(20)  # 2 seconds data at 10 Hz
    r.append(d)

    # check last 5 samples
    x, y = r.latest_frame(3, clip_min=True)
    assert len(x) == len(y)
    assert min(x) >= 0.0

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
