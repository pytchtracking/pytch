#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test Ring Buffer"""
import numpy as np
from pytch.audio import RingBuffer


def test_ring_buffer():
    test_data = np.random.rand(20, 60, 4).astype(np.float32)
    buf = RingBuffer(
        size=(60, test_data.shape[1], test_data.shape[2]), dtype=np.float32
    )
    buf.write(test_data)
    assert np.all(buf.read_latest(20) == test_data)
    assert np.all(buf.read(10, 10) == test_data[:10, :, :][::-1, :, :])
    assert np.all(buf.read(10, 10) == test_data[10:20, :, :][::-1, :, :])
    assert buf.read(10, 10).size == 0
