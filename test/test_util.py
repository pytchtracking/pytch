import numpy as num
from pytch.util import consecutive, f2cent, cent2f


def test_consecutive():
    arr = num.array([2, 3, 4, 7, 9, 10])
    i = consecutive(arr)
    compare = [num.array([2, 3, 4]), num.array([7]), num.array([9, 10])]
    for ielement, element in enumerate(i):
        assert all(element == compare[ielement])


def test_p2f2p():
    fs = num.random.random(1000) * 1000.0
    standard_frequency = 200
    ps = f2cent(fs, standard_frequency=standard_frequency)
    num.testing.assert_almost_equal(
        fs, cent2f(ps, standard_frequency=standard_frequency)
    )
