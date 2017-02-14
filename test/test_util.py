import numpy as num
import unittest
from pytch.util import consecutive, f2pitch, pitch2f
import time


class UtilTestCase(unittest.TestCase):

    def test_consecutive(self):
        arr = num.array([2,3,4,7,9,10])
        i = consecutive(arr)
        compare = [num.array([2,3,4]), num.array([7]), num.array([9, 10])]
        for ielement, element in enumerate(i):
            self.assertTrue(all(element == compare[ielement]))

    def test_p2f2p(self):
        fs = num.random.random(1000)*1000.
        ps = f2pitch(fs)
        num.testing.assert_almost_equal(fs, pitch2f(ps))
        

if __name__=='__main__':
    unittest.main()
