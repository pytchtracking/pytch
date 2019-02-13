import numpy as num
import unittest
from pytch.util import consecutive, f2cent, cent2f
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
        standard_frequency = 200
        ps = f2cent(fs, standard_frequency=standard_frequency)
        num.testing.assert_almost_equal(fs, cent2f(ps, standard_frequency=standard_frequency))
        

if __name__=='__main__':
    unittest.main()
