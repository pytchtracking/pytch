import numpy as num
import unittest
from pytch.util import consecutive
import time


class UtilTestCase(unittest.TestCase):

    def test_consecutive(self):
        arr = num.array([2,3,4,7,9,10])
        i = consecutive(arr)
        compare = [num.array([2,3,4]), num.array([7]), num.array([9, 10])]
        for ielement, element in enumerate(i):
            self.assertTrue(all(element == compare[ielement]))


if __name__=='__main__':
    unittest.main()
