import unittest
from kmer import vector

class TestVectorization(unittest.TestCase):

    def test_vector(self):
        sequence = "ATCGATCGA"
        k = 3
        expected = {'ATC': 2, 'TCG': 2, 'CGA': 2}
        result = vector(sequence, k)
        self.assertEqual(result, expected)

if __name__ == '__main__':
    unittest.main()
