import unittest

from tensorstream.helpers.flatten import flatten

class FlattenSpec(unittest.TestCase):
  def test_flatten(self):
    x = flatten((1, {'a': (1,'z'), 'b': {'c': 'd'}},))

    assert(x == [1, 1, 'z', 'd'] or x == [1, 'd', 1, 'z'])
