import unittest

from tensorstream.helpers.map_fn import map_fn

def times_2(x):
  return x * 2

def times_2_2(x, y):
  return (x * 2, y * 2)

class MapFnSpec(unittest.TestCase):
  def test_map_fn_single_value(self):
    x = 4
    self.assertEqual(map_fn(x, [x], times_2), 8)

  def test_map_fn_list(self):
    x = [2, 3, 4, 5]
    self.assertEqual(map_fn(x, [x], times_2), [4, 6, 8, 10])

  def test_map_fn_dict(self):
    x = { 'a': 4, 'b': 10 }
    self.assertEqual(map_fn(x, [x], times_2), { 'a': 8, 'b': 20 })

  def test_map_fn_complex_type(self):
    x = {
      'a': [{ 'x': 4, 'y': 3 }, 3],
      'b': { 'z': [1, 3] }
    }
    expected = {
      'a': [{ 'x': 8, 'y':6 }, 6],
      'b': { 'z': [2, 6] }
    }
    self.assertEqual(map_fn(x, [x], times_2), expected)

  def test_map_fn_dict_of_dict(self):
    x = { 'b': { 'z': 1, 'y': 2 } }
    exp = { 'b': { 'z': 2, 'y': 4 } }
    self.assertEqual(map_fn(x, [x], times_2), exp)

  def test_map_fn_multi(self):
    x = [3, [4, 8], { 'a': 5, 'b': 6 }]
    y = [6, [2, 3], { 'a': 2, 'b': 4 }]
    exp = [(6, 12), [(8, 4), (16, 6)], { 'a': (10, 4), 'b': (12, 8) }]
    self.assertEqual(map_fn(x, [x, y], times_2_2), exp)
