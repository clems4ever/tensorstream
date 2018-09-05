import math
import tensorflow as tf
import unittest

from tensorstream.helpers.any_nan import any_nan

def in_tf(x):
  with tf.Session() as sess:
    return sess.run(x)

class AnyNanSpec(unittest.TestCase):
  def test_any_nan_scalar(self):
    x = any_nan(tf.constant(4.0))
    self.assertEqual(in_tf(x), False)

    y = any_nan(tf.constant(math.nan))
    self.assertEqual(in_tf(y), True)

  def test_any_nan_tensor(self):
    x = any_nan(tf.constant([4.0, 3.0, 2.0]))
    self.assertEqual(in_tf(x), False)

    y = any_nan(tf.constant([math.nan, 3.0, 2.0]))
    self.assertEqual(in_tf(y), True)

    z = any_nan(tf.constant([math.nan, math.nan, math.nan]))
    self.assertEqual(in_tf(z), True)

  def test_any_nan_complex_type(self):
    x = any_nan({
      'a': tf.constant([3.0, 2.0]),
      'b': [tf.constant(3.2), tf.constant([2.1, 2.3, 4.3])],
      'c': {
        'z': tf.constant([5.2, 5.2]),
        'y': tf.constant([3.4, 5.2])
      }
    })

    self.assertEqual(in_tf(x), False)

    y = any_nan({
      'a': tf.constant([3.0, 2.0]),
      'b': [tf.constant(3.2), tf.constant([2.1, 2.3, math.nan])],
      'c': {
        'z': tf.constant([5.2, 5.2]),
        'y': tf.constant([3.4, 5.2])
      }
    })
    self.assertEqual(in_tf(y), True)

    z = any_nan({
      'a': tf.constant([math.nan, math.nan]),
      'b': [tf.constant(math.nan), tf.constant([math.nan, math.nan, math.nan])],
      'c': {
        'z': tf.constant([math.nan, math.nan]),
        'y': tf.constant([math.nan, math.nan])
      }
    })
    self.assertEqual(in_tf(z), True)
