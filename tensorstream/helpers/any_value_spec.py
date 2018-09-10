import math
import tensorflow as tf
import unittest

from tensorstream.helpers.any_value import any_value

def in_tf(x):
  with tf.Session() as sess:
    return sess.run(x)

class AnyValueSpec(unittest.TestCase):
  def test_any_value_scalar(self):
    x = any_value(tf.constant(4.0), 0.0)
    self.assertEqual(in_tf(x), False)

    y = any_value(tf.constant(0.0), 0.0)
    self.assertEqual(in_tf(y), True)

  def test_any_value_tensor(self):
    x = any_value(tf.constant([4.0, 3.0, 2.0]), 0.0)
    self.assertEqual(in_tf(x), False)

    y = any_value(tf.constant([0.0, 3.0, 2.0]), 0.0)
    self.assertEqual(in_tf(y), True)

    z = any_value(tf.constant([0.0, 0.0, 0.0]), 0.0)
    self.assertEqual(in_tf(z), True)

  def test_any_value_complex_type(self):
    x = any_value({
      'a': tf.constant([3.0, 2.0]),
      'b': [tf.constant(3.2), tf.constant([2.1, 2.3, 4.3])],
      'c': {
        'z': tf.constant([5.2, 5.2]),
        'y': tf.constant([3.4, 5.2])
      }
    }, 0.0)

    self.assertEqual(in_tf(x), False)

    y = any_value({
      'a': tf.constant([3.0, 2.0]),
      'b': [tf.constant(3.2), tf.constant([2.1, 2.3, 0.0])],
      'c': {
        'z': tf.constant([5.2, 5.2]),
        'y': tf.constant([3.4, 5.2])
      }
    }, 0.0)
    self.assertEqual(in_tf(y), True)

    z = any_value({
      'a': tf.constant([0.0, 0.0]),
      'b': [tf.constant(0.0), tf.constant([0.0, 0.0, 0.0])],
      'c': {
        'z': tf.constant([0.0, 0.0]),
        'y': tf.constant([0.0, 0.0])
      }
    }, 0.0)
    self.assertEqual(in_tf(z), True)
