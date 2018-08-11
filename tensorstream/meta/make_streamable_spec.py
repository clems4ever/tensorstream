import numpy as np
import tensorflow as tf
import unittest

from tensorstream.meta import make_streamable

class MakeStreamableSpec(unittest.TestCase):
  def test_make_streamable(self):
    values1 = tf.constant([1, 3, 5, 6, 7, 9])
    values2 = tf.constant([3, 4, 2, 8, 3, 1])

    streamable = make_streamable(lambda x, y: x+y, tf.int32)

    model, _ = streamable(inputs=(values1, values2))
    with tf.Session() as sess:
      data = sess.run(model)

    expected = [4, 7, 7, 14, 10, 10]
    np.testing.assert_equal(data, expected)

  def test_make_streamable_multi_dim(self):
    values1 = tf.constant([[1, 2], [3, 4], [5, 4], [3, 6], [1, 7], [9, 4]])
    values2 = tf.constant([[5, 3], [7, 4], [3, 2], [5, 8], [3, 3], [1, 4]])

    streamable = make_streamable(lambda x, y: x + y, tf.int32, shape=(2,))

    model, _ = streamable(inputs=(values1, values2))
    with tf.Session() as sess:
      data = sess.run(model)

    expected = [[6, 5], [10, 8], [8, 6], [8, 14], [4, 10], [10, 8]]
    np.testing.assert_equal(data, expected)

