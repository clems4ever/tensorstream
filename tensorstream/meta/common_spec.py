import numpy as np
import tensorflow as tf
import unittest

from tensorstream.streamable import Stream, stream_to_tensor
from tensorstream.meta import Add, Sub, Mul, Fork
from tensorstream.meta import Identity
from tensorstream.meta import Select, Positive, Negative

class CommonSpec(unittest.TestCase):
  def setUp(self):
    self.values1 = Stream(tf.constant([1, 3, 5, 6, 7, 9]))
    self.values2 = Stream(tf.constant([3, 4, 2, 8, 3, 1]))
    self.values3 = Stream(tf.constant([4, 1, 4, 2, 2, 3]))

    self.values1_m = Stream(tf.constant([
      [1, 2], [3, 4], [5, 6], [3, 2], [5, 7], [4, 9]
    ]))
    self.values2_m = Stream(tf.constant([
      [3, 4], [4, 4], [2, 2], [2, 8], [3, 4], [1, 6]
    ]))

  def test_sub(self):
    sub_op = Sub(tf.int32)
    model, _ = stream_to_tensor(sub_op(self.values1, self.values2))

    with tf.Session() as sess:
      data = sess.run(model)

    expected = [-2, -1, 3, -2, 4, 8]
    np.testing.assert_equal(data, expected)

  def test_sub_multi_dim(self):
    sub_op = Sub(dtype=tf.int32, shape=[2])
    model, _ = stream_to_tensor(sub_op(self.values1_m, self.values2_m))

    with tf.Session() as sess:
      data = sess.run(model)

    expected = [[-2, -2], [-1, 0], [3, 4], [1, -6], [2, 3], [3, 3]]
    np.testing.assert_equal(data, expected)

  def test_add(self):
    add_op = Add(tf.int32)
    model, _ = stream_to_tensor(add_op(self.values1, self.values2))

    with tf.Session() as sess:
      data = sess.run(model)

    expected = [4, 7, 7, 14, 10, 10]
    np.testing.assert_equal(data, expected)

  def test_add_multi_dim(self):
    add_op = Add(tf.int32, shape=[2])
    model, _ = stream_to_tensor(add_op(self.values1_m, self.values2_m))

    with tf.Session() as sess:
      data = sess.run(model)

    expected = [[4, 6], [7, 8], [7, 8], [5, 10], [8, 11], [5, 15]]
    np.testing.assert_equal(data, expected)
    
  def test_mul(self):
    mul_op = Mul(tf.int32)
    model, _ = stream_to_tensor(mul_op(self.values1, self.values2))

    with tf.Session() as sess:
      data = sess.run(model)

    expected = [3, 12, 10, 48, 21, 9]
    np.testing.assert_equal(data, expected)

  def test_mul_multi_dim(self):
    mul_op = Mul(tf.int32, shape=[2])
    model, _ = stream_to_tensor(mul_op(self.values1_m, self.values2_m))

    with tf.Session() as sess:
      data = sess.run(model)

    self.values1_m = tf.constant([
      [1, 2], [3, 4], [5, 6], [3, 2], [5, 7], [4, 9]
    ])
    self.values2_m = tf.constant([
      [3, 4], [4, 4], [2, 2], [2, 8], [3, 4], [1, 6]
    ])
    expected = [[3, 8], [12, 16], [10, 12], [6, 16], [15, 28], [4, 54]]
    np.testing.assert_equal(data, expected)

  def test_fork(self):
    fork_op = Fork(3, tf.int32)
    model, _ = stream_to_tensor(fork_op(self.values1))

    with tf.Session() as sess:
      data = sess.run(model)

    expected = [1, 3, 5, 6, 7, 9]
    np.testing.assert_equal(data[0], expected)
    np.testing.assert_equal(data[1], expected)
    np.testing.assert_equal(data[2], expected)

  def test_identity(self):
    id_op = Identity(tf.int32)
    model, _ = stream_to_tensor(id_op(self.values1))

    with tf.Session() as sess:
      data = sess.run(model)

    expected = [1, 3, 5, 6, 7, 9]
    np.testing.assert_equal(data, expected)

  def test_identity_multi_dim(self):
    id_op = Identity(tf.int32, shape=[2])
    model, _ = stream_to_tensor(id_op(self.values1_m))

    with tf.Session() as sess:
      data = sess.run(model)

    expected = [
      [1, 2], [3, 4], [5, 6], [3, 2], [5, 7], [4, 9]
    ]
    np.testing.assert_equal(data, expected)

  def test_select(self):
    select_op = Select([0, 2], dtype=[tf.int32, tf.int32], shape=[(), ()])
    model, _ = stream_to_tensor(select_op(self.values1, self.values2, self.values3))

    with tf.Session() as sess:
      data = sess.run(model)

    expected1 = [1, 3, 5, 6, 7, 9]
    expected2 = [4, 1, 4, 2, 2, 3]

    self.assertEqual(len(data), 2)
    np.testing.assert_equal(data[0], expected1)
    np.testing.assert_equal(data[1], expected2)

  def test_select_multi_dim(self):
    select_op = Select([0, 2], dtype=[tf.int32, tf.int32], shape=[[2], []])
    model, _ = stream_to_tensor(select_op(self.values1_m, self.values2, self.values3))

    with tf.Session() as sess:
      data = sess.run(model)

    expected1 = [
      [1, 2], [3, 4], [5, 6], [3, 2], [5, 7], [4, 9]
    ]
    expected2 = [4, 1, 4, 2, 2, 3]

    self.assertEqual(len(data), 2)
    np.testing.assert_equal(data[0], expected1)
    np.testing.assert_equal(data[1], expected2)

  def test_positive(self):
    values1 = Stream(tf.constant([1, -3, 5, -6, 7, 9]))

    op = Positive(tf.int32)
    model, _ = stream_to_tensor(op(values1))

    with tf.Session() as sess:
      data = sess.run(model)

    expected1 = [1, 0, 1, 0, 1, 1]
    np.testing.assert_equal(data, expected1)

  def test_positive_multi_dim(self):
    values1 = Stream(tf.constant([[1, 3], [-3, 2], [5, -3], [-6, -3], [7, -2], [1, 9]]))

    op = Positive(tf.int32, shape=[2])
    model, _ = stream_to_tensor(op(values1))

    with tf.Session() as sess:
      data = sess.run(model)

    expected1 = [[1, 1], [0, 1], [1, 0], [0, 0], [1, 0], [1, 1]]
    np.testing.assert_equal(data, expected1)

  def test_negative(self):
    values1 = Stream(tf.constant([1, -3, 5, -6, 7, 9]))

    op = Negative(tf.int32)
    model, _ = stream_to_tensor(op(values1))

    with tf.Session() as sess:
      data = sess.run(model)

    expected1 = [0, 1, 0, 1, 0, 0]
    np.testing.assert_equal(data, expected1)

  def test_negative_multi_dim(self):
    values1 = Stream(tf.constant([[1, 3], [-3, 2], [5, -3], [-6, -3], [7, -2], [1, 9]]))

    op = Negative(tf.int32, shape=[2])
    model, _ = stream_to_tensor(op(values1))

    with tf.Session() as sess:
      data = sess.run(model)

    expected1 = [[0, 0], [1, 0], [0, 1], [1, 1], [0, 1], [0, 0]]
    np.testing.assert_equal(data, expected1)
