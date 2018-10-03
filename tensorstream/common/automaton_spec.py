import numpy as np
import tensorflow as tf
import unittest

from tensorstream.common.automaton import Automaton

sx = -1
s0 = 0
s1 = 1
s2 = 2

automaton = Automaton(transitions=[
  [s0, s1, lambda x, y: tf.less(x, y), lambda x, y: tf.stack((s0, s1))],
  [s0, s2, lambda x, y: tf.greater(x, y), lambda x, y: tf.stack((s0, s2))],
  [s1, s2, lambda x, y: tf.equal(x, 4), lambda x, y: tf.stack((s1, s2))],
  [s2, s0, lambda x, y: tf.equal(y, 4), lambda x, y: tf.stack((s2, s0))]
], init_state=s0, default_action=lambda x, y: tf.stack((-1, -1)))

class AutomatonSpec(unittest.TestCase):
  def _test_transition(self, from_s, values, to_s):
    x = automaton(
      inputs=values, state=from_s, streamable=False
    )
    with tf.Session() as sess:
      y = sess.run(x)
    np.testing.assert_array_equal(y[0], [from_s, to_s])
    self.assertEqual(y[1], to_s)

  def _test_no_transition(self, from_s, values):
    x = automaton(
      inputs=values, state=from_s, streamable=False
    )
    with tf.Session() as sess:
      y = sess.run(x)
    np.testing.assert_array_equal(y[0], [sx, sx])
    self.assertEqual(y[1], from_s)
  

  def test_s0_s1(self):
    self._test_transition(s0, (4, 11), s1)

  def test_s0_s2(self):
    self._test_transition(s0, (12, 11), s2)

  def test_s0_s0(self):
    self._test_no_transition(s2, (5, 5))

  def test_s1_s2(self):
    self._test_transition(s1, (4, 11), s2)

  def test_s1_s1(self):
    self._test_no_transition(s2, (8, 3))

  def test_s2_s0(self):
    self._test_transition(s2, (10, 4), s0)

  def test_s2_s2(self):
    self._test_no_transition(s2, (8, 3))

  def test_stream(self):
    x = tf.constant([3, 5, 4, 10, 7, 8])
    y = tf.constant([4, 7, 9, 12, 4, 5])
    a = automaton(
      inputs=(x, y), state=s0
    )
    with tf.Session() as sess:
      b = sess.run(a)

    np.testing.assert_array_equal(b[0], [
      [s0, s1],
      [sx, sx],
      [s1, s2],
      [sx, sx],
      [s2, s0],
      [s0, s2],
    ])
    self.assertEqual(b[1], 2)
