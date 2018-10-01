import numpy as np
import tensorflow as tf

from tensorstream.streamable import Streamable
from tensorstream.meta.factory import Factory
from tensorstream.tests import TestCase

class MultiplyBy(Streamable):
  def __init__(self, nb):
    super().__init__()
    self.nb = nb

  def step(self, x, prev_x=None):
    if prev_x is None:
      prev_x = tf.constant(0.0)
    return prev_x * self.nb, x, prev_x

class FactorySpec(TestCase):
  def setUp(self):
    self.sheets = self.read_ods(
      self.from_test_res('factory.ods', __file__))

  def test_factory_simple(self):
    sheet = self.sheets['Sheet1']
    factory = Factory(MultiplyBy, ([3], [5], [10]))
    x = tf.placeholder(tf.float32)
    factory_ts, _, _ = factory((x, x, x))
    
    with tf.Session() as sess:
      output = sess.run(factory_ts, { x: sheet['x'] })

    np.testing.assert_almost_equal(output[0],
      sheet['Mul 3'].values, decimal=3)
    np.testing.assert_almost_equal(output[1],
      sheet['Mul 5'].values, decimal=3)
    np.testing.assert_almost_equal(output[2],
      sheet['Mul 10'].values, decimal=3)
