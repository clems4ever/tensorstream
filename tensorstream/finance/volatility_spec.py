import numpy as np
import tensorflow as tf

from tensorstream.tests import TestCase
from tensorstream.finance.volatility import Volatility

class VolatilitySpec(TestCase):
  def setUp(self):
    self.sheets = self.read_ods(
      self.from_test_res('volatility.ods', __file__))

  def test_volatility_single_dim(self):
    volatility = Volatility(5)
    values = tf.placeholder(tf.float32)
    volatility_ts, _ = volatility(values)

    inputs = self.sheets['single_dim']

    with tf.Session() as sess:
      output = sess.run(volatility_ts, { values: inputs['Prices'] })

    np.testing.assert_almost_equal(output,
      inputs['Volatility'].values, decimal=3)

  def test_volatility_multi_dim(self):
    volatility = Volatility(5, shape=(2,))
    values = tf.placeholder(tf.float32, shape=(None, 2))
    volatility_ts, _ = volatility(values)

    inputs = self.sheets['multi_dim']
    data = inputs[['Prices 0', 'Prices 1']]
    expected = inputs[['Volatility 0', 'Volatility 1']]

    with tf.Session() as sess:
      output = sess.run(volatility_ts, { values: data })

    np.testing.assert_almost_equal(output,
      expected.values, decimal=3)
