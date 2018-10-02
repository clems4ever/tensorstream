import numpy as np
import tensorflow as tf

from tensorstream.tests import TestCase
from tensorstream.finance.volatility import Volatility

class VolatilitySpec(TestCase):
  def setUp(self):
    self.sheets = self.read_ods(
      self.from_test_res('volatility.ods', __file__))

  def test_volatility(self):
    volatility = Volatility(5)
    values = tf.placeholder(tf.float32)
    volatility_ts, _, _ = volatility(values)

    inputs = self.sheets['Sheet1']

    with tf.Session() as sess:
      output = sess.run(volatility_ts, { values: inputs['Prices'] })

    np.testing.assert_almost_equal(output,
      inputs['Volatility'].values, decimal=3)

