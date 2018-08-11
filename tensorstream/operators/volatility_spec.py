import numpy as np
import tensorflow as tf

from tensorstream.streamable import Stream, stream_to_tensor
from tensorstream.operators.volatility import Volatility
from tensorstream.operators.tests import TestCase

class VolatilitySpec(TestCase):
  # Data from http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:standard_deviation_volatility
  def setUp(self):
    self.input_ts = self.read_csv(
      self.from_test_res('volatility.csv'))\
        .fillna(0.0).astype('float32')

  def test_global_min(self):
    volatility = Volatility(10)
    values = tf.placeholder(tf.float32)
    stream = Stream(values)
    volatility_ts, _ = stream_to_tensor(volatility(stream))

    with tf.Session() as sess:
      output = sess.run(volatility_ts, { values: self.input_ts['Value'] })

    np.testing.assert_almost_equal(output,
      self.input_ts['Volatility'].values, decimal=3)
