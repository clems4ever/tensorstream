import numpy as np
import tensorflow as tf

from tensorstream.streamable import stream_to_tensor, Stream
from tensorstream.operators.skewness import Skewness
from tensorstream.operators.tests import TestCase

class SkewnessSpec(TestCase):
  def setUp(self):
    self.sheets = self.read_ods(
      self.from_test_res('skewness.ods'))

  def test_single_dim(self):
    s = self.sheets['single_dim'].replace(r'\s*', np.nan, regex=True)
    s10 = Skewness(10)
    prices = tf.placeholder(tf.float32)
    s10_ts, _ = stream_to_tensor(s10(Stream(prices)))
    
    with tf.Session() as sess:
      output = sess.run(s10_ts, {
        prices: s['Return'],
      })

    np.testing.assert_almost_equal(output,
      s['Skewness 10D'].values, decimal=3)

  def test_multi_dim(self):
    s = self.sheets['multi_dim'].replace(r'\s*', np.nan, regex=True)
    s10 = Skewness(10, dtype=tf.float32, shape=(2,))
    prices = tf.placeholder(tf.float32, shape=[None, 2])
    s10_ts, _ = stream_to_tensor(s10(Stream(prices)))

    prices_ts = s[['Return 0', 'Return 1']]
    expected_outputs_ts = s[['Skewness 10D 0', 'Skewness 10D 1']]
    
    with tf.Session() as sess:
      output = sess.run(s10_ts, {
        prices: prices_ts
      })

    np.testing.assert_almost_equal(output,
      expected_outputs_ts.values, decimal=3)

