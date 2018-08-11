import numpy as np
import tensorflow as tf

from tensorstream.meta.factory import Factory
from tensorstream.trading.moving_average import SimpleMovingAverage
from tensorstream.tests import TestCase

class FactorySpec(TestCase):
  def setUp(self):
    self.input_ts = self.read_csv(
      self.from_test_res('factory.csv', __file__)).astype('float32')

  def test_sma_3_5_10_in_factory_list(self):
    factory = Factory(SimpleMovingAverage, ([3], [5], [10]))
    prices = tf.placeholder(tf.float32)
    factory_ts, _ = factory(inputs=(prices, prices, prices))
    
    with tf.Session() as sess:
      output = sess.run(factory_ts, 
        { prices: self.input_ts['Close'] })

    np.testing.assert_almost_equal(output[0],
      self.input_ts['SMA3'].values, decimal=3)
    np.testing.assert_almost_equal(output[1],
      self.input_ts['SMA5'].values, decimal=3)
    np.testing.assert_almost_equal(output[2],
      self.input_ts['SMA10'].values, decimal=3)
