import numpy as np
import tensorflow as tf
from tensorstream.finance.heikin_ashi import HeikinAshi
from tensorstream.tests import TestCase


class HeikinAshiSpec(TestCase):
  def setUp(self):
    self.input_ts = self.read_ods(
      self.from_test_res('heikin_ashi.ods', __file__))["Sheet1"]

  def test_heikin_ashi(self):
    heikinashi_op = HeikinAshi()
    
    close_p = tf.placeholder(tf.float32)
    open_p = tf.placeholder(tf.float32)
    low_p = tf.placeholder(tf.float32)
    high_p = tf.placeholder(tf.float32)

    heikinashi_ts, _, _ = heikinashi_op(inputs=(open_p, high_p, low_p, close_p))
  
    data = { 
              open_p  : self.input_ts["open"],
              high_p  : self.input_ts["high"],
              low_p   : self.input_ts["low"],
              close_p : self.input_ts["close"],
            }

    with tf.Session() as sess:
      output = sess.run(heikinashi_ts, data)

    np.testing.assert_almost_equal(output[0],
      self.input_ts['ha_open'].values, decimal=3)
    np.testing.assert_almost_equal(output[1],
      self.input_ts['ha_high'].values, decimal=3)
    np.testing.assert_almost_equal(output[2],
      self.input_ts['ha_low'].values, decimal=3)
    np.testing.assert_almost_equal(output[3],
      self.input_ts['ha_close'].values, decimal=3)
