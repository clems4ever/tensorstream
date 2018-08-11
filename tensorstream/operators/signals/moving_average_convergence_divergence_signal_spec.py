import numpy as np
import pandas as pd
import tensorflow as tf

from tensorstream.streamable import stream_to_tensor, Stream
from tensorstream.operators.signals import MovingAverageConvergenceDivergenceSignal as MACDSignal
from tensorstream.operators.tests import TestCase

class MACDSignalSpec(TestCase):
  def setUp(self):
    self.sheets = self.read_ods(
      self.from_test_res('macd_signals.ods', __file__))
  
  def test_single_dim(self):
    s = self.sheets['single_dim'].fillna(0.0)
    prices_ts = s['Close']

    expected_signals = s['Trade Signals']
 
    signal = MACDSignal(26, 12, 9)
    prices = tf.placeholder(tf.float32)
    signals_ts, _ = stream_to_tensor(signal(Stream(prices)))

    with tf.Session() as sess:
      output_ts = sess.run(signals_ts, {
        prices: prices_ts
      })

    np.testing.assert_almost_equal(output_ts, 
      expected_signals.values, decimal=3)

  def test_multi_dim(self):
    s = self.sheets['multi_dim'].fillna(0.0)
    prices_ts = s[['Close 0', 'Close 1']]
    expected_signals = s[['Trade Signals 0', 'Trade Signals 1']]
 
    signal = MACDSignal(26, 12, 9, shape=(2,))
    prices = tf.placeholder(tf.float32, shape=[None, 2])
    signals_ts, _ = stream_to_tensor(signal(Stream(prices)))

    with tf.Session() as sess:
      output_ts = sess.run(signals_ts, {
        prices: prices_ts
      })

    np.testing.assert_almost_equal(output_ts, 
      expected_signals.values, decimal=3)
