import numpy as np
import pandas as pd
import tensorflow as tf

from tensorstream.streamable import stream_to_tensor, Stream
from tensorstream.tests import TestCase
from tensorstream.trading.evaluators.risk_ratio_evaluator import RiskRatioEvaluator

class RiskRatioEvaluatorSpec(TestCase):
  def setUp(self):
    self.sheets = self.read_ods(
      self.from_test_res('risk_ratio_evaluator.ods', __file__))

  def test_risk_ratio_evaluator(self):
    evaluator = RiskRatioEvaluator(stop_factor=2.0, limit_factor=4.0)
    close_prices = tf.placeholder(tf.float32)
    low_prices = tf.placeholder(tf.float32)
    high_prices = tf.placeholder(tf.float32)
    signals = tf.placeholder(tf.float32)

    evaluator_ts, _ = stream_to_tensor(
      evaluator(
        Stream(low_prices),
        Stream(high_prices),
        Stream(close_prices),
        Stream(signals)
      )
    )

    input_ts = self.sheets['Sheet1']
    
    with tf.Session() as sess:
      output = sess.run(evaluator_ts, {
        close_prices: input_ts['Close'],
        low_prices: input_ts['Low'],
        high_prices: input_ts['High'],
        signals: input_ts['Signal'],
      })

    np.testing.assert_almost_equal(output[0],
      input_ts['successful'].values, decimal=3)
    np.testing.assert_almost_equal(output[1],
      input_ts['total'].values, decimal=3)
    np.testing.assert_almost_equal(output[3],
      input_ts['signal'].values, decimal=3)
