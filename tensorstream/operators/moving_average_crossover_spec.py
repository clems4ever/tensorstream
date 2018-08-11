import numpy as np
import pandas as pd
import tensorflow as tf

from tensorstream.operators.tests import TestCase
from tensorstream.streamable import Stream, stream_to_tensor
from tensorstream.operators.moving_average_crossover import SimpleMovingAverageCrossover

class MovingAverageCrossoverSpec(TestCase):
  def setUp(self):
    self.sheets = self.read_ods(
      self.from_test_res('moving_average_crossover.ods'))

  def test_simple_moving_average_crossover(self):
    ma_crossover = SimpleMovingAverageCrossover(10, 4)
    close_prices = tf.placeholder(tf.float32)
    sheet = self.sheets['Sheet1']

    ma_crossover_ts, _ = stream_to_tensor(ma_crossover(Stream(close_prices)))
    with tf.Session() as sess:
      output = sess.run(ma_crossover_ts, {
        close_prices: sheet['Close']
      })

    np.testing.assert_almost_equal(output,
      sheet['MA Crossover'], decimal=3)
