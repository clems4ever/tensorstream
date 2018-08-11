import numpy as np
import tensorflow as tf

from tensorstream.streamable import stream_to_tensor, Stream
from tensorstream.operators.relative_strength_index import RelativeStrengthIndex

from tensorstream.operators.tests import TestCase

class RelativeStrengthIndexSpec(TestCase):
  def setUp(self):
    self.input_ts = self.read_csv(
      self.from_test_res('relative_strength_index.csv'))\
        .astype('float32')

  def test_rsi(self):
    rsi = RelativeStrengthIndex(14)
    values = tf.placeholder(tf.float32)

    rsi_ts, _ = stream_to_tensor(rsi(Stream(values)))

    with tf.Session() as sess:
      output = sess.run(rsi_ts, {
        values: self.input_ts['Close'],
      })

    np.testing.assert_almost_equal(output,
      self.input_ts['RSI'].values, decimal=3)
