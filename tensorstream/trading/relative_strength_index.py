import math
import tensorflow as tf

from tensorstream.streamable import Streamable
from tensorstream.common.lag import Lag

class RelativeStrengthIndex(Streamable):
  def __init__(self, period, dtype=tf.float32, shape=()):
    super().__init__(dtype, shape)
    self.period = period
    self.buffer = Lag(period, dtype=dtype, shape=shape)
    self.initial_state = (
      self.buffer.initial_state,
      tf.constant(0), # iteration
      tf.constant(0.0), # average gain
      tf.constant(0.0) # average loss
    )

  def step(self, value, buffer_state, iteration, last_average_gain, last_average_loss):
    def compute_default_gain_loss():
      current_gain_or_loss = value - buffer_state[0]
      current_gain = tf.maximum(0.0, current_gain_or_loss)
      current_loss = -tf.minimum(0.0, current_gain_or_loss)
      new_average_gain = (last_average_gain * (self.period - 1) + \
        current_gain) / self.period
      new_average_loss = (last_average_loss * (self.period - 1) + \
        current_loss) / self.period
      return new_average_gain, new_average_loss

    def compute_first_gain_loss():
      extended_buffer_state = tf.concat([[value], buffer_state], 0)
      old = extended_buffer_state[1:]
      new = extended_buffer_state[:-1]
      diff = new - old
      new_gains = tf.map_fn(lambda x: tf.cond(x > 0.0,
        lambda: x, lambda: 0.0), diff)
      new_losses = tf.map_fn(lambda x: tf.cond(x < 0.0,
        lambda: x, lambda: 0.0), diff)
      new_average_gain = tf.reduce_mean(new_gains)
      new_average_loss = -tf.reduce_mean(new_losses)
      return new_average_gain, new_average_loss

    def compute_rsi(average_gain, average_loss):
      rs = average_gain / average_loss
      rsi = 100.0 - (100.0 / (1.0 + rs))
      return rsi

    def compute_first_rsi():
      new_average_gain, new_average_loss = compute_first_gain_loss()
      rsi = compute_rsi(new_average_gain, new_average_loss)
      return (new_average_gain, new_average_loss, rsi)

    def compute_default_rsi():
      new_average_gain, new_average_loss = compute_default_gain_loss()
      rsi = compute_rsi(new_average_gain, new_average_loss)
      return (new_average_gain, new_average_loss, rsi)

    def warmup():
      return math.nan, math.nan, math.nan
    
    new_average_gain, new_average_loss, rsi = tf.case({
      tf.equal(iteration, self.period): compute_first_rsi,
      tf.less(iteration, self.period): warmup,
    }, exclusive=True, default=compute_default_rsi)

    _, buffer_state = self.buffer(value, state=buffer_state, streamable=False)

    return rsi, (
      buffer_state,
      iteration + 1,
      new_average_gain,
      new_average_loss
    )
