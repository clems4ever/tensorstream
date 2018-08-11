import math
import tensorflow as tf

from tensorstream.common import roll
from tensorstream.streamable import Streamable

class Correlation(Streamable):
  def __init__(self, period, dtype=tf.float32, shape=()):
    super().__init__(dtype, shape)
    self.period = period
    self.initial_state = (
      tf.constant(0),
      tf.fill((period,) + shape, math.nan),
      tf.fill((period,) + shape, math.nan)
    )

  def step(self, value1, value2, iteration, last_values1, last_values2):
    def compute_correlation(values1, values2):
      mean_values1, var_values1 = tf.nn.moments(values1, axes=0)
      mean_values2, var_values2 = tf.nn.moments(values2, axes=0)
      stddev_values1 = tf.sqrt(var_values1)
      stddev_values2 = tf.sqrt(var_values2)
      covariance = tf.reduce_mean(
        (values1 - mean_values1) * (values2 - mean_values2), axis=0)
      return covariance / (stddev_values1 * stddev_values2)
    
    new_values1 = roll(value1, last_values1)
    new_values2 = roll(value2, last_values2)

    correlation = tf.cond(
      iteration < self.period,
      lambda: tf.fill(self.shape, math.nan),
      lambda: compute_correlation(new_values1, new_values2))
    return correlation, (iteration + 1, new_values1, new_values2)

class CrossCorrelation(Streamable):
  def __init__(self, period, lag, dtype=tf.float32, shape=()):
    super().__init__(dtype, shape)
    self.correlation = Correlation(period, dtype=dtype, shape=shape)
    self.lag = lag
    self.initial_state = (
      tf.constant(0),
      self.correlation.initial_state,
      tf.fill((lag,) + self.shape, math.nan),
    )

  def step(self, value1, value2, iteration, correlation_state, last_lag_buffer):
    """
    Value1 is leading
    """
    lagged_value2 = last_lag_buffer[self.lag - 1]
    correlation, new_correlation_state = tf.cond(iteration < self.lag,
      lambda: (tf.fill(self.shape, math.nan), correlation_state),
      lambda: self.correlation(
        inputs=(value1, lagged_value2),
        state=correlation_state,
        streamable=False
      )
    )
    new_lag_buffer = roll(value2, last_lag_buffer)
    return correlation, (iteration + 1, new_correlation_state, new_lag_buffer)

class AutoCorrelation(Streamable):
  def __init__(self, period, lag, dtype=tf.float32, shape=()):
    super().__init__(dtype, shape)
    self.cross_correlation = CrossCorrelation(period, lag, dtype=dtype, shape=shape)
    self.initial_state = self.cross_correlation.initial_state

  def step(self, value, *cross_correlation_state):
    return self.cross_correlation(
      inputs=(value, value),
      state=cross_correlation_state,
      streamable=False
    )
