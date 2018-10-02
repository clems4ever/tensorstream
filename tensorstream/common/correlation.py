import tensorflow as tf

from tensorstream.common import roll
from tensorstream.streamable import Streamable

class Correlation(Streamable):
  def __init__(self, period):
    super().__init__()
    self.period = period

  def step(self, value1, value2, iteration=None,
    last_values1=None, last_values2=None):

    if iteration is None:
      iteration = tf.constant(0)
    if last_values1 is None:
      shape = self.concat([self.period], tf.shape(value1))
      last_values1 = tf.zeros(shape, value1.dtype)
    if last_values2 is None:
      shape = self.concat([self.period], tf.shape(value2))
      last_values2 = tf.zeros(shape, value2.dtype)

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
      lambda: tf.zeros(tf.shape(value1), dtype=value1.dtype),
      lambda: compute_correlation(new_values1, new_values2))

    return correlation, (
      iteration + 1,
      new_values1,
      new_values2
    ), (
      iteration,
      last_values1,
      last_values2
    )

class CrossCorrelation(Streamable):
  def __init__(self, period, lag):
    super().__init__()
    self.correlation = Correlation(period)
    self.lag = lag

  def step(self, value1, value2, iteration=None,
    prev_correlation_state=None, last_lag_buffer=None):
    """
    Value1 is leading
    """
    if iteration is None:
      iteration = tf.constant(0)
    if last_lag_buffer is None:
      shape = self.concat([self.lag], tf.shape(value1))
      last_lag_buffer = tf.zeros(shape, dtype=value1.dtype)

    lagged_value2 = last_lag_buffer[self.lag - 1]

    correlation, next_correlation_state, correlation_init_state = self.correlation(
      inputs=(value1, lagged_value2),
      state=prev_correlation_state,
      streamable=False
    )

    if prev_correlation_state is None:
      prev_correlation_state = correlation_init_state

    correlation, new_correlation_state = tf.cond(iteration < self.lag,
      lambda: (tf.zeros(tf.shape(value1), dtype=value1.dtype), prev_correlation_state),
      lambda: (correlation, next_correlation_state)
    )
    new_lag_buffer = roll(value2, last_lag_buffer)
    return correlation, (
      iteration + 1,
      new_correlation_state,
      new_lag_buffer
    ), (iteration, correlation_init_state, last_lag_buffer)

class AutoCorrelation(Streamable):
  def __init__(self, period, lag):
    super().__init__()
    self.cross_correlation = CrossCorrelation(period, lag)

  def step(self, value, prev_corr_state=None):
    corr, corr_next_state, corr_init_state = self.cross_correlation(
      inputs=(value, value),
      state=prev_corr_state,
      streamable=False
    )
    return corr, (corr_next_state,), (corr_init_state,)
