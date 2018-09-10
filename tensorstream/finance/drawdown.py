import tensorflow as tf

from tensorstream.streamable import Streamable
from tensorstream.common import filter_with_mask, map_consecutive_fn

ERROR = 0.0001

class Drawdown(Streamable):
  def __init__(self):
    super().__init__((0.0, 0))

  def step(self, value, last_peak, last_periods_under_water):
    new_peak = tf.maximum(value, last_peak)
    new_periods_under_water = tf.cond(tf.abs(new_peak - last_peak) > ERROR,
      lambda: 0,
      lambda: last_periods_under_water + 1)
    drawdown = tf.cond(tf.abs(last_peak) < ERROR,
      lambda: 0.0,
      lambda: tf.minimum(0.0, value / last_peak - 1.0))
    return (drawdown, new_periods_under_water), (new_peak, new_periods_under_water)
    

def details(drawdown, drawdown_days):
  def compute_avg(begin_end):
    begin = begin_end[0]
    end = begin_end[1]
    max_drawdown = tf.reduce_min(drawdown[begin:end+1])
    drawdown_days_p = tf.reduce_max(drawdown_days[begin:end+1])
    return max_drawdown, drawdown_days_p

  d = tf.pad(drawdown_days, [[1, 1]])
  idx = tf.range(0, tf.size(d) - 1)

  period_beginnings_mask = tf.map_fn(
    lambda x: tf.equal(x, 1),
    d,
    dtype=tf.bool)

  period_endings_mask = map_consecutive_fn(
    lambda x, y: tf.logical_and(tf.greater(y, 0), tf.equal(x, 0)),
    d)

  period_beginnings = filter_with_mask(period_beginnings_mask, idx)
  period_endings = filter_with_mask(period_endings_mask, idx)

  bounds = tf.transpose(
    tf.stack([period_beginnings, period_endings])) - 1

  return tf.map_fn(
    compute_avg,
    bounds,
    dtype=(tf.float32, tf.int32))
  
def avg(drawdown, drawdown_days):
    max_drawdown_p, drawdown_days_p = details(drawdown, drawdown_days)
    mean_max_drawdown = tf.reduce_mean(max_drawdown_p)
    mean_drawdown_days = tf.reduce_mean(tf.to_float(drawdown_days_p))
    return mean_max_drawdown, mean_drawdown_days
  
