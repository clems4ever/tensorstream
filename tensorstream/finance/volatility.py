import tensorflow as tf
from tensorstream.meta.compose import Compose
from tensorstream.finance.moving_standard_deviation import MovingStandardDeviation
from tensorstream.finance.returns import LogarithmicReturn

def Volatility(moving_period, return_period=1, dtype=tf.float32, shape=()):
  return Compose(
    MovingStandardDeviation(period=moving_period, dtype=dtype, shape=shape),
    LogarithmicReturn(period=return_period, dtype=dtype, shape=shape)
  )
