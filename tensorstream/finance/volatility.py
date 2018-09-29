import tensorflow as tf
from tensorstream.common.set_during import SetDuring
from tensorstream.meta.compose import Compose
from tensorstream.finance.moving_standard_deviation import MovingStandardDeviation
from tensorstream.finance.returns import LogarithmicReturn

def Volatility(moving_period, return_period=1):
  return Compose(
    SetDuring(tf.constant(0.0), max(moving_period, return_period)),
    MovingStandardDeviation(period=moving_period),
    LogarithmicReturn(period=return_period)
  )
