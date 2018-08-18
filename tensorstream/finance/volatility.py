import tensorflow as tf
from tensorstream.meta.compose import Compose
from tensorstream.finance.moving_standard_deviation import MovingStandardDeviation
from tensorstream.finance.returns import Return

def Volatility(period, dtype=tf.float32, shape=()):
  return Compose(
    MovingStandardDeviation(period=period, dtype=dtype, shape=shape),
    Return(period=1, dtype=dtype, shape=shape)
  )
