import tensorflow as tf

from tensorstream.meta import Compose, Join, Sub, Fork
from tensorstream.finance.moving_average import SimpleMovingAverage

def SimpleMovingAverageCrossover(slow, fast, dtype=tf.float32, shape=()):
  return Compose(
    Sub(dtype=dtype, shape=shape),
    Join(SimpleMovingAverage(fast), SimpleMovingAverage(slow)),
    Fork(2, dtype=dtype, shape=shape)
  )
