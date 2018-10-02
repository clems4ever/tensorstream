import tensorflow as tf

from tensorstream.common.common import Sub, Fork
from tensorstream.common.set_during import SetDuring
from tensorstream.finance.moving_average import SimpleMovingAverage
from tensorstream.meta.compose import Compose
from tensorstream.meta.join import Join

def SimpleMovingAverageCrossover(slow, fast):
  return Compose(
    Sub(),
    Join(
      Compose(
        # We want 0.0 until index 'slow-1' of the timeseries
        SetDuring(tf.constant(0.0), slow - 1),
        SimpleMovingAverage(fast)
      ),
      SimpleMovingAverage(slow)
    ),
    Fork(2)
  )
