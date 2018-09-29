import tensorflow as tf
from tensorstream.common.set_during import SetDuring
from tensorstream.finance.moving_average_crossover import SimpleMovingAverageCrossover
from tensorstream.finance.signals.zero_crossover_signal import ZeroCrossoverSignal
from tensorstream.meta.compose import Compose

def SimpleMovingAverageCrossoverSignal(slow, fast):
  return Compose(
    SetDuring(tf.constant(0), slow),
    ZeroCrossoverSignal(),
    SimpleMovingAverageCrossover(slow, fast)
  )
