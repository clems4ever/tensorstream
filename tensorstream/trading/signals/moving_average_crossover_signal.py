import tensorflow as tf
from tensorstream.meta import Compose
from tensorstream.trading.moving_average_crossover import SimpleMovingAverageCrossover
from tensorstream.trading.signals.zero_crossover_signal import ZeroCrossoverSignal

def SimpleMovingAverageCrossoverSignal(slow, fast, dtype=tf.float32, shape=()):
  return Compose(
    ZeroCrossoverSignal(dtype=dtype, shape=shape),
    SimpleMovingAverageCrossover(slow, fast, dtype=dtype, shape=shape)
  )
