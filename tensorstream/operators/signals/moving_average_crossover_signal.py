import tensorflow as tf
from tensorstream.operators.meta import Compose
from tensorstream.operators.moving_average_crossover import SimpleMovingAverageCrossover
from tensorstream.operators.signals.zero_crossover_signal import ZeroCrossoverSignal

def SimpleMovingAverageCrossoverSignal(slow, fast, dtype=tf.float32, shape=()):
  return Compose(
    ZeroCrossoverSignal(dtype=dtype, shape=shape),
    SimpleMovingAverageCrossover(slow, fast, dtype=dtype, shape=shape)
  )
