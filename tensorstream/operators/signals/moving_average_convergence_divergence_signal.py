import tensorflow as tf

from tensorstream.operators import Streamable
from tensorstream.operators.signals.zero_crossover_signal import ZeroCrossoverSignal
from tensorstream.operators.moving_average_convergence_divergence import MovingAverageConvergenceDivergence as MACD
from tensorstream.operators.meta import Compose, Select

def MovingAverageConvergenceDivergenceSignal(slow, fast, macd, dtype=tf.float32, shape=()):
  return Compose(
    ZeroCrossoverSignal(dtype=dtype, shape=shape),
    Select(4, dtype=dtype, shape=shape),
    MACD(slow=slow, fast=fast, macd=macd, dtype=dtype, shape=shape)
  )
