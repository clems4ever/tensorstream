import tensorflow as tf

from tensorstream.streamable import Streamable
from tensorstream.finance.signals.zero_crossover_signal import ZeroCrossoverSignal
from tensorstream.finance.moving_average_convergence_divergence import MovingAverageConvergenceDivergence as MACD
from tensorstream.meta import Compose, Select

def MovingAverageConvergenceDivergenceSignal(slow, fast, macd, dtype=tf.float32, shape=()):
  return Compose(
    ZeroCrossoverSignal(dtype=dtype, shape=shape),
    Select(4, dtype=dtype, shape=shape),
    MACD(slow=slow, fast=fast, macd=macd, dtype=dtype, shape=shape)
  )
