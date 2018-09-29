import tensorflow as tf
from tensorstream.meta.compose import Compose
from tensorstream.common.common import Select
from tensorstream.common.set_during import SetDuring
from tensorstream.finance.signals.zero_crossover_signal import ZeroCrossoverSignal
from tensorstream.finance.moving_average_convergence_divergence import MovingAverageConvergenceDivergence as MACD

def MovingAverageConvergenceDivergenceSignal(slow, fast, macd):
  return Compose(
    SetDuring(tf.constant(0), slow + macd - 2),
    ZeroCrossoverSignal(),
    Select(4, dtype=tf.float32, shape=()),
    MACD(slow=slow, fast=fast, macd=macd)
  )
