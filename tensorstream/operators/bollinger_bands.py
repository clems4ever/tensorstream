import tensorflow as tf

from tensorstream.operators import Streamable
from tensorstream.operators.moving_average import SimpleMovingAverage
from tensorstream.operators.volatility import Volatility

class BollingerBands(Streamable):
  def __init__(self, n, k, dtype=tf.float32, shape=()):
    super().__init__((dtype,) * 3, (shape,) * 3)
    self.k = k
    self.sma = SimpleMovingAverage(n, dtype=dtype, shape=shape)
    self.volatility = Volatility(n, dtype=dtype, shape=shape)
    self.initial_state = ( 
      self.volatility.initial_state,
      self.sma.initial_state
    )

  def step(self, value, volatility_state, sma_state):
    sma_output, sma_state = self.sma(value, state=sma_state)
    volatility_output, volatility_state = self.volatility(value, state=volatility_state)

    middle_band = sma_output
    lower_band = sma_output - volatility_output * self.k
    upper_band = sma_output + volatility_output * self.k
    return (lower_band, middle_band, upper_band), (volatility_state, sma_state)
