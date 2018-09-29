from tensorstream.streamable import Streamable
from tensorstream.finance.moving_average import SimpleMovingAverage
from tensorstream.finance.moving_standard_deviation import MovingStandardDeviation

class BollingerBands(Streamable):
  def __init__(self, n, k):
    super().__init__()
    self.k = k
    self.sma = SimpleMovingAverage(n)
    self.msd = MovingStandardDeviation(n)

  def properties(self, value):
    msd_ph, msd_init_state = self.msd.properties(value)
    sma_ph, sma_init_state = self.sma.properties(value)
    return (value, value, value), ( 
      msd_init_state,
      sma_init_state
    )

  def step(self, value, volatility_state, sma_state):
    sma_output, sma_state = self.sma(
      value, state=sma_state, streamable=False)

    volatility_output, volatility_state = self.msd(
      value, state=volatility_state, streamable=False)

    middle_band = sma_output
    lower_band = sma_output - volatility_output * self.k
    upper_band = sma_output + volatility_output * self.k
    return (
      lower_band,
      middle_band,
      upper_band
    ), (
      volatility_state,
      sma_state
    )
