from tensorstream.streamable import Streamable
from tensorstream.finance.moving_average import SimpleMovingAverage
from tensorstream.finance.moving_standard_deviation import MovingStandardDeviation

class BollingerBands(Streamable):
  def __init__(self, n, k):
    super().__init__()
    self.k = k
    self.sma = SimpleMovingAverage(n)
    self.msd = MovingStandardDeviation(n)

  def step(self, value, prev_volatility_state=None, prev_sma_state=None):
    sma_output, next_sma_state, sma_init = self.sma(
      value, state=prev_sma_state, streamable=False)

    volatility_output, next_volatility_state, volatility_init = self.msd(
      value, state=prev_volatility_state, streamable=False)

    middle_band = sma_output
    lower_band = sma_output - volatility_output * self.k
    upper_band = sma_output + volatility_output * self.k
    return (
      lower_band,
      middle_band,
      upper_band
    ), (
      next_volatility_state,
      next_sma_state
    ), (
      volatility_init,
      sma_init
    )
