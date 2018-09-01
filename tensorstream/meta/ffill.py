import functools
import math
import tensorflow as tf

from tensorstream.streamable import Streamable
from tensorstream.helpers.any_nan import any_nan
from tensorstream.helpers.map_fn import map_fn

class FFill(Streamable):
  """
  Forward fill when nan values is encountered.
  """
  def __init__(self, operator):
    super().__init__(operator.dtype, operator.shape)
    self.operator = operator
    nan_output = map_fn(
      self.dtype,
      [self.shape],
      lambda shape: tf.fill(shape, math.nan)
    )
    self.initial_state = (
      nan_output, # previous output
      operator.initial_state
    )

  def step(self, *inputs_and_states):
    total_len = len(inputs_and_states)
    state_len = len(self.initial_state)
    in_len = total_len - state_len

    inputs = inputs_and_states[:in_len]
    previous_state = inputs_and_states[in_len:]

    next_output, next_state = self.operator(
      inputs=inputs,
      state=previous_state[1],
      streamable=False)

    inputs_any_nan = any_nan(inputs)

    selected_outputs = tf.cond(
      inputs_any_nan,
      lambda: previous_state[0],
      lambda: next_output)

    if self.operator.initial_state == ():
      selected_next_state = ()
    else:
      selected_next_state = tf.cond(
        inputs_any_nan,
        lambda: previous_state[1],
        lambda: next_state)

    return selected_outputs, (selected_outputs, selected_next_state)
