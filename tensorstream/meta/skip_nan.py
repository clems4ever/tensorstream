import functools
import math
import tensorflow as tf

from tensorstream.streamable import Streamable

class SkipNan(Streamable):
  """
  This operator executes one step of the given operator when no input is nan
  and return the last value otherwise. It also returns if execution has been
  skipped in a boolean.
  """
  def __init__(self, operator, skipped_value=math.nan):
    super().__init__((operator.dtype, tf.bool), (operator.shape, ()))
    self.operator = operator
    self.initial_state = operator.initial_state
    self.skipped_value = skipped_value

  def step(self, *inputs_and_states):
    def detect_nan(acc, input_):
      return tf.logical_or(
        tf.reduce_any(
          tf.is_nan(input_)
        ),
        acc
      )

    total_len = len(inputs_and_states)
    state_len = len(self.initial_state)
    in_len = total_len - state_len

    inputs = inputs_and_states[:in_len]
    previous_state = inputs_and_states[in_len:]
    
    has_nan = functools.reduce(detect_nan, inputs, tf.constant(False))
    nan = tf.fill(self.shape[0], self.skipped_value)
    
    output, next_state = self.operator(inputs, previous_state, streamable=False)

    selected_outputs = tf.cond(has_nan, lambda: nan, lambda: output)
    selected_next_state = tf.cond(has_nan, lambda: previous_state, lambda: next_state)
    return (selected_outputs, has_nan), selected_next_state
