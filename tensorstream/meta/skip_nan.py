import tensorflow as tf
import functools

from tensorstream.streamable import Streamable

class SkipNan(Streamable):
  """
  This operator executes one step of the given operator when no input is nan
  and return the last value otherwise. It also returns if execution has been
  skipped in a boolean.
  """
  def __init__(self, operator):
    super().__init__((operator.dtype, tf.bool), (operator.shape, ()))
    self.operator = operator
    self.initial_state = operator.initial_state

  def step(self, *inputs_and_states):
    def detect_nan(acc, x):
      return tf.logical_or(tf.reduce_any(tf.is_nan(x)), acc)

    total_len = len(inputs_and_states)
    state_len = len(self.initial_state)
    in_len = total_len - state_len

    inputs = inputs_and_states[:in_len]
    previous_state = inputs_and_states[in_len:]
    
    has_nan = functools.reduce(detect_nan, inputs, tf.constant(False))
    zeros = tf.zeros(self.shape[0], dtype=self.dtype[0])
    
    output, next_state = self.operator(*inputs, state=previous_state)

    selected_outputs = tf.cond(has_nan, lambda: zeros, lambda: output)
    selected_next_state = tf.cond(has_nan, lambda: previous_state, lambda: next_state)
    return (selected_outputs, has_nan), selected_next_state
