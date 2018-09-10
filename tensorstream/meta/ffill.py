import tensorflow as tf

from tensorstream.helpers.any_value import any_value
from tensorstream.helpers.map_fn import map_fn
from tensorstream.streamable import MetaStreamable

class FFill(MetaStreamable):
  """
  Forward fill when nan values is encountered.
  """
  def __init__(self, operator):
    super().__init__()
    self.operator = operator

  def initial_state(self, inputs):
    outputs, _ = self.operator(inputs, streamable=False)
    zeros = map_fn(
      outputs,
      [outputs],
      lambda o: tf.zeros(tf.shape(o), dtype=tf.convert_to_tensor(o).dtype)
    )
    return (
      zeros, # previous output
      self.operator.initial_state(inputs)
    )

  def step(self, inputs, previous_state):
    next_output, next_state = self.operator(
      inputs=inputs,
      state=previous_state[1],
      streamable=False
    )
    inputs_any_zero = any_value(inputs, 0.0)

    selected_outputs = tf.cond(
      inputs_any_zero,
      lambda: previous_state[0],
      lambda: next_output)

    if self.operator.initial_state(inputs) == ():
      selected_next_state = ()
    else:
      selected_next_state = tf.cond(
        inputs_any_zero,
        lambda: previous_state[1],
        lambda: next_state)

    return selected_outputs, (selected_outputs, selected_next_state)
