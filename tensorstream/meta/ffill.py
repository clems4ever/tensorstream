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

  def step(self, inputs, states=None):
    inputs_any_zero = any_value(inputs, 0.0)

    if states is None:
      prev_state = None
      prev_output = None
    else:
      prev_output = states[0]
      prev_state = states[1]

    outputs, next_state, init_state = self.operator(
      inputs=inputs,
      state=prev_state,
      streamable=False
    )

    if prev_state is None:
      prev_state = init_state

    if prev_output is None:
      zero_output = map_fn(
        outputs,
        [outputs],
        lambda p: tf.zeros(tf.shape(p), p.dtype)
      )
      prev_output = zero_output

    selected_outputs = tf.cond(
      inputs_any_zero,
      lambda: prev_output,
      lambda: outputs)
    
    if init_state == ():
      selected_next_state = ()
    else:
      selected_next_state = tf.cond(
        inputs_any_zero,
        lambda: prev_state,
        lambda: next_state
      )

    return selected_outputs, (selected_outputs, selected_next_state), (prev_output, init_state)
