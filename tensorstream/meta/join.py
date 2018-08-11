import tensorflow as tf

from tensorstream.streamable import Streamable

class Join(Streamable):
  def __init__(self, *operators):
    super().__init__()
    self.operators = operators
    self.dtype = tuple(op.dtype for op in self.operators)
    self.shape = tuple(op.shape for op in self.operators)
    self.initial_state = tuple(op.initial_state for op in self.operators)

  def step(self, *inputs_and_states):
    outputs = []
    next_states = []
    op_len = len(self.operators)

    inputs = inputs_and_states[:op_len]
    states = inputs_and_states[op_len:]

    for i in range(op_len):
      output, next_state = self.operators[i](
        inputs[i], states[i], streamable=False)
      outputs.append(output)
      next_states.append(next_state)
    return tuple(outputs), tuple(next_states)
