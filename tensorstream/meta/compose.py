from tensorstream.streamable import MetaStreamable

class Compose(MetaStreamable):
  def __init__(self, *operators):
    super().__init__()
    self.operators = tuple(reversed(operators))

  def properties(self, *inputs_placeholders):
    x = inputs_placeholders
    states = []
    for op in self.operators:
      if isinstance(x, (tuple, list)):
        y, init_states = op.properties(*x)
      else:
        y, init_states = op.properties(x)
      states.append(init_states)
      x = y
    return y, tuple(states)

  def step(self, inputs, states):
    next_states = []
    op_inputs = inputs
    op_states = list(zip(self.operators, states))

    for operator, state in op_states:
      output, next_state = operator(inputs=op_inputs, state=state, streamable=False)
      op_inputs = output
      next_states.append(next_state)

    return output, tuple(next_states)
