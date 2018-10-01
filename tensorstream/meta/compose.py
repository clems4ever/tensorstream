from tensorstream.streamable import MetaStreamable

class Compose(MetaStreamable):
  def __init__(self, *operators):
    super().__init__()
    self.operators = tuple(reversed(operators))

  def step(self, inputs, states=None):
    if states is None:
      states = tuple([None] * len(self.operators))

    next_states = []
    initial_states = []
    op_inputs = inputs
    op_states = list(zip(self.operators, states))

    for operator, state in op_states:
      output, next_state, initial_state = operator(
        inputs=op_inputs, state=state, streamable=False)
      op_inputs = output
      initial_states.append(initial_state)
      next_states.append(next_state)

    return output, tuple(next_states), tuple(initial_states)
