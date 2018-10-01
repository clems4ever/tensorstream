from tensorstream.streamable import MetaStreamable

class Join(MetaStreamable):
  def __init__(self, *operators):
    super().__init__()
    self.operators = operators

  def step(self, inputs, states=None):
    if states is None:
      states = [None] * len(self.operators)

    outputs = []
    initial_states = []
    next_states = []
    op_in_st = zip(self.operators, inputs, states)
    for op, inputs_, state in op_in_st:
      output, next_state, initial_state = op(
        inputs_, state, streamable=False)
      initial_states.append(initial_state)
      outputs.append(output)
      next_states.append(next_state)
    return tuple(outputs), tuple(next_states), tuple(initial_states)
