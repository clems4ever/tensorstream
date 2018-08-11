from tensorstream.operators import Streamable
from tensorstream.operators.meta.join import Join

def Factory(operator_clazz, operators_args):
  instances = tuple(operator_clazz(*args) for args in operators_args)
  return Join(*instances)
