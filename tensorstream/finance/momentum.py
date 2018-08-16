import tensorflow as tf

from tensorstream.meta import Compose, Join, Sub, Fork, Identity
from tensorstream.common.lag import Lag

def Momentum(period, dtype=tf.float32, shape=()):
  return Compose(
    Sub(dtype=dtype, shape=shape),
    Join(Identity(dtype=dtype, shape=shape), Lag(period=period, dtype=dtype, shape=shape)),
    Fork(2, dtype=dtype, shape=shape),
  )
