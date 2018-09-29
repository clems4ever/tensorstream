import tensorflow as tf

from tensorstream.common.common import Sub, Fork, Identity
from tensorstream.common.lag import Lag
from tensorstream.common.set_during import SetDuring
from tensorstream.meta.compose import Compose
from tensorstream.meta.join import Join

def Momentum(period):
  return Compose(
    SetDuring(tf.constant(0.0), period),
    Sub(dtype=tf.float32),
    Join(Identity(tf.float32), Lag(period)),
    Fork(2, tf.float32),
  )
