# https://github.com/google/flax/blob/main/examples/imagenet/models.py

from functools import partial
from typing import Any, Callable, Sequence, Tuple

from absl import flags
from flax import linen as nn

flags.DEFINE_integer('width', 64, help='') # 32, 64, 96
flags.DEFINE_integer('depth', 2, help='') # 1, 2, 3
FLAGS = flags.FLAGS

ModuleDef = Any

# define network with flax
class VGGNet(nn.Module):
  num_classes : int
  width : int
  depth : int

  @nn.compact
  def __call__(self, x, train=True):
    # VGG13-based architecture
    for mult in [1,2,4,8,8]:
      depth = self.depth if mult > 2 else min(self.depth, 2)
      for d in range(depth):
        x = nn.Conv(self.width * mult, (3, 3))(x)
        x = nn.BatchNorm(not train, momentum=0.9)(x)
        x = nn.relu(x)
      x = nn.max_pool(x, (2, 2), (2, 2), padding='SAME')
    x = x.reshape((x.shape[0],-1)) # flatten
    for i in range(3):
      x = nn.Dense(self.width * 8)(x)
      x = nn.relu(x)
    x = nn.Dense(self.num_classes)(x)
    return x