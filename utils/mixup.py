import jax
import jax.numpy as jnp
from absl import flags

flags.DEFINE_float('beta', 0.0, help='mixup coefficient')
FLAGS = flags.FLAGS

def mixup(rng, batch):
    x, y = batch['x'], batch['y']
    alpha = jax.random.beta(rng, FLAGS.beta, FLAGS.beta, (x.shape[0],))
    alpha = jnp.maximum(alpha, 1. - alpha)
    alpha_x = alpha.reshape(-1,1,1,1)
    alpha_y = alpha.reshape(-1,1)
    x_mix = alpha_x * x + (1. - alpha_x) * x[::-1]
    y_mix = alpha_y * y + (1. - alpha_y) * y[::-1]
    return {'x' : x_mix, 'y' : y_mix}