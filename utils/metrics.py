from functools import partial
import jax
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
import optax

import numpy as np
from absl import flags
from tqdm import tqdm

flags.DEFINE_integer('iter_max', 100, 
help='number of iteration for Hutchison method')
FLAGS = flags.FLAGS

@jax.pmap
def acc_batch(state, batch):
    pred = state.apply_fn(
        {'params': state.params, 'batch_stats': state.batch_stats},
        batch['x'], 
        train=False,
        )
    acc = jnp.mean(
        jnp.argmax(pred, axis=-1) == jnp.argmax(batch['y'],axis=-1)
    )
    return acc

def acc_dataset(state, dataset):
    acc_total = 0.
    n_total = 0
    for batch in dataset:
        batch_shape = batch['x'].shape
        n = batch_shape[0] * batch_shape[1]
        acc = acc_batch(state, batch)
        acc_total += np.mean(jax.device_get(acc)) * n
        n_total += n
    acc_total /= n_total
    return acc_total

@partial(jax.pmap, static_broadcasted_argnums=(0,))
def tr_hess_batch_p(loss_fn, state, batch):
    # Hutchinson's method for estimating trace of Hessian
    rng = jax.random.PRNGKey(FLAGS.seed)
    # redefine loss for HVP computation
    loss_fn_ = lambda params, inputs, targets : loss_fn(
        params, 
        state, 
        {'x' : inputs, 'y' : targets},
        False,
        )[0]
    def body_fn(_, carrier):
        res, rng = carrier
        rng, rng_r = jax.random.split(rng)
        v = jax.random.rademacher(
            rng_r, 
            (ravel_pytree(state.params)[0].size,), 
            jnp.float32,
            )
        Hv = optax.hvp(loss_fn_, v, state.params, batch['x'], batch['y'])
        Hv = ravel_pytree(Hv)[0] / batch['x'].shape[0]
        vHv = jnp.vdot(v, Hv)
        res += vHv / FLAGS.iter_max
        return res, rng
    res, rng = jax.lax.fori_loop(0, FLAGS.iter_max, body_fn, (0, rng))
    return res

def tr_hess_batch(loss_fn, state, batch):
    tr_hess = tr_hess_batch_p(loss_fn, state, batch)
    tr_hess = np.mean(jax.device_get(tr_hess))
    return tr_hess

def tr_hess_dataset(loss_fn, state, dataset):
    tr_hess_total = 0.
    n_total = 0
    for batch in tqdm(dataset):
        tr_hess = tr_hess_batch(loss_fn, state, batch)
        batch_shape = batch['x'].shape
        n = batch_shape[0] * batch_shape[1]
        tr_hess_total += tr_hess * n
        n_total += n
    tr_hess_total /= n_total
    return tr_hess_total

@jax.pmap
def tr_ntk_batch_p(state, batch):
    # Hutchinson's method for estimating trace of NTK
    rng = jax.random.PRNGKey(FLAGS.seed)
    # redefine forward for JVP computation
    def f(params):
        return state.apply_fn(
            {'params' : params, 'batch_stats': state.batch_stats},
            batch['x'], 
            train=False,
        )
    _, f_vjp = jax.vjp(f, state.params)
    def body_fn(_, carrier):
        res, rng = carrier
        _, rng = jax.random.split( rng )
        v = jax.random.rademacher(
        rng, 
        (batch['x'].shape[0], batch['y'].shape[-1]),
        jnp.float32,
        )
        j_p = ravel_pytree(f_vjp(v))[0]
        tr_ntk= jnp.sum(jnp.square(j_p)) / batch['x'].shape[0]
        res += tr_ntk / FLAGS.iter_max
        return res, rng
    a = jax.lax.fori_loop(0, FLAGS.iter_max, body_fn, (0.,rng))
    res, rng = a
    return res

def tr_ntk_batch(state, batch):
    tr_ntk = tr_ntk_batch_p(state, batch)
    tr_ntk = np.mean(jax.device_get(tr_ntk))
    return tr_ntk

def tr_ntk_dataset(state, dataset):
    tr_ntk_total = 0.
    n_total = 0
    for batch in tqdm(dataset):
        tr_ntk = tr_ntk_batch(state, batch)
        tr_ntk_total += tr_ntk * n
        batch_shape = batch['x'].shape
        n = batch_shape[0] * batch_shape[1]
        n_total += n
    tr_ntk_total /= n_total
    return tr_ntk_total