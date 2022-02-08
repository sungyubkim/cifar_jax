import jax
import jax.numpy as jnp
import numpy as np

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