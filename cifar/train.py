from typing import Any
from functools import partial
from absl import app, flags
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import flax
from flax import linen as nn
from flax.training import train_state, checkpoints
from flax.jax_utils import replicate
import numpy as np
import optax
import tensorflow_datasets as tfds
from tqdm import tqdm

from utils import ckpt, metrics, mixup, mp
from model import vgg, resnet
from cifar.data import load_dataset

import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

# additional hyper-parameters
flags.DEFINE_integer('epoch_num', 200, 
help='epoch number of pre-training')
flags.DEFINE_float('max_norm', 5.0, 
help='maximum norm of clipped gradient')
flags.DEFINE_enum('dataset', 'cifar100', ['cifar10', 'cifar100'],
help='training dataset')
flags.DEFINE_enum('model', 'resnet', ['vgg', 'resnet', 'wrn'],
help='network architecture')
flags.DEFINE_integer('seed', 0, 
help='random number seed')
flags.DEFINE_bool('eval', False, 
help='do not training')
flags.DEFINE_integer('test_batch_size_total', 1000, 
help='total batch size (not device-wise) for evaluation')
flags.DEFINE_integer('log_freq',10,
help='(epoch) frequency of logging')

# tunable hparams for generalization
flags.DEFINE_float('weight_decay', 0.0005, 
help='l2 regularization coeffcient')
flags.DEFINE_float('peak_lr', 0.8, 
help='peak learning during learning rate schedule')
flags.DEFINE_integer('train_batch_size_total', 1000, 
help='total batch size (not device-wise) for training')
FLAGS = flags.FLAGS

class TrainState(train_state.TrainState):
    batch_stats: Any

def create_lr_sched(num_train):
    total_step = FLAGS.epoch_num * (num_train // FLAGS.train_batch_size_total)
    warmup_step = int(0.1 * total_step)
    return optax.warmup_cosine_decay_schedule(0.0, FLAGS.peak_lr, warmup_step, total_step)

def init_state(rng, batch, num_classes, num_train):
    # parsing model
    if FLAGS.model=='vgg':
        net = vgg.VGGNet(num_classes=num_classes)
    if FLAGS.model=='resnet':
        net = resnet.ResNet18(num_classes=num_classes)
    elif FLAGS.model=='wrn':
        net = resnet.WRN28_10(num_classes=num_classes)
        
    variables = net.init(rng, batch)
    params, batch_stats = variables['params'], variables['batch_stats']
    tx = optax.chain(
    optax.clip_by_global_norm(FLAGS.max_norm),
    optax.sgd(
        learning_rate=create_lr_sched(num_train),
        momentum=0.9, 
        nesterov=True,
        )
    )
    state = TrainState.create(
    apply_fn=net.apply, 
    params=params, 
    tx=tx, 
    batch_stats = batch_stats,
    ) 
    return state

def loss_fn(params, state, batch, train):
    if train:
        logits, new_net_state = state.apply_fn(
            {'params':params, 'batch_stats': state.batch_stats},
            batch['x'], train=train, mutable=['batch_stats'],
        )
    else:
        logits = state.apply_fn(
            {'params':params, 'batch_stats': state.batch_stats},
            batch['x'], train=train,
        )
        new_net_state = None
    loss = optax.l2_loss(logits, batch['y']).sum(axis=-1).mean()
    wd = 0.5 * jnp.sum(jnp.square(ravel_pytree(params)[0]))
    loss_ = loss + FLAGS.weight_decay * wd
    acc = jnp.mean(
        jnp.argmax(logits, axis=-1) == jnp.argmax(batch['y'],axis=-1)
        )
    return loss_, (loss, wd, acc, new_net_state)

@partial(jax.pmap, axis_name='batch')
def opt_step(rng, state, batch):
    batch = mixup.mixup(rng, batch)
    grad_fn = jax.grad(loss_fn, has_aux=True)
    grads, (loss, wd, acc, new_net_state) = grad_fn(
        state.params, 
        state, 
        batch, 
        True,
        )
    # sync and update
    grads = jax.lax.pmean(grads, axis_name='batch')
    batch_stats = jax.lax.pmean(new_net_state['batch_stats'], axis_name='batch')
    new_state = state.apply_gradients(
    grads=grads, batch_stats=batch_stats
    )
    # log norm of gradient
    grad_norm = jnp.sum(jnp.square(ravel_pytree(grads)[0]))
    return loss, wd, grad_norm, acc, new_state

def main(_):
    num_devices = jax.device_count()
    batch_dims_tr = (num_devices, FLAGS.train_batch_size_total//num_devices)
    batch_dims_te = (num_devices, FLAGS.test_batch_size_total//num_devices)
    ds_tr, ds_info = tfds.load(
        "{}:3.*.*".format(FLAGS.dataset),
        data_dir='../tensorflow_datasets',
        split='train',
        with_info=True,
    )
    ds_te = tfds.load(
        "{}:3.*.*".format(FLAGS.dataset), 
        data_dir='../tensorflow_datasets',
        split='test', 
    )
    # extract info. of dataset
    ds_tr, ds_te = ds_tr.cache(), ds_te.cache()
    img_shape = ds_info.features['image'].shape
    label_info = ds_info.features['label']
    class_names = label_info.names
    num_classes = label_info.num_classes
    num_train = ds_info.splits['train'].num_examples
    num_test = ds_info.splits['test'].num_examples

    hparams = [
        FLAGS.model,
        FLAGS.beta,
        FLAGS.peak_lr,
        FLAGS.train_batch_size_total,
        FLAGS.seed,
        ]
    hparams = '_'.join(map(str, hparams))
    res_dir = f'./res_cifar/{FLAGS.dataset}/'+hparams

    print(f'hyper-parameters : {hparams}')
    ckpt.check_dir(res_dir)

    # define pseudo-random number generator
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, rng_ = jax.random.split(rng)

    # initialize network and optimizer
    state = init_state(
        rng_, 
        jax.random.normal(rng_, (1, *img_shape)), 
        num_classes,
        num_train,
        )
    if FLAGS.eval:
        state = checkpoints.restore_checkpoint(
            res_dir,
            state,
            )
    state = replicate(state)

    # train model
    eval_tr = load_dataset(
        ds_tr, 
        batch_dims=batch_dims_te, 
        aug=False, 
        num_classes=num_classes,
        )
    eval_te = load_dataset(
        ds_te, 
        batch_dims=batch_dims_te, 
        aug=False, 
        num_classes=num_classes,
        )
    eval_tr = list(eval_tr)
    eval_te = list(eval_te)

    if not(FLAGS.eval):
        pbar = tqdm(range(1,FLAGS.epoch_num+1))
        for epoch in pbar:
            train_dataset = load_dataset(
                ds_tr, 
                batch_dims=batch_dims_tr, 
                aug=True, 
                num_classes=num_classes,
                )
            for batch_tr in train_dataset:
                rng, rng_ = jax.random.split(rng)
                loss, wd, grad_norm, acc, state = opt_step(
                    replicate(rng_), 
                    state, 
                    batch_tr,
                    )
                res = {
                    'epoch': epoch,
                    'acc' : f'{np.mean(jax.device_get(acc)):.4f}',
                    'loss': f'{np.mean(jax.device_get(loss)):.4f}',
                    'wd' : f'{np.mean(jax.device_get(wd)):.4f}',
                    'grad_norm' : f'{np.mean(jax.device_get(grad_norm)):.4f}',
                    }
                pbar.set_postfix(res)

            if (epoch%FLAGS.log_freq)==0:
                ckpt.save_ckpt(state, res_dir)
                acc_tr = metrics.acc_dataset(state, eval_tr)
                res['acc_tr'] = f'{acc_tr:.4f}'
                acc_te = metrics.acc_dataset(state, eval_te)
                res['acc_te'] = f'{acc_te:.4f}'
                ckpt.dict2tsv(res, res_dir+'/log.tsv')
    
    # evaluate
    res = {}
    acc_tr = metrics.acc_dataset(state, eval_tr)
    res['acc_tr'] = f'{acc_tr:.4f}'
    acc_te = metrics.acc_dataset(state, eval_te)
    res['acc_te'] = f'{acc_te:.4f}'
    # Q1 : How many samples do we need? Is mini-batch (M>1000) sufficient?
    tr_hess_batch = metrics.tr_hess_batch(loss_fn, state, eval_tr[0])
    tr_hess_dataset = metrics.tr_hess_dataset(loss_fn, state, eval_tr)
    tr_ntk_batch = metrics.tr_ntk_batch(state, eval_tr[0])
    tr_ntk_dataset = metrics.tr_ntk_dataset(state, eval_tr)
    res['tr_hess_batch'] = f'{tr_hess_batch:.4f}'
    res['tr_hess_dataset'] = f'{tr_hess_dataset:.4f}'
    res['tr_ntk_batch'] = f'{tr_ntk_batch:.4f}'
    res['tr_ntk_dataset'] = f'{tr_ntk_dataset:.4f}'

    # Q2 : Is loss landscape for test dataset similar to train dataset?
    tr_hess_dataset_te = metrics.tr_hess_dataset(loss_fn, state, eval_te)
    tr_ntk_dataset_te = metrics.tr_ntk_dataset(state, eval_te)
    res['tr_hess_dataset_te'] = f'{tr_hess_dataset_te:.4f}'
    res['tr_ntk_dataset_te'] = f'{tr_ntk_dataset_te:.4f}'
    ckpt.dict2tsv(res, res_dir+'/sharpness.tsv')

if __name__ == "__main__":
    app.run(main)