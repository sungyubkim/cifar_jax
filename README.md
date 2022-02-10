# CIFAR training with JAX

A simple implementation of CIFAR training.

* Dataset : CIFAR-10/100 (refer `./cifar/data.py`)
* Network architectures : VGGNet, ResNet, Wide ResNet (refer `./model/*`)
* Cosine annealing scheduling with 10% initial steps for learning rate warmup.

## additional support
* Multi GPU training implemented with `jax.pmap` (refer `./cifar/train.py `and `./utils/mp.py`)
* MixUp regularization. (refer `./utils/mixup.py`)
* Sharpness metrics (e.g. trace of Hessian/empirical NTK) for model selection. (refer `./utils/metrics.py`)

# How to?

```bash
# train single model (multi GPU)
python3 -m cifar.train

# train single model (single GPU)
CUDA_VISIBLE_DEVICES=0 python3 -m cifar.train

# eval trained model
python3 -m cifar.train --eval

# train multiple models with run.sh
bash run.sh
```

# Results
