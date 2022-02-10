# CIFAR training with JAX

A simple implementation of CIFAR training.

* Dataset : CIFAR-10/100 (refer `./cifar/data.py`)
* Network architectures : VGGNet, ResNet, Wide ResNet (refer `./model/*`)
* Cosine annealing scheduling with 10% initial steps for learning rate warmup. (refer `./cifar/train.py`)

## Additional supports
* Multi GPU training implemented with `jax.pmap` (refer `./cifar/train.py `and `./utils/mp.py`)
* MixUp regularization. (refer `./utils/mixup.py`)
* Sharpness metrics (e.g. trace of Hessian/empirical NTK) implemented with Hutchison's method for model selection. (refer `./utils/metrics.py`)

# How to?

```bash
# train single model (multi GPU)
python3 -m cifar.train

# train single model (single GPU) : equivalent to jax.jit
CUDA_VISIBLE_DEVICES=0 python3 -m cifar.train

# eval trained model
python3 -m cifar.train --eval

# train multiple models with run.sh
bash run.sh

# arange tsv for multiple random seeds
python3 -m utils.arange_tsv
```

# Results

We provide two log files 

* `./res_cifar/(hyper-parameters)/log.tsv` : training log along epochs 
    * loss, accuracy for train/test, norm of weight/gradient
* `./res_cifar/(hyper-parameters)/sharpness.tsv` : sharpness log at terminal point (due to computing time)
    * trace of Hessian for single batch & entire train/test dataset
    * trace of empirical NTK for single batch & entire train/test dataset

## Benchmark results (averaged on 4 random seeds)

|  	| CIFAR-10 	| CIFAR-100 	| training time |
|---	|---	|---	|---	|
| VGGNet 	| 0.9432 ± 0.0010 | 0.7463 ± 0.0023 | 12 min/model |
| ResNet-18 | 0.9566 ± 0.0025 | 0.7895 ± 0.0030 | 20 min/model |
| WRN28-10 	| 0.9618 ± 0.0025 | 0.8022 ± 0.0002 | 60 min/model |

## Questions

* Q1 : How many samples do we need for reliable estimation of sharpness?

* Q2 : Is loss landscape for test dataset similar to the train dataset?