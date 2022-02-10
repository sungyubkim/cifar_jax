# CIFAR training with JAX

A simple implementation of CIFAR training.

* dataset : CIFAR-10/100 (refer ./cifar/data.py)
* network architectures : VGGNet, ResNet, Wide ResNet (refer ./model/*)
* cosine annealing scheduling with 10% steps for warmup.

## additional support
* Multi GPU training (refer ./cifar/train.py and ./utils/mp.py)
* MixUp regularization. (refer ./utils/mixup.py)
* Sharpness metrics (e.g. trace of Hessian/empirical NTK) for model selection. (refer ./utils/metrics.py)

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
