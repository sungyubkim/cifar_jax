# CIFAR training with JAX

A simple implementation of CIFAR training.

* dataset : CIFAR-10/100 (refer ./cifar/data.py)
* network architectures : VGGNet, ResNet, Wide ResNet (refer ./model/*)
* cosine annealing scheduling with 10% steps for warmup.
* support MixUp regularization. (refer ./utils/mixup.py)
* support sharpness (e.g. trace of Hessian/empirical NTK) metrics for model selection. (refer ./utils/metrics.py)

# How to?

```bash
# train single model
python3 -m cifar.train

# eval trained model
python3 -m cifar.train --eval

# train multiple models with run.sh
bash run.sh
```

# Results
