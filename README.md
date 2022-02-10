# CIFAR training with JAX

A simple implementation of CIFAR training.

* dataset : CIFAR-10/100 (refer ./cifar/data.py)
* network architectures : VGGNet, ResNet, Wide ResNet (refer ./model/*)
* (support) MixUp regularization. (refer ./utils/mixup.py)
* Cosine annealing scheduling with 10% steps for warmup.

# How to?

```bash
# train single model
python3 -m cifar.train

# eval trained model
python3 -m cifar.train --eval

# train multiple model with run.sh
bash run.sh
```

# Results
