#!/bin/bash

for model in vgg resnet wrn
do
    for dataset in cifar10 cifar100
    do
        for beta in 0.0 0.4
        do
            for seed in 0 1 2 3
            do
                ckpt_dir="./res_cifar/${dataset}/${model}_${beta}_0.8_1000_${seed}"
                if [ -d "${ckpt_dir}" ];then
                    echo "There is already ${ckpt_dir}."
                else
                    echo "Train ${ckpt_dir}."
                    python3 -m cifar.train \
                    --model=${model} \
                    --dataset=${dataset} \
                    --beta=${beta} \
                    --seed=${seed}
                fi
            done
        done
    done
done