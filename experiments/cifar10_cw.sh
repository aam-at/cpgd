#!/usr/bin/env bash

cd ../cifar10

# C&W-l2 attack
python test_foolbox.py --num_batches=2 --batch_size=500 --load_from="./models/cifar10_weights_plain.mat" --attack="cw" --norm="l2" --seed=123 --attack_steps=10000 --attack_stepsize=0.01 --attack_initial_const=0.01 --attack_binary_search_steps=9 --attack_abort_early=False --working_dir="../results_cifar10/test_plain/l2/cw" --name="cifar10_plain_cw_foolbox_n10000_lr0.01_C0.01_"
python test_foolbox.py --num_batches=2 --batch_size=500 --load_from="./models/cifar10_weights_linf.mat" --attack="cw" --norm="l2" --seed=123 --attack_steps=10000 --attack_stepsize=0.01 --attack_initial_const=0.01 --attack_binary_search_steps=9 --attack_abort_early=False --working_dir="../results_cifar10/test_linf/l2/cw" --name="cifar10_linf_cw_foolbox_n10000_lr0.01_C0.01_"
python test_foolbox.py --num_batches=2 --batch_size=500 --load_from="./models/cifar10_weights_l2.mat" --attack="cw" --norm="l2" --seed=123 --attack_steps=10000 --attack_stepsize=0.01 --attack_initial_const=0.01 --attack_binary_search_steps=9 --attack_abort_early=False --working_dir="../results_cifar10/test_l2/l2/cw" --name="cifar10_l2_cw_foolbox_n10000_lr0.01_C0.01_"
