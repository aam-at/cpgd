#!/usr/bin/env bash

cd ../mnist

# EAD-l1 attack
python test_foolbox.py --num_batches=2 --batch_size=500 --load_from="./models/mnist_weights_plain.mat" --attack="ead" --norm="l1" --seed=123 --attack_steps=10000 --attack_initial_const=0.01 --attack_binary_search_steps=9 --attack_decision_rule="L1" --attack_regularization=0.05 --attack_abort_early=False --working_dir="../results_mnist/test_plain/l1/ead" --name="mnist_plain_ead_foolbox_n10000_b0.05_C0.01_"
python test_foolbox.py --num_batches=2 --batch_size=500 --load_from="./models/mnist_weights_linf.mat" --attack="ead" --norm="l1" --seed=123 --attack_steps=10000 --attack_initial_const=0.01 --attack_binary_search_steps=9 --attack_decision_rule="L1" --attack_regularization=0.05 --attack_abort_early=False --working_dir="../results_mnist/test_linf/l1/ead" --name="mnist_linf_ead_foolbox_n10000_b0.05_C0.01_"
python test_foolbox.py --num_batches=2 --batch_size=500 --load_from="./models/mnist_weights_l2.mat" --attack="ead" --norm="l1" --seed=123 --attack_steps=10000 --attack_initial_const=0.01 --attack_binary_search_steps=9 --attack_decision_rule="L1" --attack_regularization=0.05 --attack_abort_early=False --working_dir="../results_mnist/test_l2/l1/ead" --name="mnist_l2_ead_foolbox_n10000_b0.05_C0.01_"
