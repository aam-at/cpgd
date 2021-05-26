#!/usr/bin/env bash

cd ../mnist

# Sparsefool-l1 attack
python test_sparsefool.py --num_batches=2 --batch_size=500 --seed=123 --load_from="./models/mnist_weights_plain.mat" --working_dir="../results_mnist/test_plain/l1/sparsefool" --attack_epsilon=0.02 --attack_max_iter=20 --name="mnist_sf_plain_l1_"
python test_sparsefool.py --num_batches=2 --batch_size=500 --seed=123 --load_from="./models/mnist_weights_linf.mat" --working_dir="../results_mnist/test_linf/l1/sparsefool" --attack_epsilon=0.02 --attack_max_iter=20 --name="mnist_sf_linf_l1_"
python test_sparsefool.py --num_batches=2 --batch_size=500 --seed=123 --load_from="./models/mnist_weights_l2.mat" --working_dir="../results_mnist/test_l2/l1/sparsefool" --attack_epsilon=0.02 --attack_max_iter=20 --name="mnist_sf_l2_l1_"
