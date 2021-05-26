#!/usr/bin/env bash

cd ../mnist

# CornerSearch-l0 attack
python test_cornersearch.py --num_batches=2 --batch_size=500 --seed=123 --load_from="./models/mnist_weights_plain.mat" --working_dir="../results_mnist/test_plain/l0/cornersearch" --attack_sparsity=784 --name="mnist_cs_plain_l0_"
python test_cornersearch.py --num_batches=2 --batch_size=500 --seed=123 --load_from="./models/mnist_weights_linf.mat" --working_dir="../results_mnist/test_linf/l0/cornersearch" --attack_sparsity=784 --name="mnist_cs_linf_l0_"
python test_cornersearch.py --num_batches=2 --batch_size=500 --seed=123 --load_from="./models/mnist_weights_l2.mat" --working_dir="../results_mnist/test_l2/l0/cornersearch" --attack_sparsity=784 --name="mnist_cs_l2_l0_"
