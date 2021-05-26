#!/usr/bin/env bash

cd ../cifar10

# JSMA-l0 attack
python test_jsma.py --num_batches=10 --batch_size=100 --seed=123 --load_from="./models/cifar10_weights_plain.mat" --working_dir="../results_cifar10/test_plain/l0/jsma" --attack_targets="all" --attack_theta=1.0 --attack_gamma=1.0 --attack_impl="art" --name="cifar10_jsma_plain_all_t1.0_g1.0_libart_"
python test_jsma.py --num_batches=10 --batch_size=100 --seed=123 --load_from="./models/cifar10_weights_plain.mat" --working_dir="../results_cifar10/test_plain/l0/jsma" --attack_targets="all" --attack_theta=0.1 --attack_gamma=1.0 --attack_impl="art" --name="cifar10_jsma_plain_all_t0.1_g1.0_libart_"
python test_jsma.py --num_batches=10 --batch_size=100 --seed=123 --load_from="./models/cifar10_weights_plain.mat" --working_dir="../results_cifar10/test_plain/l0/jsma" --attack_targets="random" --attack_theta=1.0 --attack_gamma=1.0 --attack_impl="art" --name="cifar10_jsma_plain_random_t1.0_g1.0_libart_"
python test_jsma.py --num_batches=10 --batch_size=100 --seed=123 --load_from="./models/cifar10_weights_plain.mat" --working_dir="../results_cifar10/test_plain/l0/jsma" --attack_targets="random" --attack_theta=0.1 --attack_gamma=1.0 --attack_impl="art" --name="cifar10_jsma_plain_random_t0.1_g1.0_libart_"
python test_jsma.py --num_batches=10 --batch_size=100 --seed=123 --load_from="./models/cifar10_weights_plain.mat" --working_dir="../results_cifar10/test_plain/l0/jsma" --attack_targets="second" --attack_theta=1.0 --attack_gamma=1.0 --attack_impl="art" --name="cifar10_jsma_plain_second_t1.0_g1.0_libart_"
python test_jsma.py --num_batches=10 --batch_size=100 --seed=123 --load_from="./models/cifar10_weights_plain.mat" --working_dir="../results_cifar10/test_plain/l0/jsma" --attack_targets="second" --attack_theta=0.1 --attack_gamma=1.0 --attack_impl="art" --name="cifar10_jsma_plain_second_t0.1_g1.0_libart_"
python test_jsma.py --num_batches=10 --batch_size=100 --seed=123 --load_from="./models/cifar10_weights_linf.mat" --working_dir="../results_cifar10/test_linf/l0/jsma" --attack_targets="all" --attack_theta=1.0 --attack_gamma=1.0 --attack_impl="art" --name="cifar10_jsma_linf_all_t1.0_g1.0_libart_"
python test_jsma.py --num_batches=10 --batch_size=100 --seed=123 --load_from="./models/cifar10_weights_linf.mat" --working_dir="../results_cifar10/test_linf/l0/jsma" --attack_targets="all" --attack_theta=0.1 --attack_gamma=1.0 --attack_impl="art" --name="cifar10_jsma_linf_all_t0.1_g1.0_libart_"
python test_jsma.py --num_batches=10 --batch_size=100 --seed=123 --load_from="./models/cifar10_weights_linf.mat" --working_dir="../results_cifar10/test_linf/l0/jsma" --attack_targets="random" --attack_theta=1.0 --attack_gamma=1.0 --attack_impl="art" --name="cifar10_jsma_linf_random_t1.0_g1.0_libart_"
python test_jsma.py --num_batches=10 --batch_size=100 --seed=123 --load_from="./models/cifar10_weights_linf.mat" --working_dir="../results_cifar10/test_linf/l0/jsma" --attack_targets="random" --attack_theta=0.1 --attack_gamma=1.0 --attack_impl="art" --name="cifar10_jsma_linf_random_t0.1_g1.0_libart_"
python test_jsma.py --num_batches=10 --batch_size=100 --seed=123 --load_from="./models/cifar10_weights_linf.mat" --working_dir="../results_cifar10/test_linf/l0/jsma" --attack_targets="second" --attack_theta=1.0 --attack_gamma=1.0 --attack_impl="art" --name="cifar10_jsma_linf_second_t1.0_g1.0_libart_"
python test_jsma.py --num_batches=10 --batch_size=100 --seed=123 --load_from="./models/cifar10_weights_linf.mat" --working_dir="../results_cifar10/test_linf/l0/jsma" --attack_targets="second" --attack_theta=0.1 --attack_gamma=1.0 --attack_impl="art" --name="cifar10_jsma_linf_second_t0.1_g1.0_libart_"
python test_jsma.py --num_batches=10 --batch_size=100 --seed=123 --load_from="./models/cifar10_weights_l2.mat" --working_dir="../results_cifar10/test_l2/l0/jsma" --attack_targets="all" --attack_theta=1.0 --attack_gamma=1.0 --attack_impl="art" --name="cifar10_jsma_l2_all_t1.0_g1.0_libart_"
python test_jsma.py --num_batches=10 --batch_size=100 --seed=123 --load_from="./models/cifar10_weights_l2.mat" --working_dir="../results_cifar10/test_l2/l0/jsma" --attack_targets="all" --attack_theta=0.1 --attack_gamma=1.0 --attack_impl="art" --name="cifar10_jsma_l2_all_t0.1_g1.0_libart_"
python test_jsma.py --num_batches=10 --batch_size=100 --seed=123 --load_from="./models/cifar10_weights_l2.mat" --working_dir="../results_cifar10/test_l2/l0/jsma" --attack_targets="random" --attack_theta=1.0 --attack_gamma=1.0 --attack_impl="art" --name="cifar10_jsma_l2_random_t1.0_g1.0_libart_"
python test_jsma.py --num_batches=10 --batch_size=100 --seed=123 --load_from="./models/cifar10_weights_l2.mat" --working_dir="../results_cifar10/test_l2/l0/jsma" --attack_targets="random" --attack_theta=0.1 --attack_gamma=1.0 --attack_impl="art" --name="cifar10_jsma_l2_random_t0.1_g1.0_libart_"
python test_jsma.py --num_batches=10 --batch_size=100 --seed=123 --load_from="./models/cifar10_weights_l2.mat" --working_dir="../results_cifar10/test_l2/l0/jsma" --attack_targets="second" --attack_theta=1.0 --attack_gamma=1.0 --attack_impl="art" --name="cifar10_jsma_l2_second_t1.0_g1.0_libart_"
python test_jsma.py --num_batches=10 --batch_size=100 --seed=123 --load_from="./models/cifar10_weights_l2.mat" --working_dir="../results_cifar10/test_l2/l0/jsma" --attack_targets="second" --attack_theta=0.1 --attack_gamma=1.0 --attack_impl="art" --name="cifar10_jsma_l2_second_t0.1_g1.0_libart_"
