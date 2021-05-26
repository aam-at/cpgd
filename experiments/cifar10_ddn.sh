#!/usr/bin/env bash

cd ../cifar10

# DDN-l2 attack
python test_foolbox.py --num_batches=2 --batch_size=500 --load_from="./models/cifar10_weights_plain.mat" --attack="ddn" --norm="l2" --seed=123 --attack_steps=1000 --attack_init_epsilon=1.0 --attack_gamma=0.1 --working_dir="../results_cifar10/test_plain/l2/ddn" --name="cifar10_plain_ddn_foolbox_n1000_eps1.0_gamma0.1_"
python test_foolbox.py --num_batches=2 --batch_size=500 --load_from="./models/cifar10_weights_plain.mat" --attack="ddn" --norm="l2" --seed=123 --attack_steps=1000 --attack_init_epsilon=1.0 --attack_gamma=0.05 --working_dir="../results_cifar10/test_plain/l2/ddn" --name="cifar10_plain_ddn_foolbox_n1000_eps1.0_gamma0.05_"
python test_foolbox.py --num_batches=2 --batch_size=500 --load_from="./models/cifar10_weights_plain.mat" --attack="ddn" --norm="l2" --seed=123 --attack_steps=1000 --attack_init_epsilon=1.0 --attack_gamma=0.01 --working_dir="../results_cifar10/test_plain/l2/ddn" --name="cifar10_plain_ddn_foolbox_n1000_eps1.0_gamma0.01_"
python test_foolbox.py --num_batches=2 --batch_size=500 --load_from="./models/cifar10_weights_plain.mat" --attack="ddn" --norm="l2" --seed=123 --attack_steps=1000 --attack_init_epsilon=0.1 --attack_gamma=0.1 --working_dir="../results_cifar10/test_plain/l2/ddn" --name="cifar10_plain_ddn_foolbox_n1000_eps0.1_gamma0.1_"
python test_foolbox.py --num_batches=2 --batch_size=500 --load_from="./models/cifar10_weights_plain.mat" --attack="ddn" --norm="l2" --seed=123 --attack_steps=1000 --attack_init_epsilon=0.1 --attack_gamma=0.05 --working_dir="../results_cifar10/test_plain/l2/ddn" --name="cifar10_plain_ddn_foolbox_n1000_eps0.1_gamma0.05_"
python test_foolbox.py --num_batches=2 --batch_size=500 --load_from="./models/cifar10_weights_plain.mat" --attack="ddn" --norm="l2" --seed=123 --attack_steps=1000 --attack_init_epsilon=0.1 --attack_gamma=0.01 --working_dir="../results_cifar10/test_plain/l2/ddn" --name="cifar10_plain_ddn_foolbox_n1000_eps0.1_gamma0.01_"
python test_foolbox.py --num_batches=2 --batch_size=500 --load_from="./models/cifar10_weights_linf.mat" --attack="ddn" --norm="l2" --seed=123 --attack_steps=1000 --attack_init_epsilon=1.0 --attack_gamma=0.1 --working_dir="../results_cifar10/test_linf/l2/ddn" --name="cifar10_linf_ddn_foolbox_n1000_eps1.0_gamma0.1_"
python test_foolbox.py --num_batches=2 --batch_size=500 --load_from="./models/cifar10_weights_linf.mat" --attack="ddn" --norm="l2" --seed=123 --attack_steps=1000 --attack_init_epsilon=1.0 --attack_gamma=0.05 --working_dir="../results_cifar10/test_linf/l2/ddn" --name="cifar10_linf_ddn_foolbox_n1000_eps1.0_gamma0.05_"
python test_foolbox.py --num_batches=2 --batch_size=500 --load_from="./models/cifar10_weights_linf.mat" --attack="ddn" --norm="l2" --seed=123 --attack_steps=1000 --attack_init_epsilon=1.0 --attack_gamma=0.01 --working_dir="../results_cifar10/test_linf/l2/ddn" --name="cifar10_linf_ddn_foolbox_n1000_eps1.0_gamma0.01_"
python test_foolbox.py --num_batches=2 --batch_size=500 --load_from="./models/cifar10_weights_linf.mat" --attack="ddn" --norm="l2" --seed=123 --attack_steps=1000 --attack_init_epsilon=0.1 --attack_gamma=0.1 --working_dir="../results_cifar10/test_linf/l2/ddn" --name="cifar10_linf_ddn_foolbox_n1000_eps0.1_gamma0.1_"
python test_foolbox.py --num_batches=2 --batch_size=500 --load_from="./models/cifar10_weights_linf.mat" --attack="ddn" --norm="l2" --seed=123 --attack_steps=1000 --attack_init_epsilon=0.1 --attack_gamma=0.05 --working_dir="../results_cifar10/test_linf/l2/ddn" --name="cifar10_linf_ddn_foolbox_n1000_eps0.1_gamma0.05_"
python test_foolbox.py --num_batches=2 --batch_size=500 --load_from="./models/cifar10_weights_linf.mat" --attack="ddn" --norm="l2" --seed=123 --attack_steps=1000 --attack_init_epsilon=0.1 --attack_gamma=0.01 --working_dir="../results_cifar10/test_linf/l2/ddn" --name="cifar10_linf_ddn_foolbox_n1000_eps0.1_gamma0.01_"
python test_foolbox.py --num_batches=2 --batch_size=500 --load_from="./models/cifar10_weights_l2.mat" --attack="ddn" --norm="l2" --seed=123 --attack_steps=1000 --attack_init_epsilon=1.0 --attack_gamma=0.1 --working_dir="../results_cifar10/test_l2/l2/ddn" --name="cifar10_l2_ddn_foolbox_n1000_eps1.0_gamma0.1_"
python test_foolbox.py --num_batches=2 --batch_size=500 --load_from="./models/cifar10_weights_l2.mat" --attack="ddn" --norm="l2" --seed=123 --attack_steps=1000 --attack_init_epsilon=1.0 --attack_gamma=0.05 --working_dir="../results_cifar10/test_l2/l2/ddn" --name="cifar10_l2_ddn_foolbox_n1000_eps1.0_gamma0.05_"
python test_foolbox.py --num_batches=2 --batch_size=500 --load_from="./models/cifar10_weights_l2.mat" --attack="ddn" --norm="l2" --seed=123 --attack_steps=1000 --attack_init_epsilon=1.0 --attack_gamma=0.01 --working_dir="../results_cifar10/test_l2/l2/ddn" --name="cifar10_l2_ddn_foolbox_n1000_eps1.0_gamma0.01_"
python test_foolbox.py --num_batches=2 --batch_size=500 --load_from="./models/cifar10_weights_l2.mat" --attack="ddn" --norm="l2" --seed=123 --attack_steps=1000 --attack_init_epsilon=0.1 --attack_gamma=0.1 --working_dir="../results_cifar10/test_l2/l2/ddn" --name="cifar10_l2_ddn_foolbox_n1000_eps0.1_gamma0.1_"
python test_foolbox.py --num_batches=2 --batch_size=500 --load_from="./models/cifar10_weights_l2.mat" --attack="ddn" --norm="l2" --seed=123 --attack_steps=1000 --attack_init_epsilon=0.1 --attack_gamma=0.05 --working_dir="../results_cifar10/test_l2/l2/ddn" --name="cifar10_l2_ddn_foolbox_n1000_eps0.1_gamma0.05_"
python test_foolbox.py --num_batches=2 --batch_size=500 --load_from="./models/cifar10_weights_l2.mat" --attack="ddn" --norm="l2" --seed=123 --attack_steps=1000 --attack_init_epsilon=0.1 --attack_gamma=0.01 --working_dir="../results_cifar10/test_l2/l2/ddn" --name="cifar10_l2_ddn_foolbox_n1000_eps0.1_gamma0.01_"