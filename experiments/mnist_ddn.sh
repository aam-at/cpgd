#!/usr/bin/env bash

cd ../mnist

# DDN-l2 attack
python test_foolbox.py --num_batches=2 --batch_size=500 --load_from="./models/mnist_weights_plain.mat" --attack="ddn" --norm="l2" --seed=123 --attack_steps=10000 --attack_init_epsilon=1.0 --attack_gamma=0.1 --working_dir="../results_mnist/test_plain/l2/ddn" --name="mnist_plain_ddn_foolbox_n10000_eps1.0_gamma0.1_"
python test_foolbox.py --num_batches=2 --batch_size=500 --load_from="./models/mnist_weights_plain.mat" --attack="ddn" --norm="l2" --seed=123 --attack_steps=10000 --attack_init_epsilon=1.0 --attack_gamma=0.05 --working_dir="../results_mnist/test_plain/l2/ddn" --name="mnist_plain_ddn_foolbox_n10000_eps1.0_gamma0.05_"
python test_foolbox.py --num_batches=2 --batch_size=500 --load_from="./models/mnist_weights_plain.mat" --attack="ddn" --norm="l2" --seed=123 --attack_steps=10000 --attack_init_epsilon=1.0 --attack_gamma=0.01 --working_dir="../results_mnist/test_plain/l2/ddn" --name="mnist_plain_ddn_foolbox_n10000_eps1.0_gamma0.01_"
python test_foolbox.py --num_batches=2 --batch_size=500 --load_from="./models/mnist_weights_plain.mat" --attack="ddn" --norm="l2" --seed=123 --attack_steps=10000 --attack_init_epsilon=0.1 --attack_gamma=0.1 --working_dir="../results_mnist/test_plain/l2/ddn" --name="mnist_plain_ddn_foolbox_n10000_eps0.1_gamma0.1_"
python test_foolbox.py --num_batches=2 --batch_size=500 --load_from="./models/mnist_weights_plain.mat" --attack="ddn" --norm="l2" --seed=123 --attack_steps=10000 --attack_init_epsilon=0.1 --attack_gamma=0.05 --working_dir="../results_mnist/test_plain/l2/ddn" --name="mnist_plain_ddn_foolbox_n10000_eps0.1_gamma0.05_"
python test_foolbox.py --num_batches=2 --batch_size=500 --load_from="./models/mnist_weights_plain.mat" --attack="ddn" --norm="l2" --seed=123 --attack_steps=10000 --attack_init_epsilon=0.1 --attack_gamma=0.01 --working_dir="../results_mnist/test_plain/l2/ddn" --name="mnist_plain_ddn_foolbox_n10000_eps0.1_gamma0.01_"
python test_foolbox.py --num_batches=2 --batch_size=500 --load_from="./models/mnist_weights_linf.mat" --attack="ddn" --norm="l2" --seed=123 --attack_steps=10000 --attack_init_epsilon=1.0 --attack_gamma=0.1 --working_dir="../results_mnist/test_linf/l2/ddn" --name="mnist_linf_ddn_foolbox_n10000_eps1.0_gamma0.1_"
python test_foolbox.py --num_batches=2 --batch_size=500 --load_from="./models/mnist_weights_linf.mat" --attack="ddn" --norm="l2" --seed=123 --attack_steps=10000 --attack_init_epsilon=1.0 --attack_gamma=0.05 --working_dir="../results_mnist/test_linf/l2/ddn" --name="mnist_linf_ddn_foolbox_n10000_eps1.0_gamma0.05_"
python test_foolbox.py --num_batches=2 --batch_size=500 --load_from="./models/mnist_weights_linf.mat" --attack="ddn" --norm="l2" --seed=123 --attack_steps=10000 --attack_init_epsilon=1.0 --attack_gamma=0.01 --working_dir="../results_mnist/test_linf/l2/ddn" --name="mnist_linf_ddn_foolbox_n10000_eps1.0_gamma0.01_"
python test_foolbox.py --num_batches=2 --batch_size=500 --load_from="./models/mnist_weights_linf.mat" --attack="ddn" --norm="l2" --seed=123 --attack_steps=10000 --attack_init_epsilon=0.1 --attack_gamma=0.1 --working_dir="../results_mnist/test_linf/l2/ddn" --name="mnist_linf_ddn_foolbox_n10000_eps0.1_gamma0.1_"
python test_foolbox.py --num_batches=2 --batch_size=500 --load_from="./models/mnist_weights_linf.mat" --attack="ddn" --norm="l2" --seed=123 --attack_steps=10000 --attack_init_epsilon=0.1 --attack_gamma=0.05 --working_dir="../results_mnist/test_linf/l2/ddn" --name="mnist_linf_ddn_foolbox_n10000_eps0.1_gamma0.05_"
python test_foolbox.py --num_batches=2 --batch_size=500 --load_from="./models/mnist_weights_linf.mat" --attack="ddn" --norm="l2" --seed=123 --attack_steps=10000 --attack_init_epsilon=0.1 --attack_gamma=0.01 --working_dir="../results_mnist/test_linf/l2/ddn" --name="mnist_linf_ddn_foolbox_n10000_eps0.1_gamma0.01_"
python test_foolbox.py --num_batches=2 --batch_size=500 --load_from="./models/mnist_weights_l2.mat" --attack="ddn" --norm="l2" --seed=123 --attack_steps=10000 --attack_init_epsilon=1.0 --attack_gamma=0.1 --working_dir="../results_mnist/test_l2/l2/ddn" --name="mnist_l2_ddn_foolbox_n10000_eps1.0_gamma0.1_"
python test_foolbox.py --num_batches=2 --batch_size=500 --load_from="./models/mnist_weights_l2.mat" --attack="ddn" --norm="l2" --seed=123 --attack_steps=10000 --attack_init_epsilon=1.0 --attack_gamma=0.05 --working_dir="../results_mnist/test_l2/l2/ddn" --name="mnist_l2_ddn_foolbox_n10000_eps1.0_gamma0.05_"
python test_foolbox.py --num_batches=2 --batch_size=500 --load_from="./models/mnist_weights_l2.mat" --attack="ddn" --norm="l2" --seed=123 --attack_steps=10000 --attack_init_epsilon=1.0 --attack_gamma=0.01 --working_dir="../results_mnist/test_l2/l2/ddn" --name="mnist_l2_ddn_foolbox_n10000_eps1.0_gamma0.01_"
python test_foolbox.py --num_batches=2 --batch_size=500 --load_from="./models/mnist_weights_l2.mat" --attack="ddn" --norm="l2" --seed=123 --attack_steps=10000 --attack_init_epsilon=0.1 --attack_gamma=0.1 --working_dir="../results_mnist/test_l2/l2/ddn" --name="mnist_l2_ddn_foolbox_n10000_eps0.1_gamma0.1_"
python test_foolbox.py --num_batches=2 --batch_size=500 --load_from="./models/mnist_weights_l2.mat" --attack="ddn" --norm="l2" --seed=123 --attack_steps=10000 --attack_init_epsilon=0.1 --attack_gamma=0.05 --working_dir="../results_mnist/test_l2/l2/ddn" --name="mnist_l2_ddn_foolbox_n10000_eps0.1_gamma0.05_"
python test_foolbox.py --num_batches=2 --batch_size=500 --load_from="./models/mnist_weights_l2.mat" --attack="ddn" --norm="l2" --seed=123 --attack_steps=10000 --attack_init_epsilon=0.1 --attack_gamma=0.01 --working_dir="../results_mnist/test_l2/l2/ddn" --name="mnist_l2_ddn_foolbox_n10000_eps0.1_gamma0.01_"
