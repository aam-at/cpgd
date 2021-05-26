#!/usr/bin/env bash

cd ../mnist

# Fab-li attack
python test_fab.py --attack_norm="li" --num_batches=2 --batch_size=500 --seed=123 --attack_n_iter=100 --attack_n_restarts=1 --attack_alpha_max=0.1 --attack_eta=1.05 --attack_beta=0.9 --attack_eps=0.15 --working_dir="../results_mnist/test_plain/li/fab" --load_from="./models/mnist_weights_plain.mat" --name="mnist_fab_plain_li_n100_N1_"
python test_fab.py --attack_norm="li" --num_batches=2 --batch_size=500 --seed=123 --attack_n_iter=100 --attack_n_restarts=10 --attack_alpha_max=0.1 --attack_eta=1.05 --attack_beta=0.9 --attack_eps=0.15 --working_dir="../results_mnist/test_plain/li/fab" --load_from="./models/mnist_weights_plain.mat" --name="mnist_fab_plain_li_n100_N10_"
python test_fab.py --attack_norm="li" --num_batches=2 --batch_size=500 --seed=123 --attack_n_iter=100 --attack_n_restarts=100 --attack_alpha_max=0.1 --attack_eta=1.05 --attack_beta=0.9 --attack_eps=0.15 --working_dir="../results_mnist/test_plain/li/fab" --load_from="./models/mnist_weights_plain.mat" --name="mnist_fab_plain_li_n100_N100_"
python test_fab.py --attack_norm="li" --num_batches=2 --batch_size=500 --seed=123 --attack_n_iter=100 --attack_n_restarts=1 --attack_alpha_max=0.1 --attack_eta=1.05 --attack_beta=0.9 --attack_eps=0.3 --working_dir="../results_mnist/test_linf/li/fab" --load_from="./models/mnist_weights_linf.mat" --name="mnist_fab_linf_li_n100_N1_"
python test_fab.py --attack_norm="li" --num_batches=2 --batch_size=500 --seed=123 --attack_n_iter=100 --attack_n_restarts=10 --attack_alpha_max=0.1 --attack_eta=1.05 --attack_beta=0.9 --attack_eps=0.3 --working_dir="../results_mnist/test_linf/li/fab" --load_from="./models/mnist_weights_linf.mat" --name="mnist_fab_linf_li_n100_N10_"
python test_fab.py --attack_norm="li" --num_batches=2 --batch_size=500 --seed=123 --attack_n_iter=100 --attack_n_restarts=100 --attack_alpha_max=0.1 --attack_eta=1.05 --attack_beta=0.9 --attack_eps=0.3 --working_dir="../results_mnist/test_linf/li/fab" --load_from="./models/mnist_weights_linf.mat" --name="mnist_fab_linf_li_n100_N100_"
python test_fab.py --attack_norm="li" --num_batches=2 --batch_size=500 --seed=123 --attack_n_iter=100 --attack_n_restarts=1 --attack_alpha_max=0.1 --attack_eta=1.05 --attack_beta=0.9 --attack_eps=0.3 --working_dir="../results_mnist/test_l2/li/fab" --load_from="./models/mnist_weights_l2.mat" --name="mnist_fab_l2_li_n100_N1_"
python test_fab.py --attack_norm="li" --num_batches=2 --batch_size=500 --seed=123 --attack_n_iter=100 --attack_n_restarts=10 --attack_alpha_max=0.1 --attack_eta=1.05 --attack_beta=0.9 --attack_eps=0.3 --working_dir="../results_mnist/test_l2/li/fab" --load_from="./models/mnist_weights_l2.mat" --name="mnist_fab_l2_li_n100_N10_"
python test_fab.py --attack_norm="li" --num_batches=2 --batch_size=500 --seed=123 --attack_n_iter=100 --attack_n_restarts=100 --attack_alpha_max=0.1 --attack_eta=1.05 --attack_beta=0.9 --attack_eps=0.3 --working_dir="../results_mnist/test_l2/li/fab" --load_from="./models/mnist_weights_l2.mat" --name="mnist_fab_l2_li_n100_N100_"

# Fab-l2 attack
python test_fab.py --attack_norm="l2" --num_batches=2 --batch_size=500 --seed=123 --attack_n_iter=100 --attack_n_restarts=1 --attack_alpha_max=0.1 --attack_eta=1.05 --attack_beta=0.9 --attack_eps=2.0 --working_dir="../results_mnist/test_plain/l2/fab" --load_from="./models/mnist_weights_plain.mat" --name="mnist_fab_plain_l2_n100_N1_"
python test_fab.py --attack_norm="l2" --num_batches=2 --batch_size=500 --seed=123 --attack_n_iter=100 --attack_n_restarts=10 --attack_alpha_max=0.1 --attack_eta=1.05 --attack_beta=0.9 --attack_eps=2.0 --working_dir="../results_mnist/test_plain/l2/fab" --load_from="./models/mnist_weights_plain.mat" --name="mnist_fab_plain_l2_n100_N10_"
python test_fab.py --attack_norm="l2" --num_batches=2 --batch_size=500 --seed=123 --attack_n_iter=100 --attack_n_restarts=100 --attack_alpha_max=0.1 --attack_eta=1.05 --attack_beta=0.9 --attack_eps=2.0 --working_dir="../results_mnist/test_plain/l2/fab" --load_from="./models/mnist_weights_plain.mat" --name="mnist_fab_plain_l2_n100_N100_"
python test_fab.py --attack_norm="l2" --num_batches=2 --batch_size=500 --seed=123 --attack_n_iter=100 --attack_n_restarts=1 --attack_alpha_max=0.1 --attack_eta=1.05 --attack_beta=0.9 --attack_eps=2.0 --working_dir="../results_mnist/test_linf/l2/fab" --load_from="./models/mnist_weights_linf.mat" --name="mnist_fab_linf_l2_n100_N1_"
python test_fab.py --attack_norm="l2" --num_batches=2 --batch_size=500 --seed=123 --attack_n_iter=100 --attack_n_restarts=10 --attack_alpha_max=0.1 --attack_eta=1.05 --attack_beta=0.9 --attack_eps=2.0 --working_dir="../results_mnist/test_linf/l2/fab" --load_from="./models/mnist_weights_linf.mat" --name="mnist_fab_linf_l2_n100_N10_"
python test_fab.py --attack_norm="l2" --num_batches=2 --batch_size=500 --seed=123 --attack_n_iter=100 --attack_n_restarts=100 --attack_alpha_max=0.1 --attack_eta=1.05 --attack_beta=0.9 --attack_eps=2.0 --working_dir="../results_mnist/test_linf/l2/fab" --load_from="./models/mnist_weights_linf.mat" --name="mnist_fab_linf_l2_n100_N100_"
python test_fab.py --attack_norm="l2" --num_batches=2 --batch_size=500 --seed=123 --attack_n_iter=100 --attack_n_restarts=1 --attack_alpha_max=0.1 --attack_eta=1.05 --attack_beta=0.9 --attack_eps=2.0 --working_dir="../results_mnist/test_l2/l2/fab" --load_from="./models/mnist_weights_l2.mat" --name="mnist_fab_l2_l2_n100_N1_"
python test_fab.py --attack_norm="l2" --num_batches=2 --batch_size=500 --seed=123 --attack_n_iter=100 --attack_n_restarts=10 --attack_alpha_max=0.1 --attack_eta=1.05 --attack_beta=0.9 --attack_eps=2.0 --working_dir="../results_mnist/test_l2/l2/fab" --load_from="./models/mnist_weights_l2.mat" --name="mnist_fab_l2_l2_n100_N10_"
python test_fab.py --attack_norm="l2" --num_batches=2 --batch_size=500 --seed=123 --attack_n_iter=100 --attack_n_restarts=100 --attack_alpha_max=0.1 --attack_eta=1.05 --attack_beta=0.9 --attack_eps=2.0 --working_dir="../results_mnist/test_l2/l2/fab" --load_from="./models/mnist_weights_l2.mat" --name="mnist_fab_l2_l2_n100_N100_"

# Fab-l1 attack
python test_fab.py --attack_norm="l1" --num_batches=2 --batch_size=500 --seed=123 --attack_n_iter=100 --attack_n_restarts=1 --attack_alpha_max=0.1 --attack_eta=1.05 --attack_beta=0.9 --attack_eps=40.0 --working_dir="../results_mnist/test_plain/l1/fab" --load_from="./models/mnist_weights_plain.mat" --name="mnist_fab_plain_l1_n100_N1_"
python test_fab.py --attack_norm="l1" --num_batches=2 --batch_size=500 --seed=123 --attack_n_iter=100 --attack_n_restarts=10 --attack_alpha_max=0.1 --attack_eta=1.05 --attack_beta=0.9 --attack_eps=40.0 --working_dir="../results_mnist/test_plain/l1/fab" --load_from="./models/mnist_weights_plain.mat" --name="mnist_fab_plain_l1_n100_N10_"
python test_fab.py --attack_norm="l1" --num_batches=2 --batch_size=500 --seed=123 --attack_n_iter=100 --attack_n_restarts=100 --attack_alpha_max=0.1 --attack_eta=1.05 --attack_beta=0.9 --attack_eps=40.0 --working_dir="../results_mnist/test_plain/l1/fab" --load_from="./models/mnist_weights_plain.mat" --name="mnist_fab_plain_l1_n100_N100_"
python test_fab.py --attack_norm="l1" --num_batches=2 --batch_size=500 --seed=123 --attack_n_iter=100 --attack_n_restarts=1 --attack_alpha_max=0.1 --attack_eta=1.05 --attack_beta=0.9 --attack_eps=40.0 --working_dir="../results_mnist/test_linf/l1/fab" --load_from="./models/mnist_weights_linf.mat" --name="mnist_fab_linf_l1_n100_N1_"
python test_fab.py --attack_norm="l1" --num_batches=2 --batch_size=500 --seed=123 --attack_n_iter=100 --attack_n_restarts=10 --attack_alpha_max=0.1 --attack_eta=1.05 --attack_beta=0.9 --attack_eps=40.0 --working_dir="../results_mnist/test_linf/l1/fab" --load_from="./models/mnist_weights_linf.mat" --name="mnist_fab_linf_l1_n100_N10_"
python test_fab.py --attack_norm="l1" --num_batches=2 --batch_size=500 --seed=123 --attack_n_iter=100 --attack_n_restarts=100 --attack_alpha_max=0.1 --attack_eta=1.05 --attack_beta=0.9 --attack_eps=40.0 --working_dir="../results_mnist/test_linf/l1/fab" --load_from="./models/mnist_weights_linf.mat" --name="mnist_fab_linf_l1_n100_N100_"
python test_fab.py --attack_norm="l1" --num_batches=2 --batch_size=500 --seed=123 --attack_n_iter=100 --attack_n_restarts=1 --attack_alpha_max=0.1 --attack_eta=1.05 --attack_beta=0.9 --attack_eps=40.0 --working_dir="../results_mnist/test_l2/l1/fab" --load_from="./models/mnist_weights_l2.mat" --name="mnist_fab_l2_l1_n100_N1_"
python test_fab.py --attack_norm="l1" --num_batches=2 --batch_size=500 --seed=123 --attack_n_iter=100 --attack_n_restarts=10 --attack_alpha_max=0.1 --attack_eta=1.05 --attack_beta=0.9 --attack_eps=40.0 --working_dir="../results_mnist/test_l2/l1/fab" --load_from="./models/mnist_weights_l2.mat" --name="mnist_fab_l2_l1_n100_N10_"
python test_fab.py --attack_norm="l1" --num_batches=2 --batch_size=500 --seed=123 --attack_n_iter=100 --attack_n_restarts=100 --attack_alpha_max=0.1 --attack_eta=1.05 --attack_beta=0.9 --attack_eps=40.0 --working_dir="../results_mnist/test_l2/l1/fab" --load_from="./models/mnist_weights_l2.mat" --name="mnist_fab_l2_l1_n100_N100_"
