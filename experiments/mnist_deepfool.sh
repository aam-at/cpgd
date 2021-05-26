#!/usr/bin/env bash

cd ../mnist

# DF-li attack
python test_deepfool.py --num_batches=2 --batch_size=500 --norm="li" --seed=123 --load_from="./models/mnist_weights_plain.mat" --working_dir="../results_mnist/test_plain/li/df" --attack_overshoot=0.02 --attack_max_iter=100 --name="mnist_plain_df_li_n100_os0.02_"
python test_deepfool.py --num_batches=2 --batch_size=500 --norm="li" --seed=123 --load_from="./models/mnist_weights_linf.mat" --working_dir="../results_mnist/test_linf/li/df" --attack_overshoot=0.02 --attack_max_iter=100 --name="mnist_linf_df_li_n100_os0.02_"
python test_deepfool.py --num_batches=2 --batch_size=500 --norm="li" --seed=123 --load_from="./models/mnist_weights_l2.mat" --working_dir="../results_mnist/test_l2/li/df" --attack_overshoot=0.02 --attack_max_iter=100 --name="mnist_l2_df_li_n100_os0.02_"

# DF-l2 attack
python test_deepfool.py --num_batches=2 --batch_size=500 --norm="l2" --seed=123 --load_from="./models/mnist_weights_plain.mat" --working_dir="../results_mnist/test_plain/l2/df" --attack_overshoot=0.02 --attack_max_iter=100 --name="mnist_plain_df_li_n100_os0.02_"
python test_deepfool.py --num_batches=2 --batch_size=500 --norm="l2" --seed=123 --load_from="./models/mnist_weights_linf.mat" --working_dir="../results_mnist/test_linf/l2/df" --attack_overshoot=0.02 --attack_max_iter=100 --name="mnist_linf_df_li_n100_os0.02_"
python test_deepfool.py --num_batches=2 --batch_size=500 --norm="l2" --seed=123 --load_from="./models/mnist_weights_l2.mat" --working_dir="../results_mnist/test_l2/l2/df" --attack_overshoot=0.02 --attack_max_iter=100 --name="mnist_l2_df_li_n100_os0.02_"
