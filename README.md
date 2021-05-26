# PDPGD: Primal-Dual Proximal Gradient Descent Adversarial Attack

[Alexander Matyasko](https://github.com/aam-at), Lap-Pui Chau, [**PDPGD: Primal-Dual Proximal Gradient Descent Adversarial Attack**](https://arxiv.org/abs/2106.01538).

State-of-the-art deep neural networks are sensitive to small input perturbations. Since the discovery of this intriguing vulnerability, many defence methods have been proposed that attempt to improve robustness to adversarial noise. Fast and accurate attacks are required to compare various defence methods. However, evaluating adversarial robustness has proven to be extremely challenging. Existing norm minimisation adversarial attacks require thousands of iterations (e.g. Carlini & Wagner attack), are limited to the specific norms (e.g. Fast Adaptive Boundary), or produce sub-optimal results (e.g. Brendel & Bethge attack). On the other hand, PGD attack, which is fast, general and accurate, ignores the norm minimisation penalty and solves a simpler perturbation-constrained problem. In this work, we introduce a fast, general and accurate adversarial attack that optimises the original non-convex constrained minimisation problem. We interpret optimising the Lagrangian of the adversarial attack optimisation problem as a two-player game: the first player minimises the Lagrangian wrt the adversarial noise; the second player maximises the Lagrangian wrt the regularisation penalty. Our attack algorithm simultaneously optimises primal and dual variables to find the minimal adversarial perturbation. In addition, for non-smooth lp-norm minimisation, such as linf-, l1-, and l0-norms, we introduce primal-dual proximal gradient descent attack. We show in the experiments that our attack outperforms current state-of-the-art linf-, l2-, l1-, and l0-attacks on MNIST, CIFAR-10 and Restricted ImageNet datasets against unregularised and adversarially trained models.


```txt
@Article{	  matyasko2021pdpgd,
  author	= {Alexander Matyasko, Lap-Pui Chau},
  title		= {PDPGD: Primal-Dual Proximal Gradient Descent Adversarial
		  Attack},
  journal	= {arXiv e-prints},
  year		= 2021,
  pages		= {arXiv:2106.01538},
  primaryclass	= {cs.LG}
}
```
## Installation

Clone the repository and add the repository directory to `PYTHONPATH`. `git-lfs`
must be installed to automatically download the naturally trained, linf- and l2-
adversarially trained models for MNIST and CIFAR-10. The models for the
experiments on Restricted ImageNet can be downloaded from
[url](https://github.com/MadryLab/robust-features-code).

```bash
git clone https://github.com/aam-at/cpgd
export PYTHONPATH="$PYTHONPATH:<PATH_TO_CLONE_DIR>"
```

## Requirements (tested with python 3.6)
- Tensorflow v2.2.0
- Pytorch v1.5.0
- Foolbox v3.2.1
- Cleverhans (https://github.com/aam-at/cleverhans_tf2)
- Adversarial Robustness Toolbox v1.4.2

See `requirements.txt` for the list of full requirements.

## Implementation details

Our attacks and utilities are implemented in `lib` directory. `attack_lp`
implements abstract classes `PrimalDualGradientAttack` and
`ProximalPrimalDualGradientAttack`, which corresponds to our PDGD and PDPGD
attacks, respectively.

## Steps to reproduce results

We provide scripts to reproduce all the results for all datasets, attacks and
models in folder `experiments`.

For example, one can run all experiments on MNIST for DeepFool li- and l2-norm
attack using the following commands:

```bash
cd experiments
./mnist_deepfool.sh
```

We provide scripts to parse the results of all experiments and export it as an
Excel spreadsheet:

```bash
cd mnist
python parse_results.py
```
