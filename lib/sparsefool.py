from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Tuple, Union

import eagerpy as ep
import foolbox
from foolbox.attacks import L2DeepFoolAttack
from foolbox.attacks.base import MinimizationAttack, T, get_criterion
from foolbox.criteria import Criterion
from foolbox.distances import l1
from foolbox.models import Model


class SparseFool(MinimizationAttack, ABC):
    distance = l1

    def __init__(
        self,
        *,
        boundary_attack=None,
        steps: int = 20,
        overshoot: float = 0.02,
        lambda_fac: float = 3.0,
    ):
        if boundary_attack is None:
            boundary_attack = L2DeepFoolAttack(candidates=None)
        self.boundary_attack = boundary_attack
        self.steps = steps
        self.overshoot = overshoot
        self.lambda_fac = lambda_fac

    def run(
        self,
        model: Model,
        inputs: T,
        criterion: Union[Criterion, T],
        **kwargs: Any,
    ) -> T:
        x_0, restore_type = ep.astensor_(inputs)
        del inputs, kwargs

        criterion = get_criterion(criterion)
        y = criterion.labels

        min_, max_ = model.bounds

        x_i = x_0
        for _ in range(self.steps):
            # Approximate the decision boundary using some attack.
            boundary_point = self.boundary_attack.run(model, x_0, criterion)
            boundary_dir = boundary_point - x_0

            def loss_fn(x):
                logits = model(x)
                l = logits.argmax(axis=-1)
                rows = range(x.shape[0])
                return logits[rows, l] - logits[rows, y]

            # normal to the decision boundary
            boundary_normal = ep.value_and_grad(loss_fn, boundary_point)[1]
            boundary_normal /= ep.norms.l2(boundary_normal,
                                           axis=(1, 2, 3),
                                           keepdims=True)

            # linear solver
            x_adv = x + self.lambda_fac * boundary_dir
            x_i = self.l1_linear_solver(x_i, boundary_normal, x_adv, min_, max_)

            x_fool = x_0 + (1.0 + self.overshoot) * (x_i - x_0)
            x_fool = ep.clip(x_fool, min_, max_)
            is_adv = criterion(x_fool, model(x_fool))
            if is_adv.all():
                break

        return restore_type(x_fool)

    @classmethod
    def l1_linear_solver(cls, initial_point, boundary_point, normal, min_,
                         max_):
        """Computes the L1 solution (perturbation) to the linearized problem.
        It corresponds to algorithm 1 in [1]_.
        Parameters
        ----------
        initial_point : `numpy.ndarray`
            The initial point for which we seek the L1 solution.
        boundary_point : `numpy.ndarray`
            The point that lies on the decision boundary
            (or an overshooted version).
        normal : `numpy.ndarray`
            The normal of the decision boundary at the boundary point.
        min_ : `numpy.ndarray`
            The minimum allowed image values.
        max_ : int
            The maximum allowed image values.
        """

        coordinates = normal
        normal_vec = normal.flatten()
        boundary_point_vec = boundary_point.flatten()
        initial_point_vec = initial_point.flatten()

        # Fit the initial point to the affine hyperplane and get the sign
        f_k = np.dot(normal_vec, initial_point_vec - boundary_point_vec)
        sign_true = np.sign(f_k)
        current_sign = sign_true

        perturbed = initial_point
        while current_sign == sign_true and np.count_nonzero(coordinates) > 0:
            # Fit the current point to the hyperplane.
            f_k = np.dot(normal_vec, perturbed.flatten() - boundary_point_vec)
            f_k = f_k + (1e-3 * sign_true)  # Avoid numerical instabilities

            # Compute the L1 projection (perturbation) of the current point
            # towards the direction of the maximum
            # absolute value
            mask = np.zeros_like(coordinates)
            mask[np.unravel_index(np.argmax(np.absolute(coordinates)),
                                  coordinates.shape)] = 1

            perturbation = max(
                abs(f_k) / np.amax(np.absolute(coordinates)),
                1e-4) * mask * np.sign(coordinates)

            # Apply the perturbation
            perturbed = perturbed + perturbation
            perturbed = np.clip(perturbed, min_, max_)

            # Fit the point to the (unbiased) hyperplane and get the sign
            f_k = np.dot(normal_vec, perturbed.flatten() - boundary_point_vec)
            current_sign = np.sign(f_k)

            # Remove the used coordinate from the space of the available
            # coordinates
            coordinates[perturbation != 0] = 0

        # Return the l1 solution
        return perturbed - initial_point
