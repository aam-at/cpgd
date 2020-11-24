# code: https://github.com/LTS4/SparseFool
import copy

import numpy as np
import torch as torch
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients


def deepfool(im,
             net,
             lambda_fac=3.0,
             num_classes=None,
             overshoot=0.02,
             max_iter=50,
             device="cuda"):
    assert im.shape[0] == 1

    image = copy.deepcopy(im)
    input_shape = image.size()

    f_image = (net.forward(Variable(
        image, requires_grad=True)).data.cpu().numpy().flatten())
    I = (np.array(f_image)).flatten().argsort()[::-1]
    if num_classes is None:
        num_classes = I.shape[-1]
    I = I[0:num_classes]
    label = I[0]

    pert_image = copy.deepcopy(image)
    r_tot = torch.zeros(input_shape).to(device)

    k_i = label
    loop_i = 0

    while k_i == label and loop_i < max_iter:

        x = Variable(pert_image, requires_grad=True)
        fs = net.forward(x)

        pert = torch.Tensor([np.inf])[0].to(device)
        w = torch.zeros(input_shape).to(device)

        fs[0, I[0]].backward(retain_graph=True)
        grad_orig = copy.deepcopy(x.grad.data)

        for k in range(1, num_classes):
            zero_gradients(x)

            fs[0, I[k]].backward(retain_graph=True)
            cur_grad = copy.deepcopy(x.grad.data)

            w_k = cur_grad - grad_orig
            f_k = (fs[0, I[k]] - fs[0, I[0]]).data

            pert_k = torch.abs(f_k) / w_k.norm()

            if pert_k < pert:
                pert = pert_k + 0.0
                w = w_k + 0.0

        r_i = torch.clamp(pert, min=1e-4) * w / w.norm()
        r_tot = r_tot + r_i

        pert_image = pert_image + r_i

        check_fool = image + (1 + overshoot) * r_tot
        k_i = torch.argmax(
            net.forward(Variable(check_fool, requires_grad=True)).data).item()

        loop_i += 1

    x = Variable(pert_image, requires_grad=True)
    fs = net.forward(x)
    (fs[0, k_i] - fs[0, label]).backward(retain_graph=True)
    grad = copy.deepcopy(x.grad.data)
    grad = grad / grad.norm()

    r_tot = lambda_fac * r_tot
    pert_image = image + r_tot

    return grad, pert_image


def linear_solver(x_0, normal, boundary_point, lb, ub):
    assert x_0.shape[0] == 1
    device = x_0.device

    # copy to cpu
    x_0 = x_0.cpu()
    normal = normal.cpu()
    boundary_point = boundary_point.cpu()

    input_shape = x_0.size()

    coord_vec = copy.deepcopy(normal)
    plane_normal = copy.deepcopy(coord_vec).view(-1)
    plane_point = copy.deepcopy(boundary_point).view(-1)

    x_i = copy.deepcopy(x_0)

    f_k = torch.dot(plane_normal, x_0.view(-1) - plane_point)
    sign_true = f_k.sign().item()

    beta = 0.001 * sign_true
    current_sign = sign_true

    while current_sign == sign_true and coord_vec.nonzero().size()[0] > 0:
        f_k = torch.dot(plane_normal, x_i.view(-1) - plane_point) + beta

        pert = f_k.abs() / coord_vec.abs().max()

        mask = torch.zeros_like(coord_vec)
        mask[np.unravel_index(torch.argmax(coord_vec.abs()),
                              input_shape)] = 1.0

        r_i = torch.clamp(pert, min=1e-4) * mask * coord_vec.sign()

        x_i = x_i + r_i
        x_i.clamp_(lb, ub)

        f_k = torch.dot(plane_normal, x_i.view(-1) - plane_point)
        current_sign = f_k.sign().item()

        coord_vec[r_i != 0] = 0

    return x_i.to(device)


def sparsefool(x_0,
               net,
               lb,
               ub,
               lambda_: float = 3.0,
               max_iter: int = 20,
               epsilon: float = 0.02,
               device: str="cuda"):
    assert x_0.shape[0] == 1

    pred_label = torch.argmax(
        net.forward(Variable(x_0, requires_grad=True)).data).item()

    x_i = copy.deepcopy(x_0)
    fool_im = copy.deepcopy(x_i)

    fool_label = pred_label
    loops = 0

    while fool_label == pred_label and loops < max_iter:
        normal, x_adv = deepfool(x_i, net, lambda_, device=device)

        x_i = linear_solver(x_i, normal, x_adv, lb, ub)

        fool_im = x_0 + (1 + epsilon) * (x_i - x_0)
        fool_im.clamp_(lb, ub)
        fool_label = torch.argmax(
            net.forward(Variable(fool_im, requires_grad=True)).data).item()

        loops += 1

    r = fool_im - x_0
    return fool_im, r, pred_label, fool_label, loops
