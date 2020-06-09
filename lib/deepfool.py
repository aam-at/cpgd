# copy from https://github.com/LTS4/DeepFool/
import copy
import logging

import numpy as np
import torch as torch
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients


def deepfool(image,
             net,
             num_classes: int=10,
             overshoot: float = 0.02,
             max_iter: int = 50,
             ord=2):
    """Deepfool attack

    :param image: Image of size HxWx3
    :param net: network (input: images, output: values of activation **BEFORE**
    softmax).
    :param num_classes: num_classes (limits the number of classes to test
    against, by default = 10)
    :param overshoot: used as a termination criterion to prevent vanishing
    updates (default = 0.02).
    :param max_iter: maximum number of iterations for deepfool (default = 50)
    :return: minimal perturbation that fools the classifier, number of
    iterations that it required, new estimated_label and perturbed image
    """
    is_cuda = torch.cuda.is_available()
    assert ord in [2, np.inf]

    if is_cuda:
        image = image.cuda()
        net = net.cuda()
        logging.debug("Using GPU")
    else:
        logging.debug("Using CPU")

    f_image = net.forward(
        Variable(image[None, :, :, :],
                 requires_grad=True)).data.cpu().numpy().flatten()
    I = (np.array(f_image)).flatten().argsort()[::-1]

    I = I[0:num_classes]
    label = I[0]

    input_shape = image.cpu().numpy().shape
    pert_image = copy.deepcopy(image)
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    loop_i = 0

    x = Variable(pert_image[None, :], requires_grad=True)
    fs = net.forward(x)
    fs_list = [fs[0, I[k]] for k in range(num_classes)]
    k_i = label

    while k_i == label and loop_i < max_iter:

        pert = np.inf
        fs[0, I[0]].backward(retain_graph=True)
        grad_orig = x.grad.data.cpu().numpy().copy()

        for k in range(1, num_classes):
            zero_gradients(x)

            fs[0, I[k]].backward(retain_graph=True)
            cur_grad = x.grad.data.cpu().numpy().copy()

            # set new w_k and new f_k
            w_k = cur_grad - grad_orig
            f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()

            if ord == 2:
                pert_k = abs(f_k) / np.linalg.norm(w_k.flatten())
            else:
                pert_k = abs(f_k) / np.sum(np.abs(w_k.flatten()))

            # determine which w_k to use
            if pert_k < pert:
                pert = pert_k
                w = w_k

        # compute r_i and r_tot
        if ord == 2:
            r_i = pert * w / np.linalg.norm(w)
        else:
            r_i = pert * np.sign(w)
        r_tot = np.float32(r_tot + r_i)

        if is_cuda:
            pert_image = image + (1 +
                                  overshoot) * torch.from_numpy(r_tot).cuda()
        else:
            pert_image = image + (1 + overshoot) * torch.from_numpy(r_tot)
        pert_image.clamp_(0.0, 1.0)
        r_tot = pert_image - image
        if is_cuda:
            r_tot = r_tot.cpu()
        r_tot = r_tot.numpy()
        x = Variable(pert_image, requires_grad=True)
        fs = net.forward(x)
        k_i = np.argmax(fs.data.cpu().numpy().flatten())

        loop_i += 1

    r_tot = (1 + overshoot) * r_tot

    return r_tot, loop_i, label, k_i, pert_image