import numpy as np
import torch

from nnlib.nnlib import utils
from methods import LangevinDynamics


def discrete_mi_est(xs, ys, nx=2, ny=2):
    prob = np.zeros((nx, ny))
    for a, b in zip(xs, ys):
        prob[a,b] += 1.0/len(xs)
    pa = np.sum(prob, axis=1)
    pb = np.sum(prob, axis=0)
    mi = 0
    for a in range(nx):
        for b in range(ny):
            if prob[a,b] < 1e-9:
                continue
            mi += prob[a,b] * np.log(prob[a,b]/(pa[a]*pb[b]))
    return mi


def estimate_fcmi_bound_classification(masks, preds, num_examples, num_classes,
                                       verbose=False, return_list_of_mis=False):
    bound = 0.0
    list_of_mis = []
    for idx in range(num_examples):
        ms = [p[idx] for p in masks]
        ps = [p[2*idx:2*idx+2] for p in preds]
        for i in range(len(ps)):
            ps[i] = torch.argmax(ps[i], dim=1)
            ps[i] = num_classes * ps[i][0] + ps[i][1]
            ps[i] = ps[i].item()
        cur_mi = discrete_mi_est(ms, ps, nx=2, ny=num_classes**2)
        list_of_mis.append(cur_mi)
        bound += np.sqrt(2 * cur_mi)
        if verbose and idx < 10:
            print("ms:", ms)
            print("ps:", ps)
            print("mi:", cur_mi)
    bound *= 1/num_examples

    if return_list_of_mis:
        return bound, list_of_mis

    return bound


def estimate_sgld_bound(n, batch_size, model):
    """ Computes the bound of Negrea et al. "Information-Theoretic Generalization Bounds for
    SGLD via Data-Dependent Estimates". Eq (6) of https://arxiv.org/pdf/1911.02151.pdf.
    """
    assert isinstance(model, LangevinDynamics)
    assert model.track_grad_variance
    T = len(model._grad_variance_hist)
    assert len(model._lr_hist) == T + 1
    assert len(model._beta_hist) == T + 1
    ret = 0.0
    for t in range(1, T):  # skipping the first iteration as grad_variance was not tracked for it
        ret += model._lr_hist[t] * model._beta_hist[t] / 4.0 * model._grad_variance_hist[t-1]
    ret = np.sqrt(utils.to_numpy(ret))
    ret *= np.sqrt(n / 4.0 / batch_size / (n-1) / (n-1))
    return ret
