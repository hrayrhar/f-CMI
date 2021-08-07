import numpy as np
import torch
import torch.nn.functional as F
import torch.autograd

from nnlib.nnlib.utils import capture_arguments_of_init
from nnlib.nnlib import losses, utils
from nnlib.nnlib.gradients import get_weight_gradients
from methods import BaseClassifier
from modules import nn_utils


class StandardClassifier(BaseClassifier):
    @capture_arguments_of_init
    def __init__(self, input_shape, architecture_args, device='cuda', **kwargs) -> object:
        super(StandardClassifier, self).__init__(**kwargs)

        self.args = None  # this will be modified by the decorator
        self.input_shape = [None] + list(input_shape)
        self.architecture_args = architecture_args

        # create the network
        self.classifier, output_shape = nn_utils.parse_network_from_config(args=self.architecture_args['classifier'],
                                                                           input_shape=self.input_shape,
                                                                           detailed_output=True)
        self.num_classes = output_shape[-1]
        self.classifier = self.classifier.to(device)

    def forward(self, inputs, labels=None, grad_enabled=False,
                detailed_output=False, **kwargs):
        torch.set_grad_enabled(grad_enabled)

        x = inputs[0].to(self.device)

        details = self.classifier(x)
        pred = details.pop('pred')

        out = {
            'pred': pred
        }

        if detailed_output:
            for k, v in details.items():
                out[k] = v

        return out

    def compute_loss(self, inputs, labels, outputs, grad_enabled, **kwargs):
        torch.set_grad_enabled(grad_enabled)

        pred = outputs['pred']
        y = labels[0].to(self.device)

        # classification loss
        y_one_hot = F.one_hot(y, num_classes=self.num_classes).float()
        classifier_loss = losses.get_classification_loss(target=y_one_hot, logits=pred,
                                                         loss_function='ce')

        batch_losses = {
            'classifier': classifier_loss,
        }

        return batch_losses, outputs


class LangevinDynamics(StandardClassifier):
    @capture_arguments_of_init
    def __init__(self, input_shape, architecture_args, device='cuda',
                 ld_lr=None, ld_beta=None, ld_schedule_fn=None,
                 ld_track_grad_variance=True, ld_track_every_iter=1, **kwargs) -> object:
        super(LangevinDynamics, self).__init__(input_shape=input_shape,
                                               architecture_args=architecture_args,
                                               device=device,
                                               **kwargs)
        self.lr = ld_lr
        self.beta = ld_beta
        self.schedule_fn = ld_schedule_fn
        self.track_grad_variance = ld_track_grad_variance
        self.track_every_iter = ld_track_every_iter

        self._iteration = 0
        self._lr_hist = [self.lr]
        self._beta_hist = [self.beta]
        self._grad_variance_hist = []

    @utils.with_no_grad
    def on_iteration_end(self, partition, tensorboard, loader, **kwargs):
        if partition != 'train':
            return
        # update lr and beta
        self._iteration += 1
        self.lr, self.beta = self.schedule_fn(lr=self.lr, beta=self.beta, iteration=self._iteration)
        self._lr_hist.append(self.lr)
        self._beta_hist.append(self.beta)
        tensorboard.add_scalar('LD_lr', self.lr, self._iteration)
        tensorboard.add_scalar('LD_beta', self.beta, self._iteration)

        # compute gradient variance
        if self.track_grad_variance:
            if (self._iteration - 1) % self.track_every_iter == 0:
                grads = get_weight_gradients(model=self, dataset=loader.dataset,
                                             max_num_examples=100,   # using 100 examples for speed
                                             use_eval_mode=True,
                                             random_selection=True)
                self.train()  # back to training mode
                grads_flattened = []
                for sample_idx in range(min(100, len(loader.dataset))):
                    cur_grad = [grads[k][sample_idx].flatten() for k in grads.keys()]
                    grads_flattened.append(torch.cat(cur_grad, dim=0))
                grads = torch.stack(grads_flattened)
                del grads_flattened
                mean_grad = torch.mean(grads, dim=0, keepdim=True)
                grad_variance = torch.sum((grads - mean_grad)**2, dim=1).mean(dim=0)
                self._grad_variance_hist.append(grad_variance)
                tensorboard.add_scalar('LD_grad_variance', grad_variance, self._iteration)
            else:
                self._grad_variance_hist.append(self._grad_variance_hist[-1])

    def before_weight_update(self, **kwargs):
        # manually doing the noisy gradient update
        for k, v in dict(self.named_parameters()).items():
            eps = torch.normal(mean=0.0, std=1/self.beta, size=v.grad.shape,
                               device=self.device, dtype=torch.float)
            update = -self.lr * v.grad + np.sqrt(2 * self.lr / self.beta) * eps
            v.data += update.data
            v.grad.zero_()

    def attributes_to_save(self):
        return {
            '_lr_hist': self._lr_hist,
            '_beta_hist': self._beta_hist,
            '_grad_variance_hist': self._grad_variance_hist
        }
