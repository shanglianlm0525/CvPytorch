# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/8/11 8:19
# @Author : liumin
# @File : adabelief.py
import math

import torch
from torch.optim import Adam
from typing import Optional, Callable


class AdaBelief(Adam):
    """Implements the AdaBelief optimizer from `"AdaBelief Optimizer: Adapting Stepsizes by the Belief in
    Observed Gradients" <https://arxiv.org/pdf/2010.07468.pdf>`_.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): learning rate
        betas (Tuple[float, float], optional): coefficients used for running averages (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (bool, optional): whether to use the AMSGrad variant (default: False)
    """

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('AdaBelief does not support sparse gradients')
                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if group['amsgrad']:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])

                    if group['amsgrad']:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])

            beta1, beta2 = group['betas']
            self.adabelief(params_with_grad,
                        grads,
                        exp_avgs,
                        exp_avg_sqs,
                        max_exp_avg_sqs,
                        state_steps,
                        group['amsgrad'],
                        beta1,
                        beta2,
                        group['lr'],
                        group['weight_decay'],
                        group['eps'],
                        )
        return loss

    def adabelief(self, params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs,
                  state_steps, amsgrad, beta1, beta2, lr, weight_decay, eps):
        r"""Functional API that performs AdaBelief algorithm computation.
        See :class:`~holocron.optim.AdaBelief` for details.
        """

        for i, param in enumerate(params):

            grad = grads[i]
            exp_avg = exp_avgs[i]
            exp_avg_sq = exp_avg_sqs[i]
            step = state_steps[i]
            if amsgrad:
                max_exp_avg_sq = max_exp_avg_sqs[i]

            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step

            if weight_decay != 0:
                grad = grad.add(param, alpha=weight_decay)

            # Decay the first and second moment running average coefficient
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            grad_residual = grad - exp_avg
            exp_avg_sq.mul_(beta2).addcmul_(grad_residual, grad_residual, value=1 - beta2)

            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)  # type: ignore[attr-defined]
                # Use the max. for normalizing running avg. of gradient
                denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
            else:
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

            step_size = lr / bias_correction1

            param.addcdiv_(exp_avg, denom, value=-step_size)