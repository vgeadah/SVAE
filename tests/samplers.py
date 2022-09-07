
r'''
Sampling schemes for transition operators in AIS. 
HMC code copied form: https://github.com/lxuechen/BDMC/blob/master/hmc.py 
'''
from typing import Callable
from typing import Optional

import torch

def metropolis(z, f, n_steps=10):
    '''
    Transition operator T(z'|z) using n-steps Metropolis sampler.
    '''
    for _ in range(n_steps):
        z_prime = z #+ torch.randn(z.shape) # Proposal
        accept_rate = torch.where(f(z)>0.0, f(z_prime)/f(z), torch.zeros_like(f(z)))     # Acceptance prob.
        if torch.linalg.norm(accept_rate)>0.0: print(accept_rate)
        rand_rate = torch.rand(z.shape[0])
        for i in range(z.shape[0]):             # TODO: probably smarter way than loop.
            if rand_rate[i] < accept_rate[i]:
                z[i] = z_prime[i]
    return z

def hmc_trajectory(current_z: torch.Tensor,
                   current_v: torch.Tensor,
                   grad_U: Callable,
                   epsilon: torch.Tensor,
                   L: Optional[int] = 10):
    """Propose new state-velocity pair with leap-frog integrator.

    This function does not yet do the accept-reject step.
    Follows algo box in Figure 2 of https://arxiv.org/pdf/1206.1901.pdf.

    Args:
        current_z: current position
        current_v: current velocity/momentum
        grad_U: function to compute gradients w.r.t. U
        epsilon: step size
        L: number of leap-frog steps

    Returns:
        proposed state z and velocity v after the leap-frog steps
    """
    epsilon = epsilon.view(-1, 1)
    z = current_z
    v = current_v - .5 * epsilon * grad_U(z)

    for i in range(1, L + 1):
        z = z + epsilon * v
        if i != L:
            v = v - epsilon * grad_U(z)

    v = v - .5 * epsilon * grad_U(z)
    v = -v

    return z, v


def hmc_accept_reject(current_z: torch.Tensor,
                  current_v: torch.Tensor,
                  z: torch.Tensor,
                  v: torch.Tensor,
                  epsilon: torch.Tensor,
                  accept_hist: torch.Tensor,
                  hist_len: int,
                  U: Callable,
                  K: Callable,
                  max_step_size: Optional[float] = 2.0,
                  min_step_size: Optional[float] = 1e-12,
                  acceptance_threshold: Optional[float] = 0.65):
    """Accept/reject based on Hamiltonians for current and propose.

    Args:
        current_z: position *before* leap-frog steps
        current_v: speed *before* leap-frog steps
        z: position *after* leap-frog steps
        v: speed *after* leap-frog steps
        epsilon: step size of leap-frog.
        accept_hist: a tensor of size (batch_size,), each component of which is
            the number of time the trajectory is accepted
        hist_len: an int for the chain length after the current step
        U: function to compute potential energy
        K: function to compute kinetic energy
        max_step_size: maximum step size for leap-frog
        min_step_size: minimum step size for leap-frog
        acceptance_threshold: threshold acceptance rate; increase the step size
            if the chain is accepted more than this, and decrease otherwise

    Returns:
        the new state z, the adapted step size epsilon, and the updated
        accept-reject history
    """
    current_Hamil = K(current_v) + U(current_z)
    propose_Hamil = K(v) + U(z)

    prob = torch.clamp_max(torch.exp(current_Hamil - propose_Hamil), 1.)
    accept = torch.gt(prob, torch.rand_like(prob))
    z = accept.view(-1, 1) * z + ~accept.view(-1, 1) * current_z

    accept_hist.add_(accept)
    criteria = torch.gt(accept_hist / hist_len, acceptance_threshold)
    # accept_rate = torch.sum(torch.ones_like(criteria)[criteria])/criteria.shape[0]
    # adapt = criteria * 1.2 + ~criteria * 0.8
    epsilon.clamp_(min_step_size, max_step_size)
    
    # print('accept rate (AR): {:1.4f}, cumul avg AR: {:1.4f}, mean eps: {:1.1e}'.format(
    #     torch.sum(torch.ones_like(accept)[accept])/accept.shape[0],
    #     torch.mean(accept_hist / hist_len),
    #     torch.mean(epsilon)
    #     ))
    return z, epsilon, accept_hist