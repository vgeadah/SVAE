
r'''
Annealed Importance Sampling procedure for estimation of the log likelihood for SVAE models. 
References:
    [1] Annealed Importance Sampling, R. Neal. https://arxiv.org/pdf/physics/9803008.pdf
    [2] https://github.com/lxuechen/BDMC
    [3] https://github.com/tonywu95/eval_gen
'''

import torch
from torch import distributions, nn
from tqdm import tqdm 
import math

from svae.models import SVAE, Prior, Sparsenet
from svae import horseshoe

import sys
sys.path.append('/home/vg0233/PillowLab/SVAE')
from likelihood_eval import samplers 
import logging
logger = logging.getLogger(__name__)
# device = torch.device("cuda", 0)  if torch.cuda.is_available() else torch.device("cpu")

# ---------------------------------------------------------
# Utils


def safe_repeat(x, n):
    '''[2]'''
    pad = [1 for _ in range(len(x.size()) - 1)]
    return x.repeat(n, *pad)

def logmeanexp(x, dim=1):
    '''[2]'''
    max_, _ = torch.max(x, dim=dim, keepdim=True)
    return torch.log(torch.mean(torch.exp(x - max_), dim=dim)) + max_.squeeze(dim=dim)

def log_normal(x, mean, logvar):
    """
    Log-pdf for factorized Normal distributions. [2]
    """
    return -0.5 * ((math.log(2 * math.pi) + logvar).sum(1) + ((x - mean).pow(2) / torch.exp(logvar)).sum(1))

def sigmoid_schedule(num, rad=4):
    """
    From [3].
    The sigmoid schedule defined as:
          gamma_t = sigma(rad * (2t/T - 1))
          beta_t = (gamma_t - gamma_1) / (gamma_T - gamma_1),
    where sigma is the logistic sigmoid. This schedule allocates more distributions near
    the inverse temperature 0 and 1, since these are often the places where the distributon
    changes the fastest.
    """
    if num == 1:
        return torch.tensor([0.0, 1.0])
    t = torch.linspace(-rad, rad, num)
    sigm = 1. / (1. + torch.exp(-t))
    return (sigm - sigm.min()) / (sigm.max() - sigm.min())

def get_prior_dist(
        prior: Prior = Prior.LAPLACE, prior_scale: float = 1.0, device = None,
        ) -> distributions.Distribution:
    '''
    Return prior as a Distribution object for further sampling and handling. 
    '''
    prior_type = Prior(prior.item())  # type: ignore
    prior_loc = torch.tensor(0.0).to(device)  # type: ignore
    prior_scale = prior_scale.to(device)

    if prior_type == Prior.GAUSSIAN:
        prior_dist = distributions.Normal(prior_loc, prior_scale)
    elif prior_type == Prior.LAPLACE:
        prior_dist = distributions.Laplace(prior_loc, prior_scale)
    elif prior_type == Prior.CAUCHY:
        prior_dist = distributions.Cauchy(prior_loc, prior_scale)
    elif prior_type == Prior.HORSESHOE:
        prior_dist = horseshoe.Horseshoe(prior_loc, prior_scale)
    else:
        raise RuntimeError("Unknown prior type {0}".format(prior_type))
    return prior_dist

# ---------------------------------------------------------
# Main procedure

def ais(
        model, 
        loader, 
        ais_length=100, 
        n_samples=16, 
        sampler='hmc', 
        schedule_type: str = 'sigmoid',
        verbose: bool = False, 
        forward: bool = True,
        epsilon_init: float = 0.01,
        device: torch.device = torch.device("cpu"),
    ):
    '''
    Args:
        model (svae.models.SVAE): SVAE model
        loader (iterator): dataset
        ais_length (int): number of steps in AIS
        n_samples (int): number of importance weight sampes
        sampler (str, options: ['metropolis', 'hmc']): 
            sampling procedure for the transition operator.
        schedule_type (str, options: ['linear', 'sigmoid']):
            temperature procedure, i.e. the beta_t exponents in the intermediate distributions f_i
    
    Returns:
        list of shape (batch_size, n_samples) of log importance weights for each batch of data
    '''
    prior_dist = get_prior_dist(model.prior, model.prior_scale, device=device)

    def log_f_i(z, x, beta_t):
        """
        Unnormalized density for intermediate distribution f_i:
                f_i = p(z)^(1-β_t) p(x,z)^(β_t) 
                    = p(z) p(x|z)^(β_t)
        =>  log f_i = log p(z) + β_t * log p(x|z)
        """
        if isinstance(model, SVAE):
            loc, logscale = model._encode(x)
            varposterior_dist = distributions.Normal(loc, torch.exp(logscale))
            log_prior = varposterior_dist.log_prob(z).sum(axis=1)
            # log_prior = torch.clamp(varposterior_dist.log_prob(z).sum(axis=1), -1e05, 1e05)
        else:
            log_prior = prior_dist.log_prob(z).sum(axis=1)
        likelihood_dist = distributions.Independent(
                distributions.Normal(model._decode(z), model.likelihood_scale),   # type: ignore
                reinterpreted_batch_ndims=1,
            )
        log_likelihood = likelihood_dist.log_prob(x)
        # log_likelihood = torch.clamp(likelihood_dist.log_prob(x), -1e05, 1e05)
        return log_prior + beta_t * log_likelihood

    # Set temperature schedule
    if schedule_type=='sigmoid':
        betas = sigmoid_schedule(ais_length)
    else: # default linear schedule
        betas = torch.linspace(0, 1, ais_length)
    if not forward:
        betas = torch.flip(betas, dims=(0,)).contiguous()

    log_ws = []
    avg_ARs, l1_065s = [], []
    if not verbose: 
        import itertools
        temp, loader = itertools.tee(loader, 2)
        length = len(list(temp))
        if length>1:
            pbar = tqdm(total=length, desc='Evaluate ll with AIS')
    
    for i, (batch_x, batch_z) in enumerate(loader):
        # if i> int(1000/32): break # break after 1000 images.
        
        batch_x = batch_x.flatten(start_dim=1).to(device)

        B = batch_x.shape[0] * n_samples
        batch_x = batch_x.to(device)
        batch_x = safe_repeat(batch_x, n_samples)
        
        # Init AIS states
        log_w = torch.zeros(size=(B,)).to(device)                                 # w = 1
        if forward:
            if isinstance(model, SVAE):
                loc, logscale = model._encode(batch_x)                            # z ~ q(z|x)
                current_z = model._reparameterize(loc, logscale) 
            else:
                current_z = prior_dist.sample(sample_shape=(B,model.latent_dim))  # z ~ p(z)
        else:
            current_z = safe_repeat(batch_z, n_samples)

        if sampler=='hmc':
            epsilon = torch.full(size=(B,), device=device, fill_value=epsilon_init)
            accept_hist = torch.zeros(size=(B,), device=device)

        metrics = [[torch.nan, torch.nan, torch.nan, torch.nan]]
        for it, (beta1, beta2) in enumerate(zip(betas[:-1], betas[1:])):
            # Update log importance weights
            log_prev = log_f_i(current_z, batch_x, beta1)
            log_new = log_f_i(current_z, batch_x, beta2)
            log_w += (log_new - log_prev)

            # Sampler step
            if sampler=='hmc':
                # Taken from [2]
                def U(z):
                    return -log_f_i(z, batch_x, beta2)

                @torch.enable_grad()
                def grad_U(z):
                    z = z.clone().requires_grad_(True)
                    grad, = torch.autograd.grad(U(z).sum(), z)
                    max_ = B * model.latent_dim * 100.
                    grad = torch.clamp(grad, -max_, max_)
                    return grad

                def normalized_kinetic(v):
                    zeros = torch.zeros_like(v)
                    return -log_normal(v, zeros, zeros)

                # resample velocity
                current_v = torch.randn_like(current_z)
                z, v = samplers.hmc_trajectory(current_z, current_v, grad_U, epsilon)
                previous_accept_hist = accept_hist.clone()
                current_z, epsilon, accept_hist = samplers.hmc_accept_reject(
                    current_z, 
                    current_v, 
                    z, 
                    v, 
                    epsilon, 
                    accept_hist, 
                    it,
                    U=U, 
                    K=normalized_kinetic,
                )
                # # if it==ais_length-2:
                metric1 = torch.sum(accept_hist-previous_accept_hist)/accept_hist.shape[0]
                metric2 = torch.mean(accept_hist / (it+1))
                metric3_runningavg = torch.mean(torch.tensor(metrics)[-20:,0])
                metric4_eps = torch.mean(epsilon)
                # print('accept rate (AR): {:1.4f}, AR running avg: {:1.4f}, cumul avg AR: {:1.4f}, mean eps: {:1.1e}'.format(
                #     metric1,
                #     metric3_runningavg,
                #     metric2,
                #     metric4_eps
                # ))
                metrics.append([metric1.item(), metric2.item(), metric3_runningavg.item(), metric4_eps.item()])
            else:
                current_z = samplers.metropolis(
                    current_z, 
                    lambda z: log_f_i(z, batch_x, beta2), 
                    n_steps=20
                )
        log_w = logmeanexp(log_w.view(n_samples, -1).transpose(0, 1))
        if not forward:
            log_w = -log_w
        log_ws.append(log_w.detach())
        out = torch.stack(log_ws)
        metrics = torch.tensor(metrics)

        avg_AR = metric2
        l1_065 = torch.linalg.norm(metrics[1:,1]-0.65, ord=1)
        avg_ARs.append(avg_AR.detach().item())
        l1_065s.append(l1_065.detach().item())
        if verbose:
            logger.info('Batch {}, stats: {:.3f} ± {:.3f}, avg AR: {:.4f}, L1(cumulAR - 0.65): {:3.3f}'.format(
                    i, log_w.mean().cpu().item(), log_w.std().cpu().item(), avg_AR, l1_065/ais_length
                ))
        else:
            if i>0:pbar.update()

        # import matplotlib.pyplot as plt
        # fig, (ax_eps, ax_ar, ax_ll) = plt.subplots(nrows=3, 
        #     gridspec_kw={"height_ratios":[1,3,1]}, constrained_layout=True
        #     );
        # ax_ar.plot(betas, metrics[:,:-1]);
        # ax_ar.axhline(y=0.65, c='k', zorder=-1);
        # ax_ar.legend(labels=['accept rate (AR)', 'cumul avg AR', 'AR running avg'], loc=4);
        # ax_ar.set_ylim([-0.1,1.1])
        # ax_ar.set_ylabel('Acceptance rate')

        # ax_eps.plot(betas, metrics[:,-1], color='k')
        # ax_eps.set_ylabel('eps');
        # ax_eps.set_yscale('log')

        # ax_ll.boxplot(log_w, vert=False, showmeans=True)
        # plt.show();
        # raise KeyboardInterrupt
    return out, (avg_ARs, l1_065s)
