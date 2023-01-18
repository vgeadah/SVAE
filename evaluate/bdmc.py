import hydra
from hydra.utils import to_absolute_path
import omegaconf
import torch
import numpy as np
import random 
import time 
import os
import itertools

from torch import nn, distributions
from zmq import device

from svae.models import SVAE, Sparsenet
from svae import data
from ais import ais

from scipy.io import loadmat

def likelihood_weighting(model, loader, device=None):
    from ais import get_prior_dist
    prior_dist = get_prior_dist(model.prior, model.prior_scale)
    lls_baseline = []
    for (batch_x, _) in loader:
        batch_x = batch_x.flatten(start_dim=1).to(device)
        current_z = prior_dist.sample(sample_shape=(batch_x.shape[0],169))
        likelihood_dist = distributions.Independent(
                distributions.Normal(model._decode(current_z), model.likelihood_scale),   # type: ignore
                reinterpreted_batch_ndims=1,
            )
        log_likelihood = likelihood_dist.log_prob(batch_x)
        lls_baseline.append(log_likelihood.mean().cpu().item())
    return lls_baseline

def simulate_data(model, n_batch=30, batch_size=32, device=None):
    """
    Simulate data from the model. Sample from the joint distribution p(z)p(x|z). 
    This is equivalent to sampling from p(x)p(z|x), i.e. z is from the posterior.
    
    Bidirectional Monte Carlo only works on simulated data,
    where we could obtain exact posterior samples.
    
    Args:
        model: Model for simulation (VAE, LinearGaussian, Sparsenet). 
                Requires _decode(z) function and _prior, _prior_scale attributes 
        batch_size: batch size for simulated data
        n_batch: number of batches
        device (torch.device): device to run all computation on
    Returns:
        iterator that loops over batches of torch Tensor pair x, z
    """
    batches = []
    for _ in range(n_batch):
        # z ~ p(z)
        from ais import get_prior_dist
        prior_dist = get_prior_dist(model.prior, model.prior_scale, device=device)
        z = prior_dist.sample(sample_shape=(batch_size, model.latent_dim))

        # x ~ p(x|z)
        likelihood_dist = distributions.Independent(
                distributions.Normal(model._decode(z), model.likelihood_scale),   # type: ignore
                reinterpreted_batch_ndims=1,
            )
        x = likelihood_dist.sample()

        paired_batch = (x.reshape(batch_size,12,12).to(device), z.to(device))
        batches.append(paired_batch)
    loader = iter(batches)
    return loader

def search(model, local_ais, loader) -> None:
    '''
    The gap between forward and backward AIS, the BDMC gap, decreases with the number 
    of intermediate dists (chain length). This method searches the minimal chain length 
    such that the BDMC gap is below 1.0. 

    Args: 
        model: DecoderModel
        local_ais: AIS function with local config argument
        loader (iterator): simulated dataset to compute BDMC estimates on
    Returns: 
        chain_length: (int): min {minimal chain length such that the BDMC gap is below 1.0, 2**13}
        (UBs, LBs): (tuple, torch.Tensor())
                upper bounds and lower bounds (backward/forward AIS resp.) of the LL, for each chain length. 

    
    method is verbose. 
    '''
    start = time.time()

    bdmc_gap, cnt = torch.inf, 0
    chain_length = 1
    UBs, LBs = [], []
    while bdmc_gap > 1.0 and cnt < 13:
        chain_length *= 2
        cnt += 1

        with torch.no_grad():
            forward_loader, backward_loader, loader = itertools.tee(loader, 3)
            forward_estimate = local_ais(
                model, 
                itertools.islice(forward_loader, 5),
                cl=chain_length,
                forward=True,
            )
            backward_estimate = local_ais(
                model, 
                itertools.islice(backward_loader, 5),
                cl=chain_length,
                forward=False,
            )
            bdmc_gap = (backward_estimate - forward_estimate).mean()
            if torch.isnan(torch.tensor(bdmc_gap)):
                bdmc_gap = torch.inf
        # assert bdmc_gap > 0.0
        print('Chain-length: {}, upper bound: {:5.3f}±{:5.3f}, lower bound: {:5.3f}±{:5.3f}, BDMC gap: {:5.3f}\n'.format(
            chain_length, 
            backward_estimate.mean(), backward_estimate.std(), 
            forward_estimate.mean(), forward_estimate.std(), 
            bdmc_gap
        ))
        UBs.append(backward_estimate)
        LBs.append(forward_estimate)

    # Use established chain-length to compute LL with AIS on simulated data
    print(f'Final chain length: {chain_length}, with BDMC gap: {bdmc_gap}')
    forward_loader, backward_loader = itertools.tee(loader, 2)

    forward_estimate = local_ais(model, forward_loader)
    backward_estimate = ais(model, backward_loader, forward=False)
    
    UBs.append(backward_estimate)
    LBs.append(forward_estimate)

    print("Log-likelihood on simulated data:")
    print('Forward: {:5.5f} ± {:5.5f}'.format(forward_estimate.mean(), forward_estimate.std()))
    print('Backward: {:5.5f} ± {:5.5f}'.format(backward_estimate.mean(), backward_estimate.std()))
    
    return chain_length, (torch.stack(UBs), torch.stack(LBs))

@hydra.main(config_path="../conf", config_name="config", version_base="1.1")
def main(cfg: omegaconf.OmegaConf) -> None:
    device = (
        torch.device("cuda", cfg.train.svae.cuda_device)
        if (torch.cuda.is_available() and cfg.train.svae.use_cuda)
        else torch.device("cpu")
    )
    COMPUTE_BASELINE=False

    set_seed(cfg.bin.sample_patches_vanilla.seed)

    def local_ais(model, loader, cl=cfg.eval.evaluate_ll.chain_length, forward=True):
        estimate, _ = ais(
            model, 
            loader, 
            ais_length=cl, 
            verbose=True, 
            sampler=cfg.eval.bdmc.sampler,
            schedule_type=cfg.eval.bdmc.schedule_type,
            forward=forward,
            epsilon_init=cfg.eval.evaluate_ll.hmc_epsilon,
        )
        return estimate

    # Construct model
    model = SVAE().to(device)
    model_path = to_absolute_path(cfg.eval.bdmc.mdl_path)
    model.load_state_dict(torch.load(model_path+'/svae_final.pth'))
    # # parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    # # model.load_state_dict(torch.load(str(parent_dir)+'/'+str(cfg.tests.bdmc.mdl_path)))
    model.eval()

    # Sparsnet
    #model = Sparsenet().to(device)
    #model.load_state_dict(
    #    torch.load(cfg.paths.user_home_dir+'/svae/outputs/SavedModels/sparsenet/LAPLACE_lambda6.0e-01_N 5000_nf169.pth'))
    #model.eval()

    # BDMC on simulated data only. 
    loader = simulate_data(model, n_batch=30, batch_size=32, device=device)

    print('-'*80+'\nModel: ', model)

    final_cl, (UBs, LBs) = search(model, local_ais, loader)
    return

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    main()
