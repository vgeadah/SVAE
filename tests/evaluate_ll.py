from curses import KEY_UNDO
from re import A
import hydra
import omegaconf
import pathlib
import torch
import numpy as np
import random 
import time

from torch import nn, distributions
from zmq import device

from svae.models import SVAE, Prior, Sparsenet
from svae import data
from ais import ais

from scipy.io import loadmat
from hydra.utils import to_absolute_path
import os

def dataset_to_dataloader(cfg: omegaconf.OmegaConf, dataset) -> None:
    base_sampler = torch.utils.data.RandomSampler
    test_sampler = torch.utils.data.BatchSampler(  # type: ignore
        base_sampler(dataset), cfg.bin.train_svae.batch_size, drop_last=False  # type: ignore
    )
    test_loader = torch.utils.data.DataLoader(  # type: ignore
        dataset, sampler=test_sampler, collate_fn=lambda x: x[0]  # type: ignore
    )
    batches = []
    for x in test_loader:
        paired_batch = (x, torch.empty(x.shape))
        batches.append(paired_batch)
    test_loader = iter(batches)
    return test_loader

def expand_truncate_dataloader(dataloader) -> torch.utils.data.dataloader.DataLoader:
    batches = []
    for i, x in enumerate(dataloader):
        if i>32: break
        paired_batch = (x, torch.empty(x.shape))
        batches.append(paired_batch)
    return iter(batches)

@hydra.main(config_path="../conf", config_name="config", version_base="1.2")
def main(cfg: omegaconf.OmegaConf) -> None:
    # Load model

    # SVAE
    model = SVAE().to(device)
    model_path = to_absolute_path(cfg.tests.evaluate_ll.mdl_path)
    # model_path = '/Volumes/GEADAH_3/3_Research/PillowLab/SVAE/SavedModels/SVAE/seed42/pCAUCHY_pscale-1.0_llogscale0.0_lr-2e+00_nepochs20'
    model_cfg = omegaconf.OmegaConf.load(model_path+'/.hydra/config.yaml')
    model.load_state_dict(torch.load(model_path+'/svae_final.pth'))
    model.eval()
    set_seed(model_cfg.bin.sample_patches_vanilla.seed)

    # # Sparsnet
    # model = Sparsenet().to(device)
    # model.load_state_dict(
    #     torch.load(cfg.paths.user_home_dir+'/svae/outputs/SavedModels/sparsenet/LAPLACE_lambda6.0e-01_N 5000_nf169.pth'))
    # model.eval()
    # set_seed(cfg.bin.sample_patches_vanilla.seed)

    # Get data
    _, _, test_loader = data.get_dataloaders(
        pathlib.Path(cfg.paths.user_home_dir) / pathlib.Path(cfg.paths.scratch) / "patches.pt",
        batch_size=cfg.bin.train_svae.batch_size,
        shuffle=cfg.bin.train_svae.shuffle,
        device=device,
    )
    batches = []
    for i, x in enumerate(test_loader):
        if i>31: break
        paired_batch = (x, torch.empty(x.shape))
        batches.append(paired_batch)
    test_loader = iter(batches)

    if COMPUTE_BASELINE:
        import itertools
        test_loader, test_loader_baseline = itertools.tee(test_loader, 2)

    # Evaluate ll with AIS
    ais_start = time.time()
    with torch.no_grad():
        ais_estimate, (avg_ARs, l1_065s) = ais(
                model, 
                test_loader, 
                ais_length=cfg.tests.evaluate_ll.chain_length, 
                n_samples=cfg.tests.evaluate_ll.n_sample,
                verbose=True, 
                sampler=cfg.tests.evaluate_ll.sampler,
                schedule_type=cfg.tests.evaluate_ll.schedule_type,
                epsilon_init=cfg.tests.evaluate_ll.hmc_epsilon,
            )
    # Save ll estimate
    if isinstance(model, Sparsenet):
        file_name = 'sparsenet_ll'
        model_path = '.'
    else:
        file_name = 'll_estimate'
    file_suffix = f'_cl{cfg.tests.evaluate_ll.chain_length}_schedule-{cfg.tests.evaluate_ll.schedule_type}_eps{cfg.tests.evaluate_ll.hmc_epsilon}.pt'
    torch.save(ais_estimate, model_path+'/'+file_name+file_suffix)
    
    print('-'*80+'\nModel: ', model)
    print('\nEvaluation: AIS procedure, with config:\n'+ omegaconf.OmegaConf.to_yaml(cfg.tests.evaluate_ll))
    print('Log-likelihood: {:5.5f} ± {:5.5f} per batch, averaged over dataset'.format(
        ais_estimate.mean(), ais_estimate.std()))
    
    avg_ARs = torch.tensor(avg_ARs)
    l1_065s = torch.tensor(l1_065s)/cfg.tests.evaluate_ll.chain_length
    print('Average acceptance rate        : {:5.5f} ± {:5.5f}'.format(avg_ARs.mean(), avg_ARs.std()))
    print('Average l1(cumul_avg_AR - 0.65): {:5.5f} ± {:5.5f}'.format(l1_065s.mean(), l1_065s.std()))

    print(' '*10+'Time: {:.3f} s\n'.format(time.time()-ais_start))

    # Baseline:
    if COMPUTE_BASELINE:
        baseline_start = time.time()
        lls_baseline = []
        for (batch_x, _) in test_loader_baseline:
            batch_x = batch_x.flatten(start_dim=1).to(device)
            if isinstance(model, SVAE):
                #print('Sampling z ~ q(z|x)')
                loc, logscale = model._encode(batch_x)
                prior_dist = distributions.Normal(loc, torch.exp(logscale))
            else:
                from ais import get_prior_dist
                prior_dist = get_prior_dist(model.prior, model.prior_scale)
            
            current_z = prior_dist.sample(sample_shape=(batch_x.shape[0],model.latent_dim))
            likelihood_dist = distributions.Independent(
                    distributions.Normal(model._decode(current_z), model.likelihood_scale),   # type: ignore
                    reinterpreted_batch_ndims=1,
                )
            log_likelihood = likelihood_dist.log_prob(batch_x)
            lls_baseline.append(log_likelihood.mean().cpu().item())
        print('\nBaseline: Likelihood weighting')
        print('Log-likelihood: {:5.5f} ± {:5.5f}'.format(np.mean(lls_baseline), np.std(lls_baseline)))
        print(' '*10+'Time: {:.3f} s\n'.format(time.time()-baseline_start))

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    COMPUTE_BASELINE=True
    main()
