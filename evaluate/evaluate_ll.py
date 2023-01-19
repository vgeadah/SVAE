from curses import KEY_UNDO
from re import A
import hydra
import omegaconf
import pathlib
import torch
import numpy as np
import random 
import time
import pandas as pd 

from torch import nn, distributions

from svae.models import SVAE, Prior, Sparsenet
from svae import data
from find_hmc_epsilon import search_epsilon
from ais import ais

from scipy.io import loadmat
from scipy.optimize import curve_fit, OptimizeWarning
import warnings
warnings.simplefilter("error", OptimizeWarning)

from hydra.utils import to_absolute_path
import os

import logging
logger = logging.getLogger(__name__)

def dataset_to_dataloader(cfg: omegaconf.OmegaConf, dataset) -> None:
    base_sampler = torch.utils.data.RandomSampler
    test_sampler = torch.utils.data.BatchSampler(  # type: ignore
        base_sampler(dataset), cfg.train.svae.batch_size, drop_last=False  # type: ignore
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

def sigmoid(x, a, b):
    threshold = 20.0
    shifted_x = -a*(b-x)
    if shifted_x < -threshold:
        return 1.0
    elif shifted_x > threshold:
        return 0.0
    else:
        return 1/(1+np.exp(-a*(b-x)))

def inv_sigmoid(y, a, b):
    return (1/a) * np.log(1/y - 1) + b

@hydra.main(config_path="../conf", config_name="config", version_base="1.2")
def main(cfg: omegaconf.OmegaConf) -> None:
    # Load model

    if cfg.eval.evaluate_ll.model == 'SVAE':
        model = SVAE().to(device)
        model_path = to_absolute_path(cfg.eval.evaluate_ll.mdl_path)
        # model_path = '/Volumes/GEADAH_3/3_Research/PillowLab/SVAE/SavedModels/SVAE/seed42/pCAUCHY_pscale-1.0_llogscale0.0_lr-2e+00_nepochs20'
        model_cfg = omegaconf.OmegaConf.load(model_path+'/.hydra/config.yaml')
        model.load_state_dict(torch.load(model_path+'/svae_final.pth', map_location=device))
        logger.info(f'Loaded: {model_path}')
    else:
        # Sparsenet
        model = Sparsenet().to(device)
        model_path = to_absolute_path(cfg.eval.evaluate_ll.mdl_path)
        model_cfg = omegaconf.OmegaConf.load(model_path+'/.hydra/config.yaml')
        filename = model_cfg.models.sparsenet.prior+"_final_N{:}_llscale{:1.1e}_nf{:3d}_lr{:}.pth".format(
            model_cfg.train.sparsenet.num_steps,
            model_cfg.models.sparsenet.likelihood_logscale, model_cfg.models.sparsenet.num_filters,
            model_cfg.train.sparsenet.learning_rate
            )

        # filename = model_cfg.models.sparsenet.prior+"_N1000-{:}_llscale{:1.1e}_nf{:3d}_lr{:}_manual_CGM.pth".format(
        #     model_cfg.train.sparsenet.num_steps,
        #     model_cfg.models.sparsenet.likelihood_logscale, model_cfg.models.sparsenet.num_filters,
        #     model_cfg.train.sparsenet.learning_rate
        #     )
        model.load_state_dict(
            torch.load(model_path+'/'+filename, map_location=device))
        logger.info(f'Loaded: {model_path}/{filename}')

    model.eval()
    set_seed(cfg.bin.sample_patches_vanilla.seed)

    # Get data
    _, _, test_loader = data.get_dataloaders(
        pathlib.Path(cfg.paths.user_home_dir) / pathlib.Path(cfg.paths.scratch) / "patches.pt",
        batch_size=cfg.train.svae.batch_size,
        shuffle=cfg.train.svae.shuffle,
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

    def local_ais(
            model,
            loader,
            hmc_epsilon: float = cfg.eval.evaluate_ll.hmc_epsilon,
            verbose: bool = False,
            ):
        with torch.no_grad():
            ais_estimate, (avg_ARs, l1_065s) = ais(
                    model, 
                    loader, 
                    ais_length=cfg.eval.evaluate_ll.chain_length, 
                    n_samples=cfg.eval.evaluate_ll.n_sample,
                    verbose=verbose, 
                    sampler=cfg.eval.evaluate_ll.sampler,
                    schedule_type=cfg.eval.evaluate_ll.schedule_type,
                    epsilon_init=hmc_epsilon,
                    device=device,
                )
        return ais_estimate, (avg_ARs, l1_065s)

    # Evaluate ll with AIS
    if cfg.eval.evaluate_ll.search_epsilon and not cfg.eval.evaluate_ll.vary_llscale:
        logger.info('Start search procedure for HMC epsilon')
        def ais_func(loader, hmc_epsilon):
            return local_ais(model=model, loader=loader, hmc_epsilon=hmc_epsilon)
        epsilon = search_epsilon(AIS_func=ais_func, loader=test_loader)
    else:
        epsilon = cfg.eval.evaluate_ll.hmc_epsilon
    # USE_SEARCH_EPS = False 
    # if USE_SEARCH_EPS:
    #     df = pd.read_csv(cfg.paths.user_home_dir+'/calculations/hmc_epsilon.csv')
    #     epsilon = df.query(f"Model_path=='{cfg.eval.evaluate_ll.mdl_path}'")['HMC_epsilon'].values[0]
    # else:
    #     epsilon = 
    ais_start = time.time()


    if cfg.eval.evaluate_ll.vary_llscale:
        eps = epsilon
        # epsilon_init_array = [0.2, 0.38, 0.65, 1.0395, 1.0869, 1.0875, 1.09, 1.09, 1.09, 1.09]
        # epsilon_init_array = [0.2, 0.5, 1.0395, 1.09, 1.09] # CAUCHY SVAE
        # epsilon_init_array = [0.0120, 0.03, 0.0884, 0.4319, 0.4973] # CAUCHY SC
        test_dataset = list(test_loader)
        for index, sigma in enumerate(np.exp([-4., -2., 0., 2.])):
            logger.info('='*80)
            # eps = epsilon_init# epsilon_init_array[index]
            model.likelihood_scale = torch.tensor([sigma]).to(device)
            logger.info('Likelihood scale: {:2.5f}, logscale: {:1.2f}'.format(
                model.likelihood_scale.item(), torch.log(model.likelihood_scale).item()
                ))

            # Determine HMC epsilon for noise scale

            logger.info('Start search procedure for HMC epsilon')
            routine_start = time.time()
            # l1_error, AR_error, counter = 1., 1., 0
            # # eps = 0.2
            # eps_learningrate = 0.02

            # eps_array, ars_array = [], []
            # while counter < 32 and not (np.abs(AR_error) < 0.01): # or l1_error < 0.05
            #     # Compute single batch AIS estimate and error from AR=0.65
            #     onebatch_loader = iter([test_dataset[0]])
            #     ais_estimate, (avg_ARs, l1_065s) = local_ais(model, onebatch_loader, hmc_epsilon=eps)
                
            #     l1_error = (torch.tensor(l1_065s)/cfg.eval.evaluate_ll.chain_length)[0].item()
            #     AR_error = torch.tensor(avg_ARs).mean() - 0.65
                
            #     logger.info('[{:}] AIS estimate: {:2.2f}, Avg AR: {:.4f}, HMC epsilon: {:2.3f}, L1 error: {:.4f}'.format(
            #         counter, ais_estimate.mean(), torch.tensor(avg_ARs).mean(), eps, l1_error
            #     ))

            #     # Update epsilon for next pass
            #     eps += eps_learningrate*AR_error
            #     counter += 1

            #     eps_array.append(eps)
            #     ars_array.append(torch.tensor(avg_ARs).mean().item())

            #     # if counter > 9:
            #     #     '''for potential speed up, interpolate AR(hmc_eps) as a sigmoid'''
            #     #     try:
            #     #         popt, pcov = curve_fit(sigmoid, eps_array, ars_array)
            #     #         eps = inv_sigmoid(0.65, *popt)
            #     #     except OptimizeWarning:
            #     #         continue

            #     if (counter % 8) == 0:          # add some learning rate decay
            #         eps_learningrate *= 0.5

            eps = search_epsilon(AIS_func=local_ais, loader=test_loader, epsilon_init=eps)
            logger.info(f'HMC with eps: {eps}')

            # Evaluate AIS at noise scale with determined HMC epsilon

            ais_estimate, (avg_ARs, l1_065s) = local_ais(
                model=model, 
                loader=iter(test_dataset), 
                hmc_epsilon=eps,
                verbose=True
                )

            logger.info('Log-likelihood: {:5.5f} ± {:5.5f} per batch, averaged over dataset'.format(
                ais_estimate.mean(), ais_estimate.std())
                )
            avg_ARs = torch.tensor(avg_ARs)
            l1_065s = torch.tensor(l1_065s)/cfg.eval.evaluate_ll.chain_length
            logger.info('Average acceptance rate        : {:5.5f} ± {:5.5f}'.format(avg_ARs.mean(), avg_ARs.std()))
            logger.info('Average l1(cumul_avg_AR - 0.65): {:5.5f} ± {:5.5f}'.format(l1_065s.mean(), l1_065s.std()))

    else:
        logger.info(f'HMC with eps: {epsilon}')
        
        ais_estimate, (avg_ARs, l1_065s) = local_ais(
                model=model, 
                loader=test_loader, 
                hmc_epsilon=epsilon,
                verbose=True
                )

    logger.info('-'*80)
    logger.info('AIS estimate:')
    logger.info('\tLog-likelihood: {:5.5f} ± {:5.5f} per batch, averaged over dataset'.format(
        ais_estimate.mean(), ais_estimate.std()))
    avg_ARs = torch.tensor(avg_ARs)
    l1_065s = torch.tensor(l1_065s)/cfg.eval.evaluate_ll.chain_length
    logger.info('\tAverage acceptance rate: {:5.5f} ± {:5.5f}'.format(avg_ARs.mean(), avg_ARs.std()))
    logger.info('\tAverage l1 error: {:5.5f} ± {:5.5f}'.format(l1_065s.mean(), l1_065s.std()))
    logger.info('\tTime: {:.3f} s'.format(time.time()-ais_start))
    logger.info('')

    # Save ll estimate
    if isinstance(model, Sparsenet):
        file_name = 'sparsenet_ll'
        # model_path = '.'
    else:
        file_name = 'll_estimate'
    file_suffix = f'_cl{cfg.eval.evaluate_ll.chain_length}_schedule-{cfg.eval.evaluate_ll.schedule_type}_eps{epsilon:.4f}.pt'
    logger.info('Saving AIS estimate to:'+str(model_path+'/'+file_name+file_suffix))
    torch.save(ais_estimate, model_path+'/'+file_name+file_suffix)
    
    logger.info('Further evaluation details:')
    logger.info('Model: '+cfg.eval.evaluate_ll.model)
    logger.info('Evaluation: AIS procedure, with config:')
    for key, val in cfg.eval.evaluate_ll.items():
        logger.info('\t'+key+' : '+str(val))
    
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
                prior_dist = get_prior_dist(model.prior, model.prior_scale, device=device)
            
            current_z = prior_dist.sample(sample_shape=(batch_x.shape[0],model.latent_dim))
            likelihood_dist = distributions.Independent(
                    distributions.Normal(model._decode(current_z), model.likelihood_scale),   # type: ignore
                    reinterpreted_batch_ndims=1,
                )
            log_likelihood = likelihood_dist.log_prob(batch_x)
            lls_baseline.append(log_likelihood.mean().cpu().item())
        logger.info('Baseline: Likelihood weighting:')
        logger.info('\tLog-likelihood: {:5.5f} ± {:5.5f}'.format(np.mean(lls_baseline), np.std(lls_baseline)))
        logger.info('\tTime: {:.3f} s'.format(time.time()-baseline_start))

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    COMPUTE_BASELINE=True
    main()
