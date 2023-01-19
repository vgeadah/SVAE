import torch
from svae.models import SVAE, Prior, Sparsenet
import hydra 
import omegaconf
from typing import Callable, Iterator
import numpy as np

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
        if i>31: break
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

def load_model(model_class: str, model_path: str, device=torch.device('cpu')):
    if model_class == 'SVAE':
        model = SVAE().to(device)
        model_path = hydra.utils.to_absolute_path(model_path)
        # model_path = '/Volumes/GEADAH_3/3_Research/PillowLab/SVAE/SavedModels/SVAE/seed42/pCAUCHY_pscale-1.0_llogscale0.0_lr-2e+00_nepochs20'
        model_cfg = omegaconf.OmegaConf.load(model_path+'/.hydra/config.yaml')
        model.load_state_dict(torch.load(model_path+'/svae_final.pth', map_location=device))
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
    return model

def log_ais_metrics(logger, ais_estimate, avg_ARs, l1_065s) -> None:
    ais_estimate = torch.as_tensor(ais_estimate, dtype=float)
    avg_ARs = torch.as_tensor(avg_ARs, dtype=float)
    l1_065s = torch.as_tensor(l1_065s, dtype=float)
    
    logger.info('Log-likelihood: {:5.5f} ± {:5.5f} per batch, averaged over dataset'.format(
        ais_estimate.mean(), ais_estimate.std())
        )
    logger.info('Average acceptance rate: {:5.5f} ± {:5.5f}'.format(avg_ARs.mean(), avg_ARs.std()))
    logger.info('Average l1 error: {:5.5f} ± {:5.5f}'.format(l1_065s.mean(), l1_065s.std()))

def search_epsilon(
        AIS_func: Callable, loader: Iterator, 
        eps_init: float=0.2, eps_learningrate: float=0.5,
        atol: float=0.01, verbose: bool=True,
        AIS_func_kwargs={}, logger=None
    ) -> float:
    r'''
    Find the epsilon (step size) in the HMC routine such that the acceptance rate is close to 65%. 
    Params:
        AIS_func: callable, 
            Wrapping of the `ais` function from the `ais.py` file.
            Must have signature AIS_func(loader, hmc_epsilon, **kwargs), taking as inputs 
            a loader and a value of epsilon. Must return the same outputs as `ais`.
    '''
    l1_error, AR_error, counter = 1., 1., 0
    eps = eps_init

    while counter < 32 and not (np.abs(AR_error) < atol):
        # Compute single batch AIS estimate and error from AR=0.65
        onebatch_loader = iter([next(loader)])
        ais_estimate, (avg_ARs, l1_065s) = AIS_func(
            onebatch_loader, 
            hmc_epsilon=eps, 
            **AIS_func_kwargs
        )
        
        l1_error = (torch.tensor(l1_065s))[0].item()
        AR_error = torch.tensor(avg_ARs).mean() - 0.65
        
        if logger is not None: 
            logger.info('[{:}] AIS estimate: {:2.2f}, Avg AR: {:.4f}, HMC epsilon: {:2.3f}, L1 error: {:.4f}'.format(
                counter, ais_estimate.mean(), torch.tensor(avg_ARs).mean(), eps, l1_error
            ))

        # Update epsilon for next pass
        eps += eps_learningrate*AR_error
        counter += 1
        if (counter % 8) == 0:          # add some learning rate decay
            eps_learningrate *= 0.5
    return eps.item()