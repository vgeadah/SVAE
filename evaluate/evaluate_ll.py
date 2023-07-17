import pathlib
import torch
import numpy as np
import random 
import time
import hydra
import omegaconf
from torch import nn, distributions
import re


# Import local
import sys
sys.path.append('/home/vg0233/PillowLab/SVAE')
from svae.models import SVAE, Prior, Sparsenet
from svae import data
from ais import ais, logmeanexp
import eval_utils

import logging
logger = logging.getLogger(__name__)

@hydra.main(config_path="../conf", config_name="config", version_base="1.2")
def main(cfg: omegaconf.OmegaConf) -> None:

    # Load model
    # model = eval_utils.load_model(
    #     model_class=cfg.eval.evaluate_ll.model,
    #     model_path=cfg.eval.evaluate_ll.mdl_path,
    #     device=device
    # )
    # model_path = hydra.utils.to_absolute_path(cfg.eval.evaluate_ll.mdl_path)

    if cfg.eval.evaluate_ll.model == 'SVAE':
        model = SVAE().to(device)
        models_path = '/home/vg0233/PillowLab/SVAE/outputs/SavedModels/SVAE/model_files/'
        model.load_state_dict(torch.load(models_path+cfg.eval.evaluate_ll.mdl_path, map_location=device))
    else:
        model = Sparsenet().to(device)
        models_path = '/home/vg0233/PillowLab/SVAE/outputs/SavedModels/SC/model_files/'
        model.load_state_dict(torch.load(models_path+cfg.eval.evaluate_ll.mdl_path, map_location=device))
    logger.info(f'Loaded: {cfg.eval.evaluate_ll.mdl_path}')

    model.eval()
    set_seed(cfg.bin.sample_patches_vanilla.seed)

    # Get data
    _, _, test_loader = data.get_dataloaders(
        pathlib.Path(cfg.paths.user_home_dir) / pathlib.Path(cfg.paths.scratch) / "patches.pt",
        batch_size=cfg.train.svae.batch_size,
        shuffle=cfg.train.svae.shuffle,
        device=device,
    )
    test_loader = eval_utils.expand_truncate_dataloader(test_loader)
    # test_loader = eval_utils.expand_truncate_dataloader(train_loader)
    # logger.info('Using TRAIN data')
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
                    use_posterior=use_posterior
                )
        return ais_estimate, (avg_ARs, l1_065s)

    # Set epsilon start value
    if cfg.eval.evaluate_ll.search_epsilon and not cfg.eval.evaluate_ll.vary_llscale:
        logger.info('Start search procedure for HMC epsilon')
        
        def ais_func(loader, hmc_epsilon):
            return local_ais(model=model, loader=loader, hmc_epsilon=hmc_epsilon)
        epsilon = eval_utils.search_epsilon(
            AIS_func=ais_func, loader=test_loader, logger=logger,
            eps_learningrate=0.5 if cfg.eval.evaluate_ll.model == 'SVAE' else 0.2
            )
    else:
        epsilon = cfg.eval.evaluate_ll.hmc_epsilon
    ais_start = time.time()

    # Evaluate ll with AIS
    if cfg.eval.evaluate_ll.vary_llscale:
        eps = epsilon
        # epsilon_init_array = [0.2, 0.38, 0.65, 1.0395, 1.0869, 1.0875, 1.09, 1.09, 1.09, 1.09]
        # epsilon_init_array = [0.2, 0.5, 1.0395, 1.09, 1.09] # CAUCHY SVAE
        # epsilon_init_array = [0.0120, 0.03, 0.0884, 0.4319, 0.4973] # CAUCHY SC
        test_dataset = list(test_loader)
        for index, sigma in enumerate(np.exp([-4., -2., 0., 2.])):
            logger.info('='*80)

            # New value of likelihood_scale of model
            model.likelihood_scale = torch.tensor([sigma]).to(device)
            def ais_func(loader, hmc_epsilon):
                return local_ais(model=model, loader=loader, hmc_epsilon=hmc_epsilon)

            logger.info('Likelihood scale: {:2.5f}, logscale: {:1.2f}'.format(
                model.likelihood_scale.item(), torch.log(model.likelihood_scale).item()
                ))

            # Determine HMC epsilon
            logger.info('Start search procedure for HMC epsilon')

            eps = eval_utils.search_epsilon(
                AIS_func=ais_func, loader=iter(test_dataset), eps_init=eps, logger=None,
                eps_learningrate=0.5 if cfg.eval.evaluate_ll.model == 'SVAE' else 0.1
                )

            logger.info(f'HMC with eps: {eps}')

            # Evaluate AIS at noise scale with determined HMC epsilon
            ais_estimate, (avg_ARs, l1_065s) = local_ais(
                model=model, 
                loader=iter(test_dataset), 
                hmc_epsilon=eps,
                verbose=False
                )
            eval_utils.log_ais_metrics(logger, ais_estimate, avg_ARs, l1_065s)

            # Format dictionary entry
            ais_estimates = torch.as_tensor(ais_estimate, dtype=float)
            # ais_estimate = logmeanexp(ais_estimates.flatten(), dim=0).cpu().item()
            ais_estimate = ais_estimates.mean().cpu().item()
            avg_AR = torch.as_tensor(avg_ARs, dtype=float).mean()
            l1_065 = torch.as_tensor(l1_065s, dtype=float).mean()

            if cfg.eval.evaluate_ll.model == 'SVAE':
                # Regular expression pattern
                pattern = r"svae_(\w+)_ll([\+\-]?\d+(?:\.\d+)?(?:[eE][\+\-]?\d+)?)_e(\d+)_lr([\+\-]?\d+(?:\.\d+)?(?:[eE][\+\-]?\d+)?)\.pth"

                # Extract values using regular expression
                match = re.match(pattern, cfg.eval.evaluate_ll.mdl_path)
                if match:
                    model_dict = {
                        'model':"SVAE",
                        'prior': match.group(1), 'train_liklogscale': float(match.group(2)), 
                        'epoch': int(match.group(3)), 'lr': float(match.group(4))
                        }
                else:
                    model_dict = {'model': cfg.eval.evaluate_ll.mdl_path}
            else:
                # Regular expression pattern
                pattern = r"(\w+)_final_N5000_llscale([\+\-]?\d+(?:\.\d+)?(?:[eE][\+\-]?\d+)?)_nf(\d+)_lr(\d+(?:\.\d+)?)"

                # Extract values using regular expression
                match = re.match(pattern, cfg.eval.evaluate_ll.mdl_path)
                if match:
                    model_dict = {
                        'model':"SC",
                        'prior': match.group(1), 'train_liklogscale': float(match.group(2)), 
                        'epoch': 5000, 'lr': float(match.group(4))
                        }
                else:
                    model_dict = {'model': cfg.eval.evaluate_ll.mdl_path}

            logging.info(model_dict | {
                'chain_length': cfg.eval.evaluate_ll.chain_length,
                'LL':ais_estimate, 'AR': avg_AR.item(), 'l1_error': l1_065.item(), 
                'eval_liklogscale': torch.log(model.likelihood_scale).item(),
                'HMC_epsilon': eps
                })

    else:
        logger.info(f'HMC with eps: {epsilon}')
        
        ais_estimate, (avg_ARs, l1_065s) = local_ais(
                model=model, 
                loader=test_loader, 
                hmc_epsilon=epsilon,
                verbose=False
                )

        logger.info('-'*80)
        logger.info('AIS estimate:')

        eval_utils.log_ais_metrics(logger, ais_estimate, avg_ARs, l1_065s)

        logger.info('\tTime: {:.3f} s'.format(time.time()-ais_start))
        logger.info('')

        # Save ll estimate
        if isinstance(model, Sparsenet):
            file_name = 'sparsenet_ll'
            # model_path = '.'
        else:
            file_name = 'll_estimate'
        file_suffix = f'_cl{cfg.eval.evaluate_ll.chain_length}_schedule-{cfg.eval.evaluate_ll.schedule_type}_eps{epsilon:.4f}.pt'

        try:
            logger.info('Saving AIS estimate to:'+str(model_path+'/'+file_name+file_suffix))
            torch.save(ais_estimate, model_path+'/'+file_name+file_suffix)
        except NameError:
            savepath = '/home/vg0233/PillowLab/SVAE/calculations/LL/SVAE/'
            logger.info('Saving AIS estimate to:'+savepath)
            torch.save(ais_estimate, savepath + cfg.eval.evaluate_ll.mdl_path[:-4] + file_name + file_suffix)
        
        
        logger.info('Further evaluation details:')
        logger.info('Model: '+cfg.eval.evaluate_ll.model)
        logger.info('Evaluation: AIS' + ('-post' if use_posterior else '-prior') + ' procedure, with config:')
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
            # lls_baseline.append(log_likelihood.mean().cpu().item())
            lls_baseline.append(logmeanexp(log_likelihood.flatten(), dim=0).cpu().item())
        logger.info('Baseline: Likelihood weighting:')
        # logger.info('\tLog-likelihood: {:5.5f} ± {:5.5f}'.format(np.mean(lls_baseline), np.std(lls_baseline)))
        logger.info('\tLog-likelihood: {:5.5f}'.format(logmeanexp(lls_baseline, dim=0)))
        logger.info('\tTime: {:.3f} s'.format(time.time()-baseline_start))

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    COMPUTE_BASELINE=True
    use_posterior=False
    main()
