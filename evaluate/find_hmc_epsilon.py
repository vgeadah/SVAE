import hydra
import omegaconf
import pathlib
import torch
import numpy as np
import random 
import time

from svae.models import SVAE, Prior, Sparsenet
from svae import data
from ais import ais
import eval_utils

from hydra.utils import to_absolute_path
import os

from typing import Callable, Iterator
import logging
logger = logging.getLogger(__name__)

@hydra.main(config_path="../conf", config_name="config", version_base="1.2")
def main(cfg: omegaconf.OmegaConf) -> None:

    # Load model
    model = eval_utils.load_model(
        model_class=cfg.eval.evaluate_ll.model,
        model_path=cfg.eval.evaluate_ll.mdl_path,
        device=device
    )
    model_path = hydra.utils.to_absolute_path(cfg.eval.evaluate_ll.mdl_path)
    logger.info(f'Loaded: {model_path}')

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

    # Define AIS procedure callable
    def local_ais(
            loader,
            hmc_epsilon: float = cfg.eval.evaluate_ll.hmc_epsilon
            ):
        with torch.no_grad():
            ais_estimate, (avg_ARs, l1_065s) = ais(
                    model, 
                    loader, 
                    ais_length=cfg.eval.evaluate_ll.chain_length, 
                    n_samples=cfg.eval.evaluate_ll.n_sample,
                    verbose=False, 
                    sampler=cfg.eval.evaluate_ll.sampler,
                    schedule_type=cfg.eval.evaluate_ll.schedule_type,
                    epsilon_init=hmc_epsilon,
                    device=device,
                )
        return ais_estimate, (avg_ARs, l1_065s)

    # Estimate epsilon with search procedure
    if verbose:
        logger.info('Start search procedure for epsilon.')
    routine_start = time.time()
    
    eps = eval_utils.search_epsilon(AIS_func=local_ais, loader=test_loader, logger=logger)

    # Output
    if verbose:
        logger.info('Epsilon:  {:2.8f}'.format(eps))
        logger.info('for AIS with config:')
        for key, val in  cfg.eval.evaluate_ll.items():
            logger.info('\t'+key+' : '+str(val))

        logger.info('Time: {:.3f} s'.format(time.time()-routine_start))
    else:
        print(eps)

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    device = torch.device("cuda", 0)  if torch.cuda.is_available() else torch.device("cpu")
    verbose=True
    main()
