import hydra
import omegaconf
import pathlib
import torch
import numpy as np
import random 
import time

import sys
sys.path.append('/home/vg0233/PillowLab/SVAE')
from svae.models import SVAE, Prior, Sparsenet
from svae import data
from ais import ais

from hydra.utils import to_absolute_path
import os

@hydra.main(config_path="../conf", config_name="config", version_base="1.2")
def main(cfg: omegaconf.OmegaConf) -> None:
    # Load model

    ## SVAE
    #model = SVAE().to(device)
    #model_path = to_absolute_path(cfg.eval.evaluate_ll.mdl_path)
    ## model_path = '/Volumes/GEADAH_3/3_Research/PillowLab/SVAE/SavedModels/SVAE/seed42/pCAUCHY_pscale-1.0_llogscale0.0_lr-2e+00_nepochs20'
    #model_cfg = omegaconf.OmegaConf.load(model_path+'/.hydra/config.yaml')
    #model.load_state_dict(torch.load(model_path+'/svae_final.pth', map_location=torch.device('cpu')))
    #model.eval()
    #set_seed(model_cfg.bin.sample_patches_vanilla.seed)

    # Sparsnet
    model = Sparsenet().to(device)
    model_path = to_absolute_path(cfg.eval.evaluate_ll.mdl_path)
    model_cfg = omegaconf.OmegaConf.load(model_path+'/.hydra/config.yaml')
    filename = model_cfg.models.sparsenet.prior+"_final_N{:}_llscale{:1.1e}_nf{:3d}_lr{:}.pth".format(
        model_cfg.train.sparsenet.num_steps,
        model_cfg.models.sparsenet.likelihood_logscale, model_cfg.models.sparsenet.num_filters,
        model_cfg.train.sparsenet.learning_rate
        )

    # filename = model_cfg.models.sparsenet.prior+"_N500-{:}_llscale{:1.1e}_nf{:3d}_lr{:}_manual_CGM.pth".format(
    #     model_cfg.train.sparsenet.num_steps,
    #     model_cfg.models.sparsenet.likelihood_logscale, model_cfg.models.sparsenet.num_filters,
    #     model_cfg.train.sparsenet.learning_rate
    #     )

    model.load_state_dict(
        torch.load(model_path+'/'+filename, map_location=device))
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
    

    # Start search procedure for epsilon
    if verbose: print('='*80)
    routine_start = time.time()
    l1_error, AR_error, counter = 1., 1., 0
    eps = 0.2
    eps_learningrate = 0.5

    eps_array, ars_array = [], []
    while counter < 32 and not (np.abs(AR_error) < 0.01): # or l1_error < 0.05
        # Compute single batch AIS estimate and error from AR=0.65
        onebatch_loader = iter([next(test_loader)])
        ais_estimate, (avg_ARs, l1_065s) = local_ais(onebatch_loader, hmc_epsilon=eps)
        
        l1_error = (torch.tensor(l1_065s)/cfg.eval.evaluate_ll.chain_length)[0].item()
        AR_error = torch.tensor(avg_ARs).mean() - 0.65
        
        if verbose: print('[{:}] AIS estimate: {:2.2f}, Avg AR: {:.4f}, HMC epsilon: {:2.3f}, L1 error: {:.4f}'.format(
            counter, ais_estimate.mean(), torch.tensor(avg_ARs).mean(), eps, l1_error
        ))
        # Update epsilon for next pass
        eps += eps_learningrate*AR_error
        counter += 1
        if (counter % 8) == 0:          # add some learning rate decay
            eps_learningrate *= 0.5

    print((eps - eps_learningrate*AR_error).item())
    if verbose:
        print('for AIS with config  :', cfg.eval.evaluate_ll)
        # print('on model with config :', model_cfg.models.svae)

        print('\nTime: {:.3f} s\n'.format(time.time()-routine_start))

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    device = torch.device("cuda", 0)  if torch.cuda.is_available() else torch.device("cpu")
    COMPUTE_BASELINE=True
    verbose=True
    main()
