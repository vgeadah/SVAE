from cgi import test
from re import X
import hydra
from importlib_metadata import itertools
import omegaconf
import pathlib
import torch
import numpy as np
import random 
import time 

from torch import nn, distributions
from zmq import device

from svae.models import Prior
from svae import data
from ais import ais


class LinearGaussian(nn.Module):
    '''
    Gaussian linear model for estimation and AIS validation.
    Generative model: z ~ N(0, σ_p^2 I)
                    x|z ~ N(Az+b, σ_l^2 I)
            -->    p(x) = N(b, σ_p^2 AA^T + σ_l^2 I)
    '''
    def __init__(
        self,
        prior: Prior = Prior.GAUSSIAN,
        prior_scale: float = 1.0,
        likelihood_logscale: float = -1.0
    ):
        super().__init__()
        self.register_buffer("prior", torch.tensor(prior.value))
        self.register_buffer("prior_scale", torch.tensor(prior_scale))
        self.register_buffer("likelihood_scale", torch.exp(torch.tensor(likelihood_logscale)))
        self.latent_dim = 169
        self.f = nn.Linear(self.latent_dim, 144)
        
    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.f(z)


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: omegaconf.OmegaConf) -> None:
    start = time.time()
    set_seed(cfg.bin.sample_patches_custom.seed)

    # Gaussian Linear Model for validation:
    model = LinearGaussian(likelihood_logscale=-1.0).to(device)
    model.eval()

    BDMC = False
    forward = True

    # Get data
    if BDMC:
        # BDMC on simulated data only. 
        batches = []
        n_batch, B = 30, cfg.bin.train_svae.batch_size
        for _ in range(n_batch):
            # z ~ p(z)
            from ais import get_prior_dist
            prior_dist = get_prior_dist(model.prior, model.prior_scale)
            z = prior_dist.sample(sample_shape=(B, model.latent_dim))

            # x ~ p(x|z)
            likelihood_dist = distributions.Independent(
                    distributions.Normal(model._decode(z), model.likelihood_scale),   # type: ignore
                    reinterpreted_batch_ndims=1,
                )
            x = likelihood_dist.sample()

            paired_batch = (x.reshape(B,12,12), z)
            batches.append(paired_batch)
        loader = iter(batches)
    else:
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
        loader = iter(batches)
    loader, loader_true = itertools.tee(loader, 2)
    if COMPUTE_BASELINE:
        loader, loader_baseline = itertools.tee(loader, 2)



    # Evaluate ll with AIS
    with torch.no_grad():
        ais_estimate = ais(
            model, 
            loader, 
            ais_length=cfg.tests.evaluate_ll.chain_length, 
            n_samples=cfg.tests.evaluate_ll.n_sample,
            verbose=True, 
            sampler=cfg.tests.evaluate_ll.sampler,
            schedule_type=cfg.tests.evaluate_ll.schedule_type,
            forward=forward,
        )
    
    print('-'*80+'\nModel: ', model)
    print('\nEvaluation: AIS procedure, with config:\n'+ omegaconf.OmegaConf.to_yaml(cfg.tests.evaluate_ll))
    print('Log-likelihood: {:5.5f} ± {:5.5f} per batch, mean/std over dataset'.format(
        torch.mean(ais_estimate), torch.std(ais_estimate)))

    if not BDMC:
        # Compare AIS estimate with true:
        A, b = model.f.weight, model.f.bias 
        marginal_dist = distributions.MultivariateNormal(
            loc=b,
            covariance_matrix= A @ (model.prior_scale**2 * torch.eye(model.latent_dim,model.latent_dim)) @ torch.transpose(A, 0, 1) \
                                + model.likelihood_scale**2 * torch.eye(144,144)
        )
        lls = []
        for (batch_x, _) in loader_true:
            batch_x = batch_x.flatten(start_dim=1).to(device)
            true_batchll = marginal_dist.log_prob(batch_x)
            lls.append(true_batchll.mean().cpu().item())
        print(' '*10+'True: {:5.5f} ± {:5.5f}'.format(np.mean(lls), np.std(lls)))

    end = time.time()
    print('\nTotal time: {:.3f}s\n'.format(end-start))

    # Baseline:
    if COMPUTE_BASELINE:
        from ais import get_prior_dist
        prior_dist = get_prior_dist(model.prior, model.prior_scale)
        lls_baseline = []
        for (batch_x, _) in loader_baseline:
            batch_x = batch_x.flatten(start_dim=1).to(device)
            current_z = prior_dist.sample(sample_shape=(batch_x.shape[0],169))
            likelihood_dist = distributions.Independent(
                    distributions.Normal(model._decode(current_z), model.likelihood_scale),   # type: ignore
                    reinterpreted_batch_ndims=1,
                )
            log_likelihood = likelihood_dist.log_prob(batch_x)
            lls_baseline.append(log_likelihood.mean().cpu().item())
        print(np.mean(lls_baseline), np.std(lls_baseline))

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    COMPUTE_BASELINE=False
    main()