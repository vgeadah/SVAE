'''
train_sparsenet.py - simulates the sparse coding algorithm
'''

from re import X
from turtle import xcor
from optax import lamb
import sklearn
import torch
from sklearn.linear_model import Ridge, Lasso
from sklearn.base import BaseEstimator
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import minimize
from scipy.stats import cauchy, iqr
import os 
import omegaconf
import hydra
import pathlib
import logging
import numpy as np
import time 
import sys
sys.path.append('/home/vg0233/PillowLab/SVAE')
from svae.models import Sparsenet, Prior, CauchyRegressor

# Set logger for script
logger = logging.getLogger(__name__)

# def f(local_reg, A, x):
#     local_reg.fit(A, x)
#     return local_reg.coef_

def learning_rate_schedule(optimization_step, initial_lr=5.0):
    '''From Olhaussen & Field 1997'''
    if optimization_step < 600:
        return initial_lr
    elif 600 <= optimization_step < 1200:
        return initial_lr/2
    else:
        return initial_lr/5

@hydra.main(config_path="../conf", config_name="config", version_base="1.2")
def main(cfg: omegaconf.OmegaConf) -> None:
    torch.manual_seed(cfg.train.seed)

    # Load data
    
    logger.info("Loading data")
    device = torch.device('cpu') #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("Using device: %s", device)
    
    patches = torch.load(pathlib.Path(cfg.paths.user_home_dir) / pathlib.Path(cfg.paths.scratch) / "patches.pt")
    train = patches['train'].to(device).reshape(-1, 144)    # size `(N_images, image_size)`

    avg_image_var = torch.var(train, dim=1, unbiased=False).mean().item()
    logger.info(f'Average image variance: {avg_image_var:1.5f}')
    logger.info(f'Recommended regularization lambda: {0.1* avg_image_var:1.5f}')

    # Discard images below 10% of average variance
    var_per_image = torch.var(train, dim=1, unbiased=False)
    train = train[var_per_image > 0.1 * avg_image_var]
    logger.info(f'Discarded low variance images.')

    # Initialize parameters
    
    likelihood_scale = torch.exp(torch.tensor(
            cfg.models.sparsenet.likelihood_logscale
            )).item()
    learning_rate = cfg.train.sparsenet.learning_rate

    A = torch.randn(144, cfg.models.sparsenet.num_filters).to(device)
    A = torch.mul(A, (1/torch.norm(A, 2, dim=0)).repeat(144,1))
    L, M = A.shape
    X = torch.zeros(L, cfg.train.sparsenet.minibatch_size)

    # Initialize regressor

    if cfg.models.sparsenet.prior=='GAUSSIAN':
        prior_scale = 1.0               # Note: pi/4 = argmin_{scale}( KL(Normal(0, scale) || Laplace(0,1)) )
        alpha = (likelihood_scale**2) / (prior_scale**2)
        reg = Ridge(alpha=alpha, tol=cfg.train.sparsenet.dict_coefs_tol)    # L2 penalty, or Ridge.
    
    elif cfg.models.sparsenet.prior=='CAUCHY':
        prior_scale = 1.0               # Note: 0.1 = argmin_{scale}( KL(Cauchy(0, scale) || Laplace(0,1)) )
        alpha = 2*(likelihood_scale**2) / prior_scale
        reg = CauchyRegressor(
            prior_scale=prior_scale, 
            likelihood_scale=likelihood_scale,
            tol=cfg.train.sparsenet.dict_coefs_tol
            )
    
    else:
        if cfg.models.sparsenet.prior != 'LAPLACE':
            logger.warning("Unrecognized prior. Reverting to LAPLACE")
        
        prior_scale = 1.0
        # likelihood_scale = torch.sqrt(torch.tensor(dict_coefs_lambda / 2))
        lasso_lambda = 2*(likelihood_scale**2) /prior_scale
        alpha = lasso_lambda/(2 * L)
        reg = Lasso(alpha=alpha, tol=cfg.train.sparsenet.dict_coefs_tol)    # L1 penalty, or LASSO.

    if np.linalg.norm(alpha - 0.1* avg_image_var)/np.linalg.norm(alpha) > 0.01:
        logger.warning(f'Using lambda: {alpha:1.5f}')


    # Run optimization

    logger.info("Starting optimization")
    for t in range(cfg.train.sparsenet.num_steps):

        # extract subimage patches at random to make data vector X
        
        rand_indices = torch.randint(train.shape[0], size=(cfg.train.sparsenet.minibatch_size,))
        X = train[rand_indices,:].reshape(cfg.train.sparsenet.minibatch_size, L).T
        assert (torch.var(X, dim=0, unbiased=False) > 0.1 * avg_image_var).all()

        # calculate coefficients for these data
        
        S=torch.zeros(M, cfg.train.sparsenet.minibatch_size).to(device)
        local_reg = sklearn.base.clone(reg)
        
        for i in range(cfg.train.sparsenet.minibatch_size):
            if cfg.models.sparsenet.prior=='CAUCHY':
                inference_method = 'manual_CGM'
                local_reg.fit(
                    Phi=A.cpu().numpy(), 
                    x=X[:, i].cpu().numpy(), 
                    method=inference_method,
                    optimization_step=t,
                    )
            else:
                local_reg.fit(A.cpu().numpy(), X[:, i].cpu().numpy())
            S[:, i] = torch.tensor(local_reg.coef_)
        
        assert torch.linalg.norm(S) != 0., 'Dictionary elements not updated. Optimization did not converge.'
        
        # calculate residual error
        E=X-A@S        

        # update bases
        
        dA = E @ S.T # eq (6)
        dA = dA/cfg.train.sparsenet.minibatch_size

        learning_rate = learning_rate_schedule(t, initial_lr=cfg.train.sparsenet.learning_rate)
        A = A + learning_rate*dA
        # learning_rate = learning_rate * cfg.train.sparsenet.learning_rate_decay
        
        # normalize bases to match desired output variance

        gain_adjustment = 0.01              # (Olhaussen and Field, 1997)
        var_goal = avg_image_var
        basis_functions_norm = torch.norm(A, 2, dim=0)
        avg_coefficient_norm = torch.square(S).mean(dim=1)
        assert basis_functions_norm.shape == avg_coefficient_norm.shape

        # Normalize bases associated with used (!=0) coefficients
        new_basis_functions_norm = torch.mul(
            basis_functions_norm,
            torch.where(
                avg_coefficient_norm == 0., 
                torch.ones_like(avg_coefficient_norm),
                torch.pow(avg_coefficient_norm / var_goal, gain_adjustment)
                )
            )

        # # Use IQR
        # IQR_z = iqr(S, axis=1)
        # if cfg.models.sparsenet.prior=='GAUSSIAN': IQR_prior = 1.348*prior_scale
        # elif cfg.models.sparsenet.prior=='CAUCHY': IQR_prior = 2 * prior_scale
        # else: IQR_prior = 2 * prior_scale *np.log(2) # LAPLACE

        A = torch.mul(A, (1/new_basis_functions_norm).repeat(144,1))
        # A = torch.mul(A, (1/torch.norm(A, 2, dim=0)).repeat(144,1))
        # A = torch.mul(A, torch.pow(torch.tensor(IQR_z/IQR_prior, dtype=torch.float), 0.05))

        # Optimization step finished. Log step information.

        if t%50==0: 
            logger.info("Step: {:}, lr: {:2.5f}, Residual Error: {:5.5f}, Features update: {:3.5f}".format(
                t, learning_rate, E.mean().abs(), torch.norm(dA)
            ))

        # Every `save_frequency` steps, log and save more information.

        if SAVE and (t%cfg.train.sparsenet.save_frequency == 0 or t==cfg.train.sparsenet.num_steps-1):
            if cfg.models.sparsenet.prior=='GAUSSIAN': save_prior = Prior.GAUSSIAN
            elif cfg.models.sparsenet.prior=='CAUCHY': save_prior = Prior.CAUCHY
            else: save_prior = Prior.LAPLACE

            # Cast model as Sparsenet for saving
            
            model = Sparsenet(
                prior=save_prior,
                likelihood_logscale=cfg.models.sparsenet.likelihood_logscale,
                prior_scale=1.0
                )
            model._set_Phi(A)

            # Save model 

            if t==cfg.train.sparsenet.num_steps-1:
                step_label = f'_OFnorm_final_N{cfg.train.sparsenet.num_steps}'
            else:
                step_label = f'_N{t}-{cfg.train.sparsenet.num_steps}'
            filename = cfg.models.sparsenet.prior+step_label+\
                "_llscale{:1.1e}_nf{:3d}_lr{:}".format(
                    cfg.models.sparsenet.likelihood_logscale, 
                    cfg.models.sparsenet.num_filters, 
                    cfg.train.sparsenet.learning_rate
                    )
            # if cfg.models.sparsenet.prior=='CAUCHY':
            #     filename += f'_{inference_method}'

            savedir_parent = ''
            logger.info('Saving model: '+savedir_parent+filename)
            torch.save(model.state_dict(), savedir_parent+filename+'.pth')

            # # Quick evaluation of likelihood with AIS
            # from likelihood_eval.ais import ais
            # start_time = time.time()
            # logger.info('Evaluate model evidence using AIS. one batch, cl: {}, schedule: {}'.format(
            #     cfg.tests.evaluate_ll.chain_length, cfg.tests.evaluate_ll.schedule_type,
            # ))
            # hmc_epsilon = 0.1
            # ais_estimate, (avg_ARs, l1_065s) = ais(
            #         model, 
            #         loader=iter([[X.T, torch.empty(X.T.shape)]]), 
            #         ais_length=cfg.tests.evaluate_ll.chain_length, 
            #         n_samples=cfg.tests.evaluate_ll.n_sample,
            #         verbose=False, 
            #         sampler=cfg.tests.evaluate_ll.sampler,
            #         schedule_type=cfg.tests.evaluate_ll.schedule_type,
            #         epsilon_init=hmc_epsilon,
            #         device=device,
            #     )

            # l1_error = (torch.tensor(l1_065s)/cfg.tests.evaluate_ll.chain_length)[0].item()
            
            # logger.info('Step: {:}, AIS estimate: {:2.2f}, Avg AR: {:.4f}, HMC eps: {:2.3f}, L1 error: {:.4f}, eval time: {:2.2f}s'.format(
            #     t, ais_estimate.mean(), torch.tensor(avg_ARs).mean(), hmc_epsilon, l1_error, time.time()-start_time
            # ))

    # Plot
    if PLOT:
        fig, axs = plt.subplots(figsize=[6,6], ncols=8, nrows=8, constrained_layout=True);
        for i, ax in enumerate(axs.flatten()):
            ax.imshow(A[:,i].reshape(12,12), vmin=A.min(), vmax=A.max())
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle('Learned features')
        plt.show();

if __name__=='__main__':
    SAVE = True
    PLOT = False
    main()
