'''
train_sparsenet.py - simulates the sparse coding algorithm
# 
# Before running you must first define A and load IMAGES.
'''

from numpy import save
import torch
import pathlib
import math
from sklearn.linear_model import Ridge, Lasso
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import minimize
import os 

from svae.models import Sparsenet, Prior

# Hyperparameters

num_steps = 1000                # Number of optimization steps to take
minibatch_size = 100            # Number of images in each minibatch
patch_buffer = 4                # Size of image border to exclude from patches
num_filters = 169               # Number of filters to use
learning_rate = 5.0             # Features Learning rate
learning_rate_decay = 0.9995    # Learning rate decay factor
dict_coefs_tol = 0.001          # Optimization tolerance for sparse dictionary coefficients
dict_coefs_lambda = 0.6        # use: 0.6. Sparsity penalty for sparse dictionry coefficients
SAVE = False 

prior = 'LAPLACE'

torch.manual_seed(42)

# Load data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
patches = torch.load("./patches.pt")
train = patches['train'].to(device)

# Initialize parameters

A = torch.randn(144, num_filters)
A = torch.mul(A, (1/torch.norm(A, 2, dim=0)).repeat(144,1))
L, M = A.shape

if prior=='LAPLACE':
    alpha = dict_coefs_lambda/(2 * L)
elif prior=='GAUSSIAN':
    likelihood_scale = 1.0
    alpha = likelihood_scale**2 / (torch.pi/2) # pi/4 = argmin_{scale}( KL(Normal(0, scale) || Laplace(0,1)) )
    
#! 0.1 magic value = argmin_{scale}( KL(Cauchy(0, scale) || Laplace(0,1)) )

X=torch.zeros(L,minibatch_size)

## Run optimization
for t in tqdm(range(num_steps), desc='Optimization'):

    # extract subimage patches at random to make data vector X
    
    rand_indices = torch.randint(train.shape[0], size=(minibatch_size,))
    X = train[rand_indices,:,:].reshape(minibatch_size, L).T
    # for i in range(minibatch_size):
    #     this_patch=train[rand_indices[i],:,:]
    #     X[:,i]=this_patch.reshape(L,)
    

    # calculate coefficients for these data
    
    S=torch.zeros(M, minibatch_size)
    # reg = SGDRegressor(penalty='l2', tol=dict_coefs_tol, alpha=dict_coefs_lambda)
    if prior=='LAPLACE':
        reg = Lasso(alpha=alpha, tol=dict_coefs_tol)            # L1 penalty, or LASSO. Fit model with coordinate descent
    else:
        reg = Ridge(alpha=alpha, tol=dict_coefs_tol)            # L2 penalty, or Ridge.
    for i in range(minibatch_size):
        reg.fit(A, X[:, i])
        S[:, i] = torch.tensor(reg.coef_)
        # S[:, i] = l1_ls(A, X(:, i), dict_coefs_lambda, dict_coefs_tol, 1)
    assert torch.linalg.norm(S) != 0., 'Dictionary elements not updated. Optimization did not converge.'
    
    # calculate residual error
    
    E=X-A@S

    # update bases
    
    # dA=torch.zeros(L,M)
    # for i in range(minibatch_size):
    #     dA = dA + E[:,i].unsqueeze(1)@S[:,i].unsqueeze(0) # eq (6)
    dA = E @ S.T # eq (6)
    dA = dA/minibatch_size

    A = A + learning_rate*dA
    learning_rate = learning_rate * learning_rate_decay
    
    # normalize bases to match desired output variance
    
    A = torch.mul(A, (1/torch.norm(A, 2, dim=0)).repeat(144,1))

# Save 
if SAVE:
    if prior=='GAUSSIAN': save_prior = Prior.GAUSSIAN
    elif prior=='CAUCHY': save_prior = Prior.CAUCHY
    else: save_prior = Prior.LAPLACE

    if prior=='LAPLACE':
        model = Sparsenet(
            prior=save_prior,
            likelihood_logscale=torch.log(torch.sqrt(torch.tensor(dict_coefs_lambda / 2))).item(),
            prior_scale=1.0
            )
    elif prior=='GAUSSIAN':
        model = Sparsenet(
            prior=save_prior,
            likelihood_logscale=torch.log(torch.tensor(likelihood_scale)).item(),
            prior_scale=1.0
            )
    model._set_Phi(A)
    filename = prior+"_lambda{:1.1e}_N{:5d}_nf{:3d}.pth".format(dict_coefs_lambda, num_steps, num_filters)
    savedir_parent = '/Users/victorgeadah-mac-ii/Documents/Documents - Victor’s MacBook Pro/3_Research/PillowLab/SVAE/svae/outputs/SavedModels/sparsenet/'
    if not os.path.exists(savedir_parent):
        os.makedirs(savedir_parent)
    torch.save(model.state_dict(), savedir_parent+filename)

# Plot
fig, axs = plt.subplots(figsize=[6,6], ncols=8, nrows=8, constrained_layout=True);
for i, ax in enumerate(axs.flatten()):
    ax.imshow(A[:,i].reshape(12,12), vmin=A.min(), vmax=A.max())
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle('Learned features')
plt.show();