from matplotlib import scale
from numpy import linspace
import torch
from torch import distributions
import matplotlib.pyplot as plt
from tqdm import tqdm

Laplace01 = distributions.Laplace(torch.tensor([0.0]), torch.tensor([1.0]))

scale_argmins = []
logscale_space = torch.linspace(-5,2,100)
for n_samples in range(8):
    KLs = []
    for logscale in tqdm(logscale_space):
        n_kl_samples = int(10**n_samples)
        Cauchy_dist = distributions.Cauchy(torch.tensor([0.0]), torch.exp(logscale))
        cauchy_samples = Cauchy_dist.rsample((n_kl_samples,))
        kl_divergences = (
            Cauchy_dist.log_prob(cauchy_samples)
            - Laplace01.log_prob(cauchy_samples)
        ).mean(0)
        KLs.append(kl_divergences.item())
    KLs=torch.tensor(KLs)
    logscale_argmin = logscale_space[torch.argmin(KLs)]
    scale_argmin = torch.exp(logscale_argmin)
    scale_argmins.append(scale_argmin)

print(scale_argmin, scale_argmin**2, scale_argmin/torch.pi, scale_argmin**2/torch.pi)

fig, ax = plt.subplots();
ax.plot(scale_argmins);
ax.set_yscale('log');
plt.show();

fig, ax = plt.subplots();
ax.plot(torch.exp(logscale_space), KLs);
ax.set_yscale('log');
ax.set_xscale('log');
plt.show();