import hamiltorch
import torch
import matplotlib.pyplot as plt

hamiltorch.set_random_seed(123)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def log_prob_func(params):
    mean = torch.tensor([1.,2.,3.])
    stddev = torch.tensor([0.5,0.5,0.5])
    return torch.distributions.Normal(mean, stddev).log_prob(params).sum()

num_samples = 400
step_size = .3
num_steps_per_sample = 5

hamiltorch.set_random_seed(123)
params_init = torch.zeros(3)
params_hmc = hamiltorch.sample(log_prob_func=log_prob_func, params_init=params_init,  num_samples=num_samples, step_size=step_size, num_steps_per_sample=num_steps_per_sample)