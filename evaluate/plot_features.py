import hydra
import omegaconf
import torch
import math

from svae.models import SVAE
from evaluate_ll import Sparsenet

from scipy.io import loadmat

# # -----------------------------------
@hydra.main(config_path="../conf", config_name="config", version_base="1.1")
def main(cfg: omegaconf.OmegaConf) -> None:
    # mdl_path = 'svae/outputs/2022-08-20/14-19-26/svae_final.pth'

    # model = SVAE() #.to(device)
    # model.load_state_dict(torch.load(cfg.paths.user_home_dir+'/'+mdl_path))
    # model.eval()

    # Sparsnet
    model = Sparsenet() #.to(device)
    model.load_state_dict(
        torch.load(cfg.paths.user_home_dir+'/svae/outputs/SavedModels/sparsenet/LAPLACE_lambda6.0e-01_N 5000_nf169.pth'))
    model.eval()

    matrix = model.Phi.weight if isinstance(model, Sparsenet) else model.fc5.weight
    for p in ['fro', 'nuc', torch.inf,-torch.inf, -1, 1, -2, 2]:
        print('ord={}, norm={:4.6f}'.format(p, torch.linalg.norm(matrix, ord=p).detach().item()))
    matrix = matrix.detach().cpu().numpy()

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(figsize=[6,6], ncols=8, nrows=8, constrained_layout=True);
    for i, ax in enumerate(axs.flatten()):
        ax.imshow(matrix[:,i].reshape(12,12), vmin=matrix.min(), vmax=matrix.max())
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle('Learned features')
    plt.show();

    # fig, ax = plt.subplots();
    # ax.imshow(matrix)
    # ax.set_xticks([])
    # ax.set_yticks([])
    # fig.suptitle('Full')
    # plt.show();

if __name__ =='__main__':
    main()