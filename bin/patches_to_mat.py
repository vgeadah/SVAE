import torch
from scipy.io import savemat
import hydra
import omegaconf

@hydra.main(config_path="../conf", config_name="config", version_base="1.2")
def main(cfg: omegaconf.OmegaConf):
    device = torch.device("cpu")
    patches = torch.load(cfg.paths.patches)
    patches = {k: v.to(device).type(torch.float64).numpy() for k,v in patches.items()}

    matlab_dir = "/Users/victorgeadah-mac-ii/Documents/Documents - Victor’s MacBook Pro/3_Research/PillowLab/SVAE/daniel's_sparsenet/"
    savemat(matlab_dir+f"IMAGES_py_seed{cfg.bin.sample_patches_vanilla.seed}.mat", patches)

if __name__=='__main__':
    main()