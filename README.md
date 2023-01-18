# Sparse-Coding Variational Auto-Encoders

Train and evaluate the Sparse-Coding Variational Auto-Encoder (SVAE) and the Olshausen & Field (1996, 1997) original sparse coding model (coined *sparsenet*) on naturalistic images from the Berkeley Segmentation Dataset (BDSD300). 

## Usage

First, create and activate the `svae` conda environment using the `environment.yml` file, and install the local `svae` package with `setup.py`. 

The SVAE model can be trained using the `bin/train_svae.py` file, with arguments handled through Hydra. For example:
```
python3 bin/train_svae.py \
    models.svae.prior='CAUCHY' \
    train.svae.learning_rate=1e-05 \
    train.svae.num_epochs=32
```
