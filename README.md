# Sparse-Coding Variational Auto-Encoders

Train and evaluate the Sparse-Coding Variational Auto-Encoder (SVAE) and the Olshausen & Field (1996, 1997) original sparse coding model (coined *sparsenet*) on naturalistic images from the Berkeley Segmentation Dataset (BDSD300). 

## Usage

First, create and activate the `svae` conda environment using the `environment.yml` file, and install the local `svae` package with `setup.py`. 

### Training

The SVAE model can be trained using the `bin/train_svae.py` file, with arguments handled through Hydra. For example:
```
python3 bin/train_svae.py \
    models.svae.prior='CAUCHY' \
    train.svae.learning_rate=1e-05 \
    train.svae.num_epochs=32
```

### Evaluation

We use Annealed Importance Sampling (AIS) to estimate the log-likelihood on held out data of all models under consideration. We use Hamiltionian Monte Carlo for sampling, which requires a step-size dependent on the model parameters. First run `evaluate/find_hmc_epsilon.py` to determine this step size, and use it to evaluate the log-likelihood using `evaluate/evaluate_ll.py`. As a shell script:
```
path="path/to/pretrained_model/"
cl=1024

hmcepsilon=$(python3 evaluate/find_hmc_epsilon.py eval.evaluate_ll.chain_length=$cl eval.evaluate_ll.mdl_path=$path)
python3 evaluate/evaluate_ll.py \
    eval.evaluate_ll.mdl_path=$path \
    eval.evaluate_ll.chain_length=$cl \
    eval.evaluate_ll.hmc_epsilon=$hmcepsilon
```