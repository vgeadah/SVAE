# Sparse-Coding Variational Auto-Encoders

Accompanying code to [Sparse-Coding Variational Auto-Encoders](https://www.biorxiv.org/content/10.1101/399246v2#:~:text=The%20sparse%2Dcoding%20variational%20auto,by%20a%20deep%20neural%20network.) (SVAE). 
Train and evaluate the SVAE and the Olshausen & Field (1996, 1997) original sparse coding model (*Sparsenet* here) on naturalistic images from the Berkeley Segmentation Dataset (BDSD300). 

## Usage

First, create and activate the `svae` conda environment using the `environment.yml` file, and install the local `svae` package with `setup.py`. Finally, make sure to change the `paths.user_home_dir` in the `conf/config.yaml` file to your working directory.

### Training

The SVAE model can be trained using the `bin/train_svae.py` file, with arguments handled through [Hydra](https://hydra.cc/docs/intro/). For example:
```
python3 bin/train_svae.py \
    models.svae.prior='CAUCHY' \
    train.svae.learning_rate=1e-05 \
    train.svae.num_epochs=32
```

### Evaluation

We use Annealed Importance Sampling (AIS) to estimate the log-likelihood on held out data of all models under consideration. The appropriate step-size is determined in the script if `eval.evaluate_ll.search_epsilon` is True (default). To test a `SVAE` model saved in `relative/path/to/pretrained_model/` run 
```
python3 evaluate/evaluate_ll.py \
    eval.evaluate_ll.mdl_path="relative/path/to/pretrained_model" \
    eval.evaluate_ll.chain_length=1024
```