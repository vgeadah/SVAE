#!/usr/bin/env python3
"""Tests training of the SVAE."""
import logging
import math
import pathlib
import subprocess
import random
import os

import hydra
import ignite
import omegaconf
import torch
import torch.optim as optim
from torch.utils import tensorboard

from svae import data, models
import sample_patches_vanilla as sample_patches

logger = logging.getLogger(__name__)
logging.getLogger("ignite.engine.engine.Engine").setLevel(logging.WARNING)


def output_weights_to_image(W: torch.Tensor) -> torch.Tensor:
    """Convert the output weight matrix to an NCHW image batch."""
    W_min = torch.min(W, 0)[0]
    W_range = torch.max(W, 0)[0] - W_min
    W_normed = (W - W_min[None, :]) / W_range[None, :]
    return W_normed.t().view(-1, 12, 12).unsqueeze(1).expand(-1, 3, -1, -1)


def log_train_metrics(
    trainer: ignite.engine.Engine,
    model: models.SVAE,
    cfg: omegaconf.OmegaConf,
    writer: tensorboard.writer.SummaryWriter,
) -> None:
    """Log metrics after each epoch."""
    if trainer.state.epoch % cfg.train.svae.log_every != 0:
        return
    avg_neg_elbo = trainer.state.metrics["loss"]
    avg_nll = trainer.state.metrics["nll"]
    avg_kld = trainer.state.metrics["kld"]

    logger.info(
        "Epoch %d, train - Avg Negative ELBO %.2f, Avg NLL %.2f, Avg KLD %.2f",
        trainer.state.epoch,
        avg_neg_elbo,
        avg_nll,
        avg_kld,
    )
    writer.add_scalar("train/loss", avg_neg_elbo, trainer.state.epoch)
    writer.add_scalar("train/nll", avg_nll, trainer.state.epoch)
    writer.add_scalar("train/kld", avg_kld, trainer.state.epoch)

    img_batch = output_weights_to_image(model.output_weights())
    writer.add_images("train/W", img_batch, trainer.state.epoch)


def compute_and_print_eval_metrics(
    trainer: ignite.engine.Engine,
    evaluator: ignite.engine.Engine,
    dataloader: torch.utils.data.DataLoader,
    cfg: omegaconf.OmegaConf,
    writer: tensorboard.writer.SummaryWriter,
) -> None:
    """Compute and prints metrics on the validation set."""
    if trainer.state.epoch % cfg.train.svae.log_every != 0:
        return
    evaluator.run(dataloader, max_epochs=1)
    metrics = evaluator.state.metrics
    avg_mse = metrics["mse"]
    avg_nll = metrics["nll"]
    avg_kld = metrics["kld"]
    sparsity_fraction = metrics["sparsity_fraction"]
    neg_elbo = len(dataloader.dataset) * (avg_nll + avg_kld)

    logger.info(
        (
            "Epoch %d, val - Avg Negative ELBO %.2f, Avg NLL %.2f, Avg KLD %.2f, "
            "Avg MSE %.2f"
        ),
        trainer.state.epoch,
        neg_elbo,
        avg_nll,
        avg_kld,
        avg_mse,
    )
    writer.add_scalar("val/loss", neg_elbo, trainer.state.epoch)
    writer.add_scalar("val/nll", avg_nll, trainer.state.epoch)
    writer.add_scalar("val/kld", avg_kld, trainer.state.epoch)
    writer.add_scalar("val/mse", avg_mse, trainer.state.epoch)
    writer.add_scalar("val/sparsity_fraction", sparsity_fraction, trainer.state.epoch)


@hydra.main(config_path="../conf", config_name="config", version_base="1.1")
def run(cfg: omegaconf.OmegaConf) -> None:
    """Trains the SVAE."""
    torch.manual_seed(cfg.train.seed)
    random.seed(cfg.train.seed)

    git_hash = (
        subprocess.run(["git", "describe", "--always", "--dirty"], capture_output=True)
        .stdout.decode("utf-8")
        .rstrip()
    )
    logger.info("Git hash: %s", git_hash)

    device = (
        torch.device("cuda", cfg.train.svae.cuda_device)
        if (torch.cuda.is_available() and cfg.train.svae.use_cuda)
        else torch.device("cpu")
    )
    logger.info("Using device: %s", device)
    writer = tensorboard.writer.SummaryWriter(".")

    logger.info("Preprocessing data...")
    cfg.bin.sample_patches_custom.destination = "scratch"
    cfg.bin.sample_patches_vanilla.destination = "scratch"
    sample_patches.run(cfg)

    logger.info("Loading data...")
    train_loader, val_loader, _ = data.get_dataloaders(
        pathlib.Path(cfg.paths.user_home_dir) / pathlib.Path(cfg.paths.scratch) / "patches.pt",
        batch_size=cfg.train.svae.batch_size,
        shuffle=cfg.train.svae.shuffle,
        device=device,
    )

    logger.info("Constructing model...")
    model = models.SVAE(
        prior=models.Prior[cfg.models.svae.prior],
        prior_scale=cfg.models.svae.prior_scale,
        likelihood_logscale=cfg.models.svae.likelihood_logscale,
        collapse_delta=cfg.models.svae.collapse_delta,
    ).to(device)
    optimizer = optim.Adam(  # type: ignore
        model.parameters(), lr=cfg.train.svae.learning_rate
    )
    # optimizer = optim.Adam([ # type: ignore
    #     {'params': model._encoder.parameters(), 'lr':cfg.bin.train_svae.learning_rate},
    #     {'params': model._decoder.parameters(), 'lr':10*cfg.bin.train_svae.learning_rate}
    # ])
    # torch_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=32, gamma=0.1, verbose=True)
    # lr_scheduler = ignite.handlers.param_scheduler.LRScheduler(torch_lr_scheduler)
    nll_loss, kld_loss = model.loss_fns()

    def process_fn(engine: ignite.engine.Engine, batch: torch.Tensor):
        """Process a batch."""
        model.train()
        optimizer.zero_grad()
        x = batch.flatten(start_dim=1).to(device)
        x_pred, loc, logscale = model(x)
        nll = nll_loss(x, x_pred, loc, logscale)
        kld, sparsity_fraction = kld_loss(x, x_pred, loc, logscale)

        # some (unneccessary) annealing thing
        # marker_1 = 2
        # marker_2 = 4
        # period = marker_2 - marker_1
        # if engine.state.epoch < marker_1:
        #     beta = 0.0
        # elif engine.state.epoch < marker_2:
        #     beta = math.sin((engine.state.epoch - period) * math.pi / (2 * period))
        # else:
        #     beta = 1.0
        beta = 1.0
        #print('beta-VAE training with beta = ',beta)

        loss = len(train_loader.dataset) * (nll + beta*kld)
        loss.backward()
        optimizer.step()
        return loss.item(), nll.item(), kld.item(), sparsity_fraction

    trainer = ignite.engine.Engine(process_fn)
    ignite.metrics.RunningAverage(output_transform=lambda x: x[0]).attach(
        trainer, "loss"
    )
    ignite.metrics.RunningAverage(output_transform=lambda x: x[1]).attach(
        trainer, "nll"
    )
    ignite.metrics.RunningAverage(output_transform=lambda x: x[2]).attach(
        trainer, "kld"
    )
    trainer.add_event_handler(
        ignite.engine.Events.EPOCH_COMPLETED, log_train_metrics, model, cfg, writer
    )

    def evaluate_fn(engine: ignite.engine.Engine, batch: torch.Tensor):
        """Evaluate a batch."""
        model.eval()
        with torch.no_grad():
            x = batch.flatten(start_dim=1).to(device)
            x_pred, loc, logscale = model(x)
            kwargs = {"loc": loc, "logscale": logscale}
            return x_pred, x, kwargs

    evaluator = ignite.engine.Engine(evaluate_fn)
    ignite.metrics.RunningAverage(
        ignite.metrics.MeanSquaredError(output_transform=lambda x: [x[0], x[1]])
    ).attach(evaluator, "mse")
    ignite.metrics.Loss(nll_loss).attach(evaluator, "nll")
    ignite.metrics.Loss(
        lambda x_pred, x, loc, logscale: kld_loss(x_pred, x, loc, logscale)[0]
    ).attach(evaluator, "kld")
    ignite.metrics.Average(
        output_transform=lambda output: kld_loss(
            output[0], output[1], output[2]["loc"], output[2]["logscale"]
        )[1]
    ).attach(evaluator, "sparsity_fraction")
    trainer.add_event_handler(
        ignite.engine.Events.EPOCH_COMPLETED,
        compute_and_print_eval_metrics,
        evaluator,
        val_loader,
        cfg,
        writer,
    )
    # trainer.add_event_handler(
    #     ignite.engine.Events.EPOCH_COMPLETED, 
    #     lr_scheduler
    # )

    checkpointer = ignite.handlers.ModelCheckpoint(".", "svae")
    trainer.add_event_handler(
        ignite.engine.Events.EPOCH_COMPLETED(every=cfg.train.svae.checkpoint_every),
        checkpointer,
        {"checkpoint": model},
    )

    # Absolute dir for all models
    absolute_savedir = cfg.paths.user_home_dir + '/outputs/SavedModels/SVAE/model_files'
    def model_filename(epoch):
        return "svae_{:}_ll{:1.1e}_e{:}_lr{:1.1e}.pth".format(
            cfg.models.svae.prior,
            cfg.models.svae.likelihood_logscale,
            epoch,
            cfg.train.svae.learning_rate,
        )
        
    @trainer.on(ignite.engine.Events.EPOCH_COMPLETED(every=16))  # Event fires every 16 epochs
    def save_model_absolute(engine):
        if not os.path.exists(absolute_savedir):
            os.makedirs(absolute_savedir)
        model_path = absolute_savedir + "/" + model_filename(engine.state.epoch)
        torch.save(model.state_dict(), model_path)
        logging.info(f"Saved checkpoint at {model_path}")

    logger.info("Beginning training...")
    trainer.run(train_loader, max_epochs=cfg.train.svae.num_epochs)
    torch.save(model.state_dict(), "svae_final.pth")


    torch.save(model.state_dict(), absolute_savedir + "/" + model_filename(cfg.train.svae.num_epochs))


if __name__ == "__main__":
    run()
