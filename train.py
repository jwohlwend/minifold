import argparse
import subprocess
import sys
import yaml

import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

from minifold.train.model import MiniFold
from minifold.train.data import MiniFoldDataModule


def main(args):
    # Load config
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Set defaults
    if "trainer" not in config:
        config["trainer"] = {}
    if "data" not in config:
        config["data"] = {}
    if "model" not in config:
        config["model"] = {}

    # Flip some arguments in debug mode
    if args.debug:
        config["trainer"]["fast_dev_run"] = True
        config["trainer"]["devices"] = 1
        config["data"]["num_workers"] = 0
        if "wandb" in config:
            config.pop("wandb")

    data_module = MiniFoldDataModule(**config["data"])
    model_module = MiniFold(**config["model"])

    # Create objects
    data_module.prepare_data()
    data_module.setup("fit")

    # Create wandb logger
    loggers = []
    if "wandb" in config:
        wdb_logger = WandbLogger(
            group=config["wandb"]["name"],
            save_dir=args.output,
            project=config["wandb"]["project"],
            entity=config["wandb"]["entity"],
            log_model=False,
        )
        loggers.append(wdb_logger)
        freeze_path = f"{wdb_logger.experiment.dir}/package_versions.txt"
        with open(freeze_path, "w") as freeze_file:
            subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=freeze_file)
        wdb_logger.experiment.save(f"{freeze_path}")
        wdb_logger.experiment.save(args.config)

    # Create checkpoint callback
    if args.disable_checkpoint:
        callbacks = []
    else:
        mc = ModelCheckpoint(
            monitor="val/lddt", save_top_k=1, save_last=True, mode="max"
        )
        callbacks = [mc]

    # Set up trainer
    strategy = "auto"
    if config["trainer"]["devices"] > 1:
        strategy = DDPStrategy(find_unused_parameters=False)

    trainer = pl.Trainer(
        default_root_dir=args.output,
        strategy=strategy,
        callbacks=callbacks,
        logger=loggers,
        enable_checkpointing=not args.disable_checkpoint,
        reload_dataloaders_every_n_epochs=1,
        **config["trainer"],
    )

    # Run training
    trainer.fit(model_module, datamodule=data_module, ckpt_path=args.resume)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Config file to execute")
    parser.add_argument(
        "--output",
        default="./",
        help="Directory to save the model and logs",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to checkpoint to resume training",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Whether to run in debug mode",
    )
    parser.add_argument(
        "--disable_checkpoint",
        action="store_true",
        default=False,
        help="Whether to disable checkpointing",
    )
    args = parser.parse_args()
    main(args)
