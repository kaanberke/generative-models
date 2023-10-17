import pretty_errors
import lightning.pytorch as pl
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from torchvision import transforms
from pathlib import Path
import yaml
from datamodules import CustomDataModule
from models.gan import Generator, Discriminator, GANLightning  # Assuming the GAN classes are in a module called 'gan'

if __name__ == "__main__":
    
    # Load configuration from config.yaml
    with open("config.yaml", "r") as stream:
        config = yaml.safe_load(stream)
    
    # Data module setup
    data_transforms = transforms.Compose([
        transforms.Resize(config["Trainer"]["image_size"]),
        transforms.ToTensor(),
    ])
    data_module = CustomDataModule(
        data_dir=config["Trainer"]["data_dir"],
        batch_size=config["Trainer"]["batch_size"],
        train_transforms=data_transforms,
        val_test_transforms=data_transforms,
        train_val_test_split=config["Trainer"]["train_val_test_split"]
    )

    # GAN Model setup
    gan_lightning = GANLightning(config=config)  # Use the GAN section from the config
    checkpoint_callback = ModelCheckpoint(
            dirpath=config["Checkpoint"]["dirpath"],
            save_top_k=config["Checkpoint"]["save_top_k"],
            monitor=config["Checkpoint"]["monitor"],
            filename=config["Checkpoint"]["filename"],
            mode=config["Checkpoint"]["mode"],
            verbose=config["Checkpoint"]["verbose"],
    )
    lr_callback = LearningRateMonitor(logging_interval="step")

    # Trainer setup
    trainer = Trainer(
        max_epochs=config["Trainer"]["max_epochs"],
        accelerator=config["Trainer"]["accelerator"],
        num_sanity_val_steps=config["Trainer"]["num_sanity_val_steps"],
        logger=pl.loggers.TensorBoardLogger("lightning_logs/"),
        callbacks=[
            checkpoint_callback,
            lr_callback
        ]
    )
    
    # Training
    trainer.fit(gan_lightning, datamodule=data_module)
