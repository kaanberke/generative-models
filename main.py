import pretty_errors
import lightning.pytorch as pl
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from torchvision import transforms
from pathlib import Path
import yaml
from datamodules import CustomDataModule
from models.vae import VariationalAutoencoder, VAELightning

if __name__ == "__main__":
    
    # Load configuration from config.yaml
    with open("config.yaml", "r") as stream:
        config = yaml.safe_load(stream)
    
    # Data module setup
    data_transforms = transforms.Compose([
        transforms.Resize(config["image_size"]),
        transforms.ToTensor(),
    ])
    data_module = CustomDataModule(
        data_dir=config["data_dir"],
        batch_size=config["batch_size"],
        train_transforms=data_transforms,
        val_test_transforms=data_transforms,
        train_val_test_split=config["train_val_test_split"]
    )

    # VAE Model setup
    vae_model = VariationalAutoencoder(
        input_dimension=config["VAE"]["input_dimension"],
        hidden_dimension=config["VAE"]["hidden_dimension"],
        latent_dimension=config["VAE"]["latent_dimension"]
    )

    vae_lightning = VAELightning(vae_model, lr=config["learning_rate"])
    # TODO: Solve "Expected Parent" value error
    # checkpoint_callback = ModelCheckpoint(
    #         dirpath="checkpoints/",
    #         save_top_k=1,
    #         monitor="val_loss",
    #         filename="model_{epoch:02d}_{val_loss:.2f}",
    #         mode="min"
    # ),
    # lr_callback = LearningRateMonitor(logging_interval="step")

    # Trainer setup
    trainer = Trainer(
        max_epochs=config["max_epochs"],
        accelerator=config["accelerator"],
        num_sanity_val_steps=config["num_sanity_val_steps"],
        logger=pl.loggers.TensorBoardLogger("lightning_logs/"),
        callbacks=[
            # checkpoint_callback,
            # lr_callback
        ]
    )
    
    # Training
    trainer.fit(vae_lightning, datamodule=data_module)
