import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

ACTIVATIONS_MAPPING = {
    "relu": "ReLU",
    "leaky_relu": "LeakyReLU",
    "sigmoid": "Sigmoid",
    "tanh": "Tanh",
    "softmax": "Softmax",
    "prelu": "PReLU",
}


class DCGANGenerator(nn.Module):

    def __init__(self, config):
        super(DCGANGenerator, self).__init__()
        self.config = config

        layers = []

        # Then add transposed convolution layers from the config
        for layer_config in config["DCGAN"]["Generator"]["layers"]:
            layers.append(nn.ConvTranspose2d(*layer_config))
            if layer_config != config["DCGAN"]["Generator"]["layers"][
                    -1]:  # If it's not the last layer
                activation = getattr(
                    nn, ACTIVATIONS_MAPPING[config["DCGAN"]["Generator"]
                                            ["intermediate_activations"]])()
                layers.append(activation)
                layers.append(nn.BatchNorm2d(layer_config[1]))
            else:
                activation = getattr(
                    nn, ACTIVATIONS_MAPPING[config["DCGAN"]["Generator"]
                                            ["final_activation"]])()
                layers.append(activation)

        self.main = nn.Sequential(*layers)

    def forward(self, z):
        z = z.view(z.size(0),
                   *self.config["DCGAN"]["Generator"]["initial_size"])
        return self.main(z)


class DCGANDiscriminator(nn.Module):

    def __init__(self, config):
        super(DCGANDiscriminator, self).__init__()
        self.config = config

        layers = []
        for layer_config in config["DCGAN"]["Discriminator"]["layers"]:
            layers.append(nn.Conv2d(*layer_config))
            # If it's not the last layer
            if layer_config != config["DCGAN"]["Discriminator"]["layers"][-1]:
                activation = getattr(
                    nn, 
                    ACTIVATIONS_MAPPING[
                        config["DCGAN"]["Discriminator"]["intermediate_activations"]
                    ]
                )()
                layers.append(activation)
                layers.append(nn.BatchNorm2d(layer_config[1]))
            else:
                activation = getattr(
                    nn, ACTIVATIONS_MAPPING[
                        config["DCGAN"]["Discriminator"]["final_activation"]
                    ]
                )()
                layers.append(activation)

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        # Flattening the output
        return self.main(x).view(x.size(0), -1)


class DCGANLightning(pl.LightningModule):

    def __init__(self, config):
        super(DCGANLightning, self).__init__()
        self.save_hyperparameters(config)

        # Generator and Discriminator
        self.generator = DCGANGenerator(config)
        self.discriminator = DCGANDiscriminator(config)

        # Set manual optimization
        self.automatic_optimization = False

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx):
        imgs = batch
        batch_size = imgs.size(0)

        # Real images label as 1, fake images label as 0
        valid = torch.ones(batch_size, 1).to(self.device)
        fake = torch.zeros(batch_size, 1).to(self.device)

        # Get optimizers
        opt_d, opt_g = self.optimizers()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        opt_d.zero_grad()

        # Real images
        real_preds = self.discriminator(imgs)
        real_loss = self.adversarial_loss(real_preds, valid)

        # Fake images
        z = torch.randn(
            batch_size,
            self.hparams["DCGAN"]["Generator"]["latent_dim"],
            1,
            1
        ).to(self.device)
        fake_imgs = self.generator(z)
        fake_preds = self.discriminator(fake_imgs.detach())
        fake_loss = self.adversarial_loss(fake_preds, fake)

        # Total discriminator loss
        d_loss = (real_loss + fake_loss) / 2
        self.log(
            "discriminator_loss",
            d_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )
        d_loss.backward()
        opt_d.step()

        # -----------------
        #  Train Generator
        # -----------------
        opt_g.zero_grad()

        # Generate fake images
        gen_imgs = self.generator(z)
        gen_preds = self.discriminator(gen_imgs)

        # Generator loss
        g_loss = self.adversarial_loss(gen_preds, valid)
        self.log(
            "generator_loss",
            g_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )
        g_loss.backward()
        opt_g.step()

        return {
            "loss": (g_loss + d_loss) / 2
        }

    def validation_step(self, batch, batch_idx):
        imgs = batch
        batch_size = imgs.size(0)

        # Real images label as 1, fake images label as 0
        valid = torch.ones(batch_size, 1).to(self.device)
        fake = torch.zeros(batch_size, 1).to(self.device)

        # Fake images
        z = torch.randn(
            batch_size,
            self.hparams["DCGAN"]["Generator"]["latent_dim"],
            1,
            1
        ).to(self.device)
        gen_imgs = self.generator(z)
        gen_preds = self.discriminator(gen_imgs)

        # Generator loss
        g_val_loss = self.adversarial_loss(gen_preds, valid)
        self.log(
            "val_generator_loss",
            g_val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )

        # Discriminator loss
        real_preds = self.discriminator(imgs)
        real_val_loss = self.adversarial_loss(real_preds, valid)
        fake_preds = self.discriminator(gen_imgs.detach())
        fake_val_loss = self.adversarial_loss(fake_preds, fake)
        d_val_loss = (real_val_loss + fake_val_loss) / 2
        self.log(
            "val_discriminator_loss",
            d_val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )

        return {
            "val_gen_loss": g_val_loss,
            "val_dis_loss": d_val_loss
        }

    def configure_optimizers(self):
        """
        Configure the optimizer and learning rate scheduler used for training.

        Returns:
            dict: The optimizer and learning rate scheduler to be used during training.
        """

        optimizer_mapping = {
            # Combines the benefits of Adagrad and RMSProp
            "Adam": torch.optim.Adam,

            # Widely used, often preferred for a lot of data
            "SGD": torch.optim.SGD,

            # Suitable for non-stationary objectives
            "RMSprop": torch.optim.RMSprop,

            # Adapts learning rates based on a moving window of gradient updates
            "Adadelta": torch.optim.Adadelta,

            # Suitable for dealing with sparse data
            "Adagrad": torch.optim.Adagrad,

            # Modifies the weight decay in Adam, often leading to better generalization
            "AdamW": torch.optim.AdamW,

            # A variant of Adam optimized for sparse tensors
            "SparseAdam": torch.optim.SparseAdam,

            # A variant of Adam robust to noisy gradient estimates
            "Adamax": torch.optim.Adamax,

            # Converges to an average of all parameters across iterations
            "ASGD": torch.optim.ASGD,

            # Suited for problems with a large number of variables
            "LBFGS": torch.optim.LBFGS,

            # Often used for batch training
            "Rprop": torch.optim.Rprop,
        }

        scheduler_mapping = {
            # Chains schedulers together sequentially
            "ChainedScheduler": torch.optim.lr_scheduler._LRScheduler,

            # Keeps the learning rate constant
            "ConstantLR": torch.optim.lr_scheduler.LambdaLR,

            # Anneals the learning rate with a cosine function
            "CosineAnnealingLR": torch.optim.lr_scheduler.CosineAnnealingLR,

            # Uses a cosine annealing schedule
            "CosineAnnealingWarmRestarts":
            torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,

            # Cycles the learning rate between two boundaries
            "CyclicLR": torch.optim.lr_scheduler.CyclicLR,

            # Decays the learning rate exponentially
            "ExponentialLR": torch.optim.lr_scheduler.ExponentialLR,

            # Applies a user-defined lambda function to the learning rate
            "LambdaLR": torch.optim.lr_scheduler.LambdaLR,

            # Decays the learning rate at specific milestones
            "MultiStepLR": torch.optim.lr_scheduler.MultiStepLR,

            # Multiplies the learning rate by a given factor
            "MultiplicativeLR": torch.optim.lr_scheduler.MultiplicativeLR,

            # Cycles the learning rate and momentum
            "OneCycleLR": torch.optim.lr_scheduler.OneCycleLR,

            # Reduces learning rate when a metric has stopped improving
            "ReduceLROnPlateau": torch.optim.lr_scheduler.ReduceLROnPlateau,

            # Decays the learning rate at regular intervals
            "StepLR": torch.optim.lr_scheduler.StepLR,
        }

        # Create the generator optimizer dynamically based on the configuration
        optimizer_name_g = self.hparams["GAN"]["Optimizer"]["name"]
        optimizer_class_g = optimizer_mapping.get(optimizer_name_g)
        if optimizer_class_g is None:
            raise ValueError(
                f"Optimizer '{optimizer_name_g}' not recognized for the generator"
            )
        opt_g = optimizer_class_g(
            self.generator.parameters(),
            **self.hparams["GAN"]["Optimizer"]["params"],
        )

        # Create the discriminator optimizer dynamically based on the configuration
        optimizer_name_d = self.hparams["GAN"]["Optimizer"]["name"]
        optimizer_class_d = optimizer_mapping.get(optimizer_name_d)
        if optimizer_class_d is None:
            raise ValueError(
                f"Optimizer '{optimizer_name_d}' not recognized for the discriminator"
            )
        opt_d = optimizer_class_d(
            self.discriminator.parameters(),
            **self.hparams["GAN"]["Optimizer"]["params"],
        )

        # Create the generator learning rate scheduler dynamically based on the configuration
        scheduler_name_g = self.hparams["GAN"]["Scheduler"]["name"]
        scheduler_class_g = scheduler_mapping.get(scheduler_name_g)
        if scheduler_class_g is None:
            raise ValueError(
                f"Scheduler '{scheduler_name_g}' not recognized for the generator"
            )
        scheduler_g = {
            "scheduler":
            scheduler_class_g(
                opt_g,
                **self.hparams["GAN"]["Scheduler"]["params"],
            ),
            "monitor": self.hparams["Trainer"]["monitor"]
        }

        # Create the discriminator learning rate scheduler dynamically based on the configuration
        scheduler_name_d = self.hparams["GAN"]["Scheduler"]["name"]
        scheduler_class_d = scheduler_mapping.get(scheduler_name_d)
        if scheduler_class_d is None:
            raise ValueError(
                f"Scheduler '{scheduler_name_d}' not recognized for the discriminator"
            )
        scheduler_d = {
            "scheduler": scheduler_class_d(
                opt_d,
                **self.hparams["GAN"]["Scheduler"]["params"],
            ),
            "monitor": self.hparams["Trainer"]["monitor"]
        }

        return [opt_d, opt_g], [scheduler_d, scheduler_g]
