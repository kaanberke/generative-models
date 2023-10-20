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


class Generator(nn.Module):

    def __init__(self, config):
        super(Generator, self).__init__()
        self.config = config

        layers = []

        # Add transposed convolution layers from the config
        for layer_config in config["CycleGAN"]["Generator"]["layers"]:
            layers.append(nn.ConvTranspose2d(*layer_config))
            if layer_config != config["CycleGAN"]["Generator"]["layers"][-1]:
                activation = getattr(
                    nn,
                    ACTIVATIONS_MAPPING[
                        config["CycleGAN"]["Generator"]["intermediate_activations"]
                    ]
                )()
                layers.append(activation)
                layers.append(nn.BatchNorm2d(layer_config[1]))
            else:
                activation = getattr(
                    nn, ACTIVATIONS_MAPPING[
                        config["CycleGAN"]["Generator"]["final_activation"]
                    ]
                )()
                layers.append(activation)

        self.main = nn.Sequential(*layers)

    def forward(self, z):
        z = z.view(
            z.size(0),
            *self.config["CycleGAN"]["Generator"]["initial_size"]
        )
        return self.main(z)


class Discriminator(nn.Module):

    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.config = config

        layers = []
        for layer_config in config["CycleGAN"]["Discriminator"]["layers"]:
            layers.append(nn.Conv2d(*layer_config))
            if layer_config != config["CycleGAN"]["Discriminator"]["layers"][-1]:
                activation = getattr(
                    nn,
                    ACTIVATIONS_MAPPING[
                        config["CycleGAN"]["Discriminator"]["intermediate_activations"]
                    ]
                )()
                layers.append(activation)
                layers.append(nn.BatchNorm2d(layer_config[1]))
            else:
                activation = getattr(
                    nn,
                    ACTIVATIONS_MAPPING[
                        config["CycleGAN"]["Discriminator"]["final_activation"]
                    ]
                )()
                layers.append(activation)

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x).view(x.size(0), -1)


class CycleGANLightning(pl.LightningModule):

    def __init__(self, config):
        super(CycleGANLightning, self).__init__()
        self.save_hyperparameters(config)

        # Generators and Discriminators
        self.G = Generator(config)
        self.F = Generator(config)
        self.D_A = Discriminator(config)
        self.D_B = Discriminator(config)

        # Set manual optimization
        self.automatic_optimization = False

    def forward(self, x, direction="AtoB"):
        if direction == "AtoB":
            return self.G(x)
        else:
            return self.F(x)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def cycle_loss(self, recovered, real):
        return F.l1_loss(recovered, real)

    def identity_loss(self, same, real):
        return F.l1_loss(same, real)


    def training_step(self, batch, batch_idx):
        real_X, real_Y = batch

        valid = torch.ones(real_X.size(0), 1).to(self.device)
        fake = torch.zeros(real_X.size(0), 1).to(self.device)

        # Get optimizers
        opt_d_X, opt_d_Y, opt_g = self.optimizers()

        # ---------------------
        #  Train Generators
        # ---------------------
        opt_g.zero_grad()

        # Translate images to the opposite domain
        fake_Y = self.generator_G(real_X)
        fake_X = self.generator_F(real_Y)

        # Generator losses
        loss_G_Y = self.adversarial_loss(self.discriminator_Y(fake_Y), valid)
        loss_G_X = self.adversarial_loss(self.discriminator_X(fake_X), valid)
        g_loss = 0.5 * (loss_G_Y + loss_G_X)
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

        # ---------------------
        #  Train Discriminator Y
        # ---------------------
        opt_d_Y.zero_grad()

        real_preds_Y = self.discriminator_Y(real_Y)
        fake_preds_Y = self.discriminator_Y(fake_Y.detach())

        real_loss_Y = self.adversarial_loss(real_preds_Y, valid)
        fake_loss_Y = self.adversarial_loss(fake_preds_Y, fake)
        d_loss_Y = 0.5 * (real_loss_Y + fake_loss_Y)
        self.log(
            "discriminator_Y_loss",
            d_loss_Y,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )
        d_loss_Y.backward()
        opt_d_Y.step()

        # ---------------------
        #  Train Discriminator X
        # ---------------------
        opt_d_X.zero_grad()

        real_preds_X = self.discriminator_X(real_X)
        fake_preds_X = self.discriminator_X(fake_X.detach())

        real_loss_X = self.adversarial_loss(real_preds_X, valid)
        fake_loss_X = self.adversarial_loss(fake_preds_X, fake)
        d_loss_X = 0.5 * (real_loss_X + fake_loss_X)
        self.log(
            "discriminator_X_loss",
            d_loss_X,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )
        d_loss_X.backward()
        opt_d_X.step()

        # ---------------------
        # Return Losses for Logging
        # ---------------------
        return {
            "loss": (g_loss + d_loss_Y + d_loss_X) / 3,
            "g_loss": g_loss,
            "d_loss_Y": d_loss_Y,
            "d_loss_X": d_loss_X
        }


    def validation_step(self, batch, batch_idx):
        real_X, real_Y = batch

        valid = torch.ones(real_X.size(0), 1).to(self.device)
        fake = torch.zeros(real_X.size(0), 1).to(self.device)

        # Translate images to opposite domain
        fake_Y = self.generator_G(real_X)
        fake_X = self.generator_F(real_Y)

        # Generator loss
        loss_G_Y = self.adversarial_loss(self.discriminator_Y(fake_Y), valid)
        loss_G_X = self.adversarial_loss(self.discriminator_X(fake_X), valid)
        g_val_loss = 0.5 * (loss_G_Y + loss_G_X)
        self.log(
            "val_generator_loss",
            g_val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )

        # Discriminator loss for Y
        real_preds_Y = self.discriminator_Y(real_Y)
        fake_preds_Y = self.discriminator_Y(fake_Y.detach())
        real_loss_Y = self.adversarial_loss(real_preds_Y, valid)
        fake_loss_Y = self.adversarial_loss(fake_preds_Y, fake)
        d_val_loss_Y = 0.5 * (real_loss_Y + fake_loss_Y)

        # Discriminator loss for X
        real_preds_X = self.discriminator_X(real_X)
        fake_preds_X = self.discriminator_X(fake_X.detach())
        real_loss_X = self.adversarial_loss(real_preds_X, valid)
        fake_loss_X = self.adversarial_loss(fake_preds_X, fake)
        d_val_loss_X = 0.5 * (real_loss_X + fake_loss_X)

        # Average Discriminator loss
        d_val_loss = 0.5 * (d_val_loss_Y + d_val_loss_X)
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
            "val_dis_loss_Y": d_val_loss_Y,
            "val_dis_loss_X": d_val_loss_X,
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
            "monitor":
            self.hparams["Trainer"]["monitor"]
        }

        # Create the discriminator learning rate scheduler dynamically based on the configuration
        scheduler_name_d = self.hparams["GAN"]["Scheduler"]["name"]
        scheduler_class_d = scheduler_mapping.get(scheduler_name_d)
        if scheduler_class_d is None:
            raise ValueError(
                f"Scheduler '{scheduler_name_d}' not recognized for the discriminator"
            )
        scheduler_d = {
            "scheduler":
            scheduler_class_d(
                opt_d,
                **self.hparams["GAN"]["Scheduler"]["params"],
            ),
            "monitor":
            self.hparams["Trainer"]["monitor"]
        }

        return [opt_d, opt_g], [scheduler_d, scheduler_g]
