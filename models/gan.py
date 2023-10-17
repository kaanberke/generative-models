import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):

    def __init__(self, latent_dim, layers):
        super(Generator, self).__init__()
        self.layers = nn.ModuleList()
        for input_dim, output_dim in zip(layers[:-1], layers[1:]):
            self.layers.append(nn.Linear(input_dim, output_dim))

    def forward(self, z):
        for layer in self.layers[:-1]:
            z = F.relu(layer(z))
        out = torch.tanh(self.layers[-1](z))  # Tanh activation for the output
        return out


class Discriminator(nn.Module):

    def __init__(self, layers):
        super(Discriminator, self).__init__()
        self.layers = nn.ModuleList()
        for input_dim, output_dim in zip(layers[:-1], layers[1:]):
            self.layers.append(nn.Linear(input_dim, output_dim))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        out = torch.sigmoid(
            self.layers[-1](x))  # Sigmoid activation for binary classification
        return out


class GANLightning(pl.LightningModule):

    def __init__(self, config):
        super(GANLightning, self).__init__()
        self.save_hyperparameters(config)

        # Generator and Discriminator
        self.generator = Generator(
            config["GAN"]["latent_dim"],
            config["GAN"]["generator_layers"]
        )
        self.discriminator = Discriminator(
            config["GAN"]["discriminator_layers"]
        )

        # Set manual optimization
        self.automatic_optimization = False

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx):
        real_imgs = batch
        batch_size = real_imgs.size(0)

        # Adversarial ground truths
        valid = torch.ones(batch_size, 1).to(self.device)
        fake = torch.zeros(batch_size, 1).to(self.device)

        # Get optimizers
        opt_d, opt_g = self.optimizers()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Zero discriminator gradients
        opt_d.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = self.adversarial_loss(
            self.discriminator(real_imgs.view(batch_size, -1)),
            valid
        )

        z = torch.randn(
            batch_size,
            self.hparams["GAN"]["latent_dim"]
        ).to(self.device)
        gen_imgs = self.generator(z).detach()
        fake_loss = self.adversarial_loss(
            self.discriminator(gen_imgs.view(batch_size, -1)), fake)

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

        # Backward pass for discriminator
        d_loss.backward()
        opt_d.step()

        # -----------------
        #  Train Generator
        # -----------------

        # Zero generator gradients
        opt_g.zero_grad()

        # Generate a batch of images
        gen_imgs = self.generator(z)

        # Generator's objective is to have the Discriminator classify its output as real
        g_loss = self.adversarial_loss(
            self.discriminator(gen_imgs),
            valid
        )
        self.log(
            "generator_loss",
            g_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )

        # Backward pass for generator
        g_loss.backward()
        opt_g.step()

        return {"loss": (g_loss + d_loss) / 2}

    def validation_step(self, batch, batch_idx):
        real_imgs = batch
        batch_size = real_imgs.size(0)

        # Flatten the images
        real_imgs = real_imgs.view(batch_size, -1)

        # Adversarial ground truths
        valid = torch.ones(batch_size, 1).to(self.device)
        fake = torch.zeros(batch_size, 1).to(self.device)

        # Generate a batch of images
        z = torch.randn(
            batch_size,
            self.hparams["GAN"]["latent_dim"]
        ).to(self.device)
        gen_imgs = self.generator(z)

        # Flatten the generated images
        gen_imgs = gen_imgs.view(batch_size, -1)

        # Generator's loss on validation data
        g_val_loss = self.adversarial_loss(self.discriminator(gen_imgs), valid)

        # Discriminator's ability to classify real from generated samples on validation data
        real_val_loss = self.adversarial_loss(
            self.discriminator(real_imgs),
            valid
        )
        fake_val_loss = self.adversarial_loss(
            self.discriminator(gen_imgs.detach()),
            fake
        )
        d_val_loss = (real_val_loss + fake_val_loss) / 2

        self.log(
            "val_generator_loss",
            g_val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )
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
