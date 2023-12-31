import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


# Adaptive Instance Normalization (AdaIN)
class AdaIN(nn.Module):

    def __init__(self, style_dim, feature_dim):
        super(AdaIN, self).__init__()
        self.fc = nn.Linear(style_dim, feature_dim * 2)

    def forward(self, x, s):
        y = self.fc(s)
        gamma, beta = y.chunk(2, 1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        out = F.instance_norm(x) * (1 + gamma) + beta
        return out


class StyleGANGenerator(nn.Module):

    def __init__(self, config):
        super(StyleGANGenerator, self).__init__()

        synthesis_layers = config["StyleGAN"]["synthesis_layers"]
        self.latent_dim = config["StyleGAN"]["latent_dim"]

        layers = []
        # Initial layer
        initial_layer = synthesis_layers[0]
        layers.append(
            nn.ConvTranspose2d(self.latent_dim,
                               initial_layer[0],
                               initial_layer[2],
                               1,
                               0,
                               bias=False))
        layers.append(nn.BatchNorm2d(initial_layer[0]))
        layers.append(nn.ReLU(True))

        # Subsequent layers
        for i in range(1, len(synthesis_layers)):
            layers.append(nn.ConvTranspose2d(*synthesis_layers[i], bias=False))
            if i < len(synthesis_layers) - 1:  # Intermediate layers
                layers.append(nn.BatchNorm2d(synthesis_layers[i][1]))
                layers.append(nn.ReLU(True))
            else:  # Last layer
                layers.append(nn.Tanh())

        self.main = nn.Sequential(*layers)

    def forward(self, z):
        # Reshape the input latent vector into a 4D tensor
        z = z.view(-1, self.latent_dim, 1, 1)

        for layer in self.main:
            z = layer(z)

        return z


class StyleGANDiscriminator(nn.Module):

    def __init__(self, config):
        super(StyleGANDiscriminator, self).__init__()

        synthesis_layers = config["StyleGAN"]["synthesis_layers"]

        layers = []
        # Reverse the order of layers for the discriminator
        reversed_layers = list(reversed(synthesis_layers))

        # First layer (no batch norm and using LeakyReLU)
        layers.append(
            nn.Conv2d(reversed_layers[0][1],
                      reversed_layers[0][0],
                      reversed_layers[0][2],
                      reversed_layers[0][3],
                      reversed_layers[0][4],
                      bias=False))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Intermediate layers
        for i in range(1, len(reversed_layers)):
            layers.append(
                nn.Conv2d(reversed_layers[i][1],
                          reversed_layers[i][0],
                          reversed_layers[i][2],
                          reversed_layers[0][3],
                          reversed_layers[0][4],
                          bias=False))
            if i < len(reversed_layers) - 1:  # Intermediate layers
                layers.append(nn.BatchNorm2d(reversed_layers[i][0]))
                layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Last layer (no batch norm and using Sigmoid)
        layers.append(nn.Conv2d(reversed_layers[-1][0], 1, 2, 1, 0,
                                bias=False))
        layers.append(nn.Sigmoid())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x).view(x.size(0), -1)


class StyleGAN(pl.LightningModule):

    def __init__(self, config):
        super(StyleGAN, self).__init__()
        self.save_hyperparameters(config)
        self.config = config
        self.generator = StyleGANGenerator(config)
        self.discriminator = StyleGANDiscriminator(config)

        # Set manual optimization
        self.automatic_optimization = False

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx):
        imgs = batch
        batch_size = imgs.size(0)
        valid = torch.ones(batch_size, 1).to(self.device)
        fake = torch.zeros(batch_size, 1).to(self.device)

        # Access optimizers
        opt_g, opt_d = self.optimizers()

        # Train Generator
        opt_g.zero_grad()
        z = torch.randn(batch_size,
                        self.config["StyleGAN"]["latent_dim"]).to(self.device)
        gen_imgs = self.generator(z)
        gen_preds = self.discriminator(gen_imgs)
        g_loss = self.adversarial_loss(gen_preds, valid)
        self.log("generator_loss",
                 g_loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)
        g_loss.backward()
        opt_g.step()

        # Train Discriminator
        opt_d.zero_grad()
        real_preds = self.discriminator(imgs)
        real_loss = self.adversarial_loss(real_preds, valid)
        fake_imgs = self.generator(z)
        fake_preds = self.discriminator(fake_imgs.detach())
        fake_loss = self.adversarial_loss(fake_preds, fake)
        d_loss = (real_loss + fake_loss) / 2
        self.log("discriminator_loss",
                 d_loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)
        d_loss.backward()
        opt_d.step()

        # It's unclear whether you want to return both losses, or just one of them, or none at all.
        # Depending on your framework's requirements, you may choose what to return or to return nothing.
        # For instance, you could return a dictionary with both losses:
        return {"g_loss": g_loss, "d_loss": d_loss}

    def validation_step(self, batch, batch_idx):
        imgs = batch
        batch_size = imgs.size(0)
        valid = torch.ones(batch_size, 1).to(self.device)
        fake = torch.zeros(batch_size, 1).to(self.device)

        # Generate fake images
        z = torch.randn(batch_size,
                        self.config["StyleGAN"]["latent_dim"]).to(self.device)
        gen_imgs = self.generator(z)
        gen_preds = self.discriminator(gen_imgs)

        # Generator loss
        g_val_loss = self.adversarial_loss(gen_preds, valid)
        self.log("val_generator_loss",
                 g_val_loss,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)

        # Discriminator loss
        real_preds = self.discriminator(imgs)
        real_val_loss = self.adversarial_loss(real_preds, valid)
        fake_preds = self.discriminator(gen_imgs.detach())
        fake_val_loss = self.adversarial_loss(fake_preds, fake)
        d_val_loss = (real_val_loss + fake_val_loss) / 2
        self.log("val_discriminator_loss",
                 d_val_loss,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)

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
        optimizer_name_g = self.hparams["StyleGAN"]["Optimizer"]["name"]
        optimizer_class_g = optimizer_mapping.get(optimizer_name_g)
        if optimizer_class_g is None:
            raise ValueError(
                f"Optimizer '{optimizer_name_g}' not recognized for the generator"
            )
        opt_g = optimizer_class_g(
            self.generator.parameters(),
            **self.hparams["StyleGAN"]["Optimizer"]["params"],
        )

        # Create the discriminator optimizer dynamically based on the configuration
        optimizer_name_d = self.hparams["StyleGAN"]["Optimizer"]["name"]
        optimizer_class_d = optimizer_mapping.get(optimizer_name_d)
        if optimizer_class_d is None:
            raise ValueError(
                f"Optimizer '{optimizer_name_d}' not recognized for the discriminator"
            )
        opt_d = optimizer_class_d(
            self.discriminator.parameters(),
            **self.hparams["StyleGAN"]["Optimizer"]["params"],
        )

        # Create the generator learning rate scheduler dynamically based on the configuration
        scheduler_name_g = self.hparams["StyleGAN"]["Scheduler"]["name"]
        scheduler_class_g = scheduler_mapping.get(scheduler_name_g)
        if scheduler_class_g is None:
            raise ValueError(
                f"Scheduler '{scheduler_name_g}' not recognized for the generator"
            )
        scheduler_g = {
            "scheduler":
            scheduler_class_g(
                opt_g,
                **self.hparams["StyleGAN"]["Scheduler"]["params"],
            ),
            "monitor":
            self.hparams["Trainer"]["monitor"]
        }

        # Create the discriminator learning rate scheduler dynamically based on the configuration
        scheduler_name_d = self.hparams["StyleGAN"]["Scheduler"]["name"]
        scheduler_class_d = scheduler_mapping.get(scheduler_name_d)
        if scheduler_class_d is None:
            raise ValueError(
                f"Scheduler '{scheduler_name_d}' not recognized for the discriminator"
            )
        scheduler_d = {
            "scheduler":
            scheduler_class_d(
                opt_d,
                **self.hparams["StyleGAN"]["Scheduler"]["params"],
            ),
            "monitor":
            self.hparams["Trainer"]["monitor"]
        }

        return [opt_d, opt_g], [scheduler_d, scheduler_g]
