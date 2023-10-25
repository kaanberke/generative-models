import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl

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
        for layer_config in config["WGAN"]["generator_layers"]:
            layers.append(nn.ConvTranspose2d(*layer_config))
            if layer_config != config["WGAN"]["generator_layers"][-1]:
                activation = getattr(
                    nn,
                    ACTIVATIONS_MAPPING[
                        config["WGAN"]["intermediate_activations"]
                    ]
                )()
                layers.append(activation)
                layers.append(nn.BatchNorm2d(layer_config[1]))
            else:
                activation = getattr(
                    nn, ACTIVATIONS_MAPPING[
                        config["WGAN"]["final_activation"]
                    ]
                )()
                layers.append(activation)

        self.main = nn.Sequential(*layers)

    def forward(self, z):
        z = z.view(
            z.size(0),
            *self.config["WGAN"]["initial_size"]
        )
        return self.main(z)


class Critic(nn.Module):

    def __init__(self, config):
        super(Critic, self).__init__()
        self.config = config

        layers = []

        # Add convolution layers from the config
        for layer_config in config["WGAN"]["critic_layers"]:
            layers.append(nn.Conv2d(*layer_config))
            if layer_config != config["WGAN"]["critic_layers"][-1]:
                activation = getattr(
                    nn,
                    ACTIVATIONS_MAPPING[
                        config["WGAN"]["intermediate_activations"]
                    ]
                )()
                layers.append(activation)
                layers.append(nn.BatchNorm2d(layer_config[1]))
            else:
                activation = getattr(
                    nn, ACTIVATIONS_MAPPING[
                        config["WGAN"]["final_activation"]
                    ]
                )()
                layers.append(activation)

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)



class WGANLightning(pl.LightningModule):

    def __init__(self, config):
        super(WGANLightning, self).__init__()
        self.save_hyperparameters(config)

        # Generator and Critic
        self.generator = Generator(config=config)
        self.critic = Critic(config=config)

        # Set manual optimization
        self.automatic_optimization = False

        # Weight clipping parameter
        self.clip_value = 0.01

    def forward(self, z):
        return self.generator(z)

    def training_step(self, batch, batch_idx):
        real_imgs = batch
        batch_size = real_imgs.size(0)

        # Sample noise as generator input
        z = torch.randn(batch_size, self.hparams["WGAN"]["latent_dim"]).to(self.device)
        
        # Generate a batch of images
        gen_imgs = self.generator(z)
        
        # Get optimizers
        opt_critic, opt_gen = self.optimizers()

        # ---------------------
        #  Train Critic
        # ---------------------
        # Zero critic gradients
        opt_critic.zero_grad()

        # Measure critic's ability to differentiate real from fake
        real_loss = -torch.mean(self.critic(real_imgs.view(batch_size, -1)))
        fake_loss = torch.mean(self.critic(gen_imgs.detach().view(batch_size, -1)))
        critic_loss = real_loss + fake_loss

        self.log("critic_loss", critic_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        # Backward pass for critic
        critic_loss.backward()
        opt_critic.step()

        # Clip weights of critic
        for p in self.critic.parameters():
            p.data.clamp_(-self.clip_value, self.clip_value)

        # -----------------
        #  Train Generator
        # -----------------
        # Zero generator gradients
        opt_gen.zero_grad()

        # Regenerate a batch of images
        gen_imgs = self.generator(z)
        
        # Generator's objective is to make the Critic assign high scores to its outputs
        generator_loss = -torch.mean(self.critic(gen_imgs.view(batch_size, -1)))
        self.log("generator_loss", generator_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # Backward pass for generator
        generator_loss.backward()
        opt_gen.step()

        return {
            "critic_loss": critic_loss,
            "generator_loss": generator_loss
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
        optimizer_name_g = self.hparams["WGAN"]["Optimizer"]["name"]
        optimizer_class_g = optimizer_mapping.get(optimizer_name_g)
        if optimizer_class_g is None:
            raise ValueError(
                f"Optimizer '{optimizer_name_g}' not recognized for the generator"
            )
        opt_g = optimizer_class_g(
            self.generator.parameters(),
            **self.hparams["WGAN"]["Optimizer"]["params"],
        )

        # Create the critic optimizer dynamically based on the configuration
        optimizer_name_d = self.hparams["WGAN"]["Optimizer"]["name"]
        optimizer_class_d = optimizer_mapping.get(optimizer_name_d)
        if optimizer_class_d is None:
            raise ValueError(
                f"Optimizer '{optimizer_name_d}' not recognized for the critic"
            )
        opt_d = optimizer_class_d(
            self.critic.parameters(),
            **self.hparams["WGAN"]["Optimizer"]["params"],
        )

        # Create the generator learning rate scheduler dynamically based on the configuration
        scheduler_name_g = self.hparams["WGAN"]["Scheduler"]["name"]
        scheduler_class_g = scheduler_mapping.get(scheduler_name_g)
        if scheduler_class_g is None:
            raise ValueError(
                f"Scheduler '{scheduler_name_g}' not recognized for the generator"
            )
        scheduler_g = {
            "scheduler":
            scheduler_class_g(
                opt_g,
                **self.hparams["WGAN"]["Scheduler"]["params"],
            ),
            "monitor":
            self.hparams["Trainer"]["monitor"]
        }

        # Create the critic learning rate scheduler dynamically based on the configuration
        scheduler_name_d = self.hparams["WGAN"]["Scheduler"]["name"]
        scheduler_class_d = scheduler_mapping.get(scheduler_name_d)
        if scheduler_class_d is None:
            raise ValueError(
                f"Scheduler '{scheduler_name_d}' not recognized for the critic"
            )
        scheduler_d = {
            "scheduler":
            scheduler_class_d(
                opt_d,
                **self.hparams["WGAN"]["Scheduler"]["params"],
            ),
            "monitor":
            self.hparams["Trainer"]["monitor"]
        }

        return [opt_d, opt_g], [scheduler_d, scheduler_g]
