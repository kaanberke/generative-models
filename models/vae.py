import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb


class Encoder(nn.Module):
    def __init__(self, input_dimension, encoder_conv_layers, latent_dimension):
        """
        Initialize the Encoder network.
        
        Parameters:
            input_dimension (list): Size of the input data [width, height, channels].
            encoder_conv_layers (list): List of convolutional layer parameters.
            latent_dimension (int): Size of the latent space.
        """
        super(Encoder, self).__init__()

        # Creating convolutional layers dynamically based on encoder_conv_layers
        self.conv_layers = nn.ModuleList()
        in_channels = input_dimension[0]  # The number of input channels is specified by the third element
        for out_channels, kernel_size, stride, padding in encoder_conv_layers:
            self.conv_layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            )
            in_channels = out_channels  # Update the number of input channels for the next layer
        
        with torch.no_grad():
            dummy_input = torch.ones(1, *input_dimension, dtype=torch.float32)
            for conv_layer in self.conv_layers:
                dummy_input = F.relu(conv_layer(dummy_input))
            flattened_size = dummy_input.view(1, -1).size(1)
        
        # Linear layers for mean and log-variance
        self.fc_mu = nn.Linear(flattened_size, latent_dimension)
        self.fc_log_var = nn.Linear(flattened_size, latent_dimension)
    
    def forward(self, x):
        for conv_layer in self.conv_layers:
            x = F.relu(conv_layer(x))
        x = x.view(x.size(0), -1)  # Flatten the output
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var



class Decoder(nn.Module):
    def __init__(self, latent_dimension, decoder_deconv_layers, output_dimension):
        """
        Initialize the Decoder network.
        
        Parameters:
            latent_dimension (int): Size of the latent space.
            decoder_deconv_layers (list): List of deconvolutional layer parameters.
            output_dimension (list): Size of the output data [width, height, channels].
        """
        super(Decoder, self).__init__()

        # Creating deconvolutional layers dynamically based on decoder_deconv_layers
        self.deconv_layers = nn.ModuleList()
        in_channels = latent_dimension
        for out_channels, kernel_size, stride, padding in decoder_deconv_layers[:-1]:
            self.deconv_layers.append(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
            )
            in_channels = out_channels  # Update the number of input channels for the next layer

        # The last layer uses a sigmoid activation function
        out_channels, kernel_size, stride, padding = decoder_deconv_layers[-1]
        self.deconv_layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
                nn.Sigmoid()
            )
        )
    
    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)  # Ensure a 4D tensor with [batch_size, latent_dim, 1, 1]
        
        for deconv in self.deconv_layers[:-1]:
            z = F.relu(deconv(z))
            
        z = torch.sigmoid(self.deconv_layers[-1](z))  # sigmoid for the last layer

        return z


class VariationalAutoencoder(nn.Module):

    def __init__(self, config):
        """
        Initialize the Variational Autoencoder (VAE).

        Parameters:
            input_dimension (int): Size of the input data.
            hidden_dimension (int): Size of the hidden layers.
            latent_dimension (int): Size of the latent space.
        """
        super(VariationalAutoencoder, self).__init__()

        input_dimension = config["VAE"]["input_dimension"]
        latent_dimension = config["VAE"]["latent_dimension"]
        encoder_conv_layers = config["VAE"]["encoder_conv_layers"]
        decoder_deconv_layers = config["VAE"]["decoder_deconv_layers"]

        # Instantiate encoder and decoder networks.
        self.encoder = Encoder(input_dimension, encoder_conv_layers, latent_dimension)
        self.decoder = Decoder(latent_dimension, decoder_deconv_layers, input_dimension)


    def reparameterize(self, mean, log_variance):
        """
        Apply the reparameterization trick to sample from the latent space.

        Parameters:
            mean (torch.Tensor): Mean of the latent space distribution.
            log_variance (torch.Tensor): Log-variance of the latent space distribution.

        Returns:
            torch.Tensor: A sample from the latent space distribution.
        """
        # Calculate the standard deviation from log-variance.
        standard_deviation = torch.exp(0.5 * log_variance)

        # Generate a random tensor from a standard normal distribution.
        epsilon = torch.randn_like(standard_deviation)

        # Apply the reparameterization trick to obtain a sample from the latent space.
        latent_sample = mean + epsilon * standard_deviation

        return latent_sample

    def forward(self, input_data):
        """
        Forward pass through the VAE.

        Parameters:
            input_data (torch.Tensor): Input data.

        Returns:
            tuple: Reconstructed data, mean and log-variance of latent space distribution.
        """
        # Pass input data through encoder to obtain mean and log-variance of latent space distribution.
        mean, log_variance = self.encoder(input_data)

        # Use the reparameterization trick to sample from the latent space.
        latent_sample = self.reparameterize(mean, log_variance)

        # Pass the latent sample through the decoder to reconstruct the data.
        reconstructed_data = self.decoder(latent_sample)

        return reconstructed_data, mean, log_variance

    def loss_function(self, x, reconstructed_x, mean, log_variance):
        """
        Computes the VAE loss function.

        Parameters:
            x (torch.Tensor): The original inputs.
            reconstructed_x (torch.Tensor): The reconstruction of the inputs
                obtained from the decoder.
            mean (torch.Tensor): The mean of the variational distribution
                q(z|x).
            log_variance (torch.Tensor): The log variance of the variational
                distribution q(z|x).

        Returns:
            torch.Tensor: The total scalar loss.
        """
        # Reconstruction loss: Measures how well the decoder is doing in reconstructing
        # the input data. We use Binary Cross Entropy loss since our input values are normalized
        # and in the range [0, 1].
        recon_loss = F.binary_cross_entropy(
            reconstructed_x,
            x,
            reduction="sum"
        )

        # KL Divergence loss: Measures how much the learned latent variable distribution
        # deviates from a standard normal distribution.
        kl_loss = -0.5 * torch.sum(
                1 + log_variance - mean.pow(2) -log_variance.exp()
            )

        # Total VAE loss = reconstruction loss + KL divergence loss
        return recon_loss + kl_loss


class VAELightning(pl.LightningModule):

    def __init__(self, vae_model, config):
        """
        Initialize the Lightning Module.
        
        Parameters:
            vae_model (nn.Module): The VAE model to be trained.
            config (dict): Configuration dictionary.
        """
        super(VAELightning, self).__init__()
        self.save_hyperparameters(config)
        
        # Instantiate the VAE model to be trained.
        self.vae = vae_model

    def forward(self, x):
        """
        Forward pass through the VAE model.

        Parameters:
            x (torch.Tensor): Input data.

        Returns:
            tuple: Reconstructed data, mean, and log-variance of latent space distribution.
        """
        return self.vae(x)

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
            "CosineAnnealingWarmRestarts": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,  
            
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

        # Create the optimizer dynamically based on the configuration
        optimizer_name = self.hparams["Optimizer"]["name"]
        optimizer_class = optimizer_mapping.get(optimizer_name)
        if optimizer_class is None:
            raise ValueError(f"Optimizer '{optimizer_name}' not recognized")
        optimizer = optimizer_class(
            self.parameters(),
            # Unpack optimizer params
            **self.hparams["Optimizer"]["params"],
        )

        # Create the learning rate scheduler dynamically based on the configuration
        scheduler_name = self.hparams["Scheduler"]["name"]
        scheduler_class = scheduler_mapping.get(scheduler_name)
        if scheduler_class is None:
            raise ValueError(f"Scheduler '{scheduler_name}' not recognized")
        scheduler = {
            "scheduler": scheduler_class(
                optimizer,
                # Unpack scheduler params
                **self.hparams["Scheduler"]["params"],
            ),
            "monitor": self.hparams["Trainer"]["monitor"]
        }
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": self.hparams["Trainer"]["monitor"]
        }

    def training_step(self, batch, batch_idx):
        """
        Perform a training step.

        Parameters:
            batch (tuple): A batch of data and its corresponding labels.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: The training loss.
        """
        x = batch
        reconstructed_x, mean, log_variance = self.vae(x)  # Forward pass through VAE
        loss = self.vae.loss_function(x, reconstructed_x, mean, log_variance)  # Compute the VAE loss
        self.log("loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)  # Log the training loss
        return loss  # Return the loss

    def validation_step(self, batch, batch_idx):
        """
        Perform a validation step.

        Parameters:
            batch (tuple): A batch of data and its corresponding labels.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: The validation loss.
        """
        x = batch
        reconstructed_x, mean, log_variance = self.vae(x)  # Forward pass through VAE
        loss = self.vae.loss_function(x, reconstructed_x, mean, log_variance)  # Compute the VAE loss
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)  # Log the validation loss
        return loss  # Return the loss
    