import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):

    def __init__(self, input_dimension, hidden_dimension, latent_dimension):
        """
        Define the encoder network that maps input data to the latent space.

        Parameters:
            input_dimension (int): Size of the input data.
            hidden_dimension (int): Size of the hidden layer.
            latent_dimension (int): Size of the latent space.
        """
        super(Encoder, self).__init__()

        # Define a fully connected layer to map input data to the hidden layer.
        self.hidden_layer = nn.Linear(input_dimension, hidden_dimension)

        # Define fully connected layers to map hidden layer to mean and log-variance.
        self.mean_layer = nn.Linear(hidden_dimension, latent_dimension)
        self.log_variance_layer = nn.Linear(hidden_dimension, latent_dimension)

    def forward(self, input_data):
        """
        Forward pass through the encoder network.

        Parameters:
            input_data (torch.Tensor): Input data.

        Returns:
            tuple: Mean and log-variance of the latent space distribution.
        """
        # Apply a ReLU activation function after passing input data through the hidden layer.
        hidden_representation = F.relu(self.hidden_layer(input_data))

        # Obtain mean and log-variance from the hidden representation.
        mean = self.mean_layer(hidden_representation)
        log_variance = self.log_variance_layer(hidden_representation)

        return mean, log_variance


class Decoder(nn.Module):

    def __init__(self, latent_dimension, hidden_dimension, output_dimension):
        """
        Define the decoder network that reconstructs data from the latent space.

        Parameters:
            latent_dimension (int): Size of the latent space.
            hidden_dimension (int): Size of the hidden layer.
            output_dimension (int): Size of the reconstructed data.
        """
        super(Decoder, self).__init__()

        # Define a fully connected layer to map from latent space to hidden layer.
        self.hidden_layer = nn.Linear(latent_dimension, hidden_dimension)

        # Define a fully connected layer to map from hidden layer to output data.
        self.output_layer = nn.Linear(hidden_dimension, output_dimension)

    def forward(self, latent_representation):
        """
        Forward pass through the decoder network.

        Parameters:
            latent_representation (torch.Tensor): Sample from the latent space.

        Returns:
            torch.Tensor: Reconstructed data.
        """
        # Apply a ReLU activation function after passing latent sample through the hidden layer.
        hidden_representation = F.relu(
            self.hidden_layer(latent_representation))

        # Use a sigmoid activation function to reconstruct the input data from the hidden representation.
        reconstructed_data = torch.sigmoid(
            self.output_layer(hidden_representation))

        return reconstructed_data

class VariationalAutoencoder(nn.Module):

    def __init__(self, input_dimension, hidden_dimension, latent_dimension):
        """
        Initialize the Variational Autoencoder (VAE).

        Parameters:
            input_dimension (int): Size of the input data.
            hidden_dimension (int): Size of the hidden layers.
            latent_dimension (int): Size of the latent space.
        """
        super(VariationalAutoencoder, self).__init__()

        # Instantiate encoder and decoder networks.
        self.encoder = Encoder(input_dimension, hidden_dimension, latent_dimension)
        self.decoder = Decoder(latent_dimension, hidden_dimension, input_dimension)

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
        recon_loss = F.binary_cross_entropy(reconstructed_x, x, reduction="sum")

        # KL Divergence loss: Measures how much the learned latent variable distribution 
        # deviates from a standard normal distribution.
        kl_loss = -0.5 * torch.sum(1 + log_variance - mean.pow(2) - log_variance.exp())
        
        # Total VAE loss = reconstruction loss + KL divergence loss
        return recon_loss + kl_loss