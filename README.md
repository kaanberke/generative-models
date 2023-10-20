# Generative Models

This repository contains implementations of various generative models, including DCGAN, GAN, and VAE.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
  - [DCGAN](#dcgan)
  - [GAN](#gan)
  - [VAE](#vae)
  - [CycleGAN](#cyclegan)
- [Configuration](#configuration)
- [License](#license)
- [Contributing](#contributing)

## Introduction

Generative models are a subset of unsupervised learning that generate new sample/data that can be considered as part of the original data. This repository provides implementations of some of the popular generative models.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/kaanberke/generative-models.git
   cd generative-models
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) If you wish to develop or contribute, you can also install the development dependencies:
   ```bash
   python setup.py develop
   ```

## Usage

To run the main script:
```bash
python main.py
```

## Models

### CycleGAN

Cycle Generative Adversarial Networks (CycleGAN) is a method for training unsupervised image-to-image translation models without paired data. The models are trained in a way that they can convert images from one domain to another and vice versa. For the code and implementation details, see models/cyclegan.py.

### DCGAN

Deep Convolutional Generative Adversarial Networks (DCGAN) is a type of GAN where the generator and discriminator are deep convnets. You can find the implementation in `models/dcgan.py`.

### GAN

Generative Adversarial Networks (GAN) consists of two networks, a generator and a discriminator, that are trained together. The generator tries to produce data that comes from some probability distribution, while the discriminator tries to tell real from fake data. Check out the implementation in `models/gan.py`.

### VAE

Variational Autoencoders (VAE) are a kind of autoencoder that's trained to learn the probability distribution of its input data. For more details, refer to `models/vae.py`.

## Configuration

The repository includes a `config.yaml` file which allows users to configure various parameters related to the models, training, and other settings.

## License

This project is licensed under the MIT License. For more details, see the [LICENSE](LICENSE) file.

## Contributing

We'd love to see your contributions! Just a friendly reminder: please stick to our coding standards and project guidelines. And, before you hit that "submit" button for a pull request, do give your changes a good test run. Thanks for being awesome! 🌟
