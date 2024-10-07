
# GAN Model for Fashion MNIST

This repository contains the implementation of a Generative Adversarial Network (GAN) to generate fashion-related images using the Fashion MNIST dataset. The GAN model is built using TensorFlow and includes a training loop for both the generator and discriminator models.


## Project Overview

The goal of this project is to implement a Generative Adversarial Network (GAN) that can generate images similar to those in the Fashion MNIST dataset. The network consists of two primary components:

- **Generator**: Creates new images from random noise.
- **Discriminator**: Evaluates whether the generated images are real or fake by distinguishing them from real images from the dataset.
## Prerequisites

**Required Libraries:**  
- tensorflow  
- tensorflow-datasets  
- matplotlib  
- numpy  
- keras

You can install these dependencies using the following command: 
```bash
pip install tensorflow tensorflow-datasets matplotlib numpy keras
```

## Installation  

Clone this repository to your local machine:  
```bash
git clone https://github.com/Codder-lab/GAN_Model.git
cd GAN_Model
```
## Dataset

The project uses the Fashion MNIST dataset, which is automatically downloaded using the TensorFlow Datasets API. This dataset contains 60,000 grayscale images of fashion items like shoes, t-shirts, etc.

## Model Architecture

- **Generator**: The generator model consists of a series of dense layers that upsample the noise into a larger, more structured image. It uses ReLU activation for intermediate layers and Tanh activation for the output layer.
- **Discriminator**: The discriminator is a CNN-based model that classifies whether an image is real or generated. It uses LeakyReLU for activation and Sigmoid for the output.

## Training Process

The training process is implemented in a loop that alternates between:

1. **Discriminator Training**: The discriminator is trained on both real and fake images, computing a loss that quantifies how well it can distinguish between the two.
2. **Generator Training**: The generator is trained to fool the discriminator, optimizing the generator loss.  

The model saves checkpoints every 10 epochs to allow for resuming training.

## Results

At the end of training, generated images should closely resemble those in the Fashion MNIST dataset. You can visualize the progress by plotting the loss curves of both the generator and discriminator.


## Usage

Run the Jupyter notebook to start training the model:

```bash
jupyter notebook GAN_Model.ipynb
```

Modify the number of epochs or batch sizes in the notebook as per your preference to adjust the training process.

