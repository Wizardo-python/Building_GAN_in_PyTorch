import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
NOISE_DIM = 96


def sample_noise(batch_size, noise_dim, dtype=torch.float, device='cpu'):
  """
  Generate a PyTorch Tensor of uniform random noise.

  Input:
  - batch_size: Integer giving the batch size of noise to generate.
  - noise_dim: Integer giving the dimension of noise to generate.
  
  Output:
  - A PyTorch Tensor of shape (batch_size, noise_dim) containing uniform
    random noise in the range (-1, 1).
  """
  # Implementing sample_noise.
  noise = torch.rand(batch_size, noise_dim, dtype = dtype, device = device)
  noise = noise * 2 - 1

  return noise



def discriminator():
  """
  Build and return a PyTorch nn.Sequential model implementing the architecture in the notebook.
  """
  # Implementing discriminator.
  model = nn.Sequential(
          nn.Linear(784, 256),
          nn.LeakyReLU(0.01),
          nn.Linear(256, 256),
          nn.LeakyReLU(0.01),
          nn.Linear(256, 1))
  
  return model


def generator(noise_dim=NOISE_DIM):
  """
  Build and return a PyTorch nn.Sequential model implementing the architecture in the notebook.
  """
  # Implementing generator.
  model = nn.Sequential(
          nn.Linear(noise_dim, 1024),
          nn.ReLU(),
          nn.Linear(1024, 1024),
          nn.ReLU(),
          nn.Linear(1024, 784),
          nn.Tanh())

  return model 

def discriminator_loss(logits_real, logits_fake):
  """
  Computes the discriminator loss described above.
  
  Inputs:
  - logits_real: PyTorch Tensor of shape (N,) giving scores for the real data.
  - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
  
  Returns:
  - loss: PyTorch Tensor containing (scalar) the loss for the discriminator.
  """

  # Implementing discriminator_loss.
  bce_with_logits = nn.BCEWithLogitsLoss()
  t_lab = torch.ones_like(logits_real, device = logits_real.device)
  f_lab = torch.zeros_like(logits_fake, device = logits_fake.device)
  real_loss = bce_with_logits(logits_real, t_lab)
  fake_loss = bce_with_logits(logits_fake, f_lab)
  loss = (real_loss + fake_loss) / 2

  return loss

def generator_loss(logits_fake):
  """
  Computes the generator loss described above.

  Inputs:
  - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
  
  Returns:
  - loss: PyTorch Tensor containing the (scalar) loss for the generator.
  """
  # Implementing generator_loss.
  bce_with_logits = nn.BCEWithLogitsLoss()
  t_lab = torch.ones_like(logits_fake, device = logits_fake.device)
  loss = bce_with_logits(logits_fake, t_lab)

  return loss

def get_optimizer(model):
  """
  Construct and return an Adam optimizer for the model with learning rate 1e-3,
  beta1=0.5, and beta2=0.999.
  
  Input:
  - model: A PyTorch model that we want to optimize.
  
  Returns:
  - An Adam optimizer for the model with the desired hyperparameters.
  """
  # Implementing optimizer.
  optimizer = optim.Adam(model.parameters(), lr = 1e-3, betas = (0.5, 0.999))
  
  return optimizer


def ls_discriminator_loss(scores_real, scores_fake):
  """
  Compute the Least-Squares GAN loss for the discriminator.
  
  Inputs:
  - scores_real: PyTorch Tensor of shape (N,) giving scores for the real data.
  - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
  
  Outputs:
  - loss: A PyTorch Tensor containing the loss.
  """

  # Implementing ls_discriminator_loss.
  loss = 0.5 * (torch.mean((scores_real - 1) ** 2) + torch.mean(scores_fake ** 2))
  
  return loss

def ls_generator_loss(scores_fake):
  """
  Computes the Least-Squares GAN loss for the generator.
  
  Inputs:
  - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
  
  Outputs:
  - loss: A PyTorch Tensor containing the loss.
  """
  # Implementing ls_generator_loss.
  loss = 0.5 * torch.mean((scores_fake - 1) ** 2)
  
  return loss
