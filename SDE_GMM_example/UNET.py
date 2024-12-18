import jax
import jax.numpy as jnp


import flax
from flax import linen as nn
from flax.linen import initializers


from Utilities import PositionalEmbedding, ResNetBlock



class UNet(nn.Module):
  dim: int 
  dim_scale_factor: tuple 
  out: int


  @nn.compact # simpler since there are for loop
  def __call__(self, inputs, training):
    input, time = inputs

    x = nn.Dense(features=self.dim, kernel_init=initializers.xavier_uniform())(input)
    x = nn.BatchNorm(use_running_average=not training)(x)
    x = nn.relu(x)
    time_emb = PositionalEmbedding(self.dim, self.dim*4)(time) # here the number of embeddings is choosen from empirical performance (it could be added as hyper-parameter)
    
    dims = [self.dim * i for i in self.dim_scale_factor]
    pre_downsampling = []
    
    # Downsampling phase : (batch_size, height, width, chan) x2 --> (batch_size, height/2, width/2, chan*2) x2 --> (batch_size, height/4, width/4, chan*4) x2 --> (batch_size, height/8, width/8, chan*8) x2  
    for index, dim in enumerate(dims):
      x = ResNetBlock(dim, dim)(x, time_emb)
      x = nn.BatchNorm(use_running_average=not training)(x)
      x = ResNetBlock(dim, dim)(x, time_emb)
      # Saving this output for residual connection with the upsampling layer
      pre_downsampling.append(x)
      if index != len(dims) - 1:
        x = nn.Dense(features=dim//2, kernel_init=initializers.xavier_uniform())(x)
    # Middle block
    x = ResNetBlock(dims[-1], dims[-1])(x, time_emb)
    x = nn.BatchNorm(use_running_average=not training)(x)
    x = ResNetBlock(dims[-1], dims[-1])(x, time_emb) # shape here --> (batch_size, height/8, width/8, chan*8)
    
    # Upsampling phase
    for index, dim in enumerate(reversed(dims)):
      x = jnp.concatenate([pre_downsampling.pop(), x], -1)
      x = ResNetBlock(dim, dim)(x, time_emb)
      x = nn.BatchNorm(use_running_average=not training)(x)
      x = ResNetBlock(dim, dim)(x, time_emb)
      if index != len(dims) - 1:  # if it is not the first block of the "decoder"/"upsampling block" then start to increase height and width by decreasing channels. Opposite of downsampling operations
        x =  nn.Dense(features=int(dim*2), kernel_init=initializers.xavier_uniform())(x)

    x = nn.BatchNorm(use_running_average=not training)(x)    
    x = nn.Dense(features=self.out, kernel_init=initializers.xavier_uniform())(x)

    return x


if __name__== "__main__":
  from Dataset import build_dataset
  n_rows = 64  # Total number of rows in the dataset
  n_samples_per_row = 400  # Samples per row (from GMM)
  means = [0, 5, -5]  # Means of the Gaussian components
  variances = [1, 0.5, 0.78]  # Variances of the components
  weights = [0.4, 0.4, 0.2]  # Mixing coefficients (must sum to 1)
  key = jax.random.PRNGKey(0)
  # Generate the dataset
  dataset, mean, var  = build_dataset(n_rows, n_samples_per_row, means, variances, weights, 32)
  model = UNet(32, (4,2))
  i_t = jnp.ones(1)
  time = jax.random.normal(key, 1)
  i_d = jnp.ones((1, 400))
  data = next(iter(dataset))[0][None, :]
  print(data.shape)
  variables = model.init(key, [i_d, i_t], training=True)
  o, _ = model.apply(variables, [data, time], training=True, mutable=['batch_stats'])
  print(o.shape)


