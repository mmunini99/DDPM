import jax
import jax.numpy as jnp


import flax
from flax import linen as nn


from DiffModel.Utilities import PositionalEmbedding, ResNetBlock, Attention



class UNet(nn.Module):
  dim: int 
  dim_scale_factor: tuple = (1, 2, 4, 8)
  num_groups: int = 8


  @nn.compact # simpler since there are for loop
  def __call__(self, inputs):
    inputs_data, time = inputs
    channels = inputs_data.shape[-1]
    x = nn.Conv(self.dim, kernel_size=(3,3), padding="SAME")(inputs_data)
    x = nn.relu(x)
    time_emb = PositionalEmbedding(self.dim, self.dim*4)(time) # here the number of embeddings is choosen from empirical performance (it could be added as hyper-parameter)
    
    dims = [self.dim * i for i in self.dim_scale_factor]
    pre_downsampling = []
    
    # Downsampling phase : (batch_size, height, width, chan) x2 --> (batch_size, height/2, width/2, chan*2) x2 --> (batch_size, height/4, width/4, chan*4) x2 --> (batch_size, height/8, width/8, chan*8) x2  
    for index, dim in enumerate(dims):
      x = ResNetBlock(dim, self.num_groups)(x, time_emb)
      x = ResNetBlock(dim, self.num_groups)(x, time_emb)
      att = Attention(dim, self.num_groups)(x)
      norm = nn.GroupNorm(self.num_groups)(att)
      x = norm + x
      # Saving this output for residual connection with the upsampling layer
      pre_downsampling.append(x)
      if index != len(dims) - 1:
        x = nn.Conv(dim, kernel_size=(3,3), strides=(2,2))(x)
    
    # Middle block
    x = ResNetBlock(dims[-1], self.num_groups)(x, time_emb)
    att = Attention(dim, self.num_groups)(x)
    norm = nn.GroupNorm(self.num_groups)(att)
    x = norm + x 
    x = ResNetBlock(dims[-1], self.num_groups)(x, time_emb) # shape here --> (batch_size, height/8, width/8, chan*8)
    
    # Upsampling phase
    for index, dim in enumerate(reversed(dims)):
      x = jnp.concatenate([pre_downsampling.pop(), x], -1)
      x = ResNetBlock(dim, self.num_groups)(x, time_emb)
      x = ResNetBlock(dim, self.num_groups)(x, time_emb)
      att = Attention(dim, self.num_groups)(x)
      norm = nn.GroupNorm(self.num_groups)(att)
      x = norm + x
      if index != len(dims) - 1:  # if it is not the first block of the "decoder"/"upsampling block" then start to increase height and width by decreasing channels. Opposite of downsampling operations
        x = nn.ConvTranspose(dim,  kernel_size=(4,4), strides=(2,2))(x) # deconvolution layer
        


    # Final ResNet block and output convolutional layer
    x = ResNetBlock(dim, self.num_groups)(x, time_emb)
    x = nn.Conv(channels, (1,1), padding="SAME")(x) # the shape is the same of the input image

    return x

