import math
import numpy as np
from typing import Callable

import jax
import jax.numpy as jnp
from jax import random

import flax
from flax import linen as nn

from optax.losses import squared_error


# define the loss function
def loss_function(pred, true):
    loss = jnp.mean(squared_error(pred, true))
    return loss



# we want to pass the time, so the step in the forward process, that goes to 0 (input image) to T(in theory a WN iamge)
# there are a lot of methods, one is sinusoidal embedding
class PositionalEmbedding(nn.Module):
    dimension : int
    expand_dimension : int 

    @nn.compact
    def __call__(self, inputs):
        
        half_dim = self.dimension//2
        emb1 = math.log(10000)/(half_dim-1)
        emb2 = jnp.exp(jnp.arange(half_dim)*(-emb1))
        emb = inputs[:, None]*emb2[None, :]
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], -1)
        x = nn.Dense(features=self.expand_dimension)(emb)
        x = nn.gelu(x)
        x = nn.Dense(features=self.expand_dimension)(x)
        x = nn.silu(x)

        return x # given the current settings it will be a (batch_size, expand_dimension) matrix --> then in the ResNet block there will be Feed Forward layer that will change the number of columsn to match the number of channels at each down/up sample phase in the U-Net



# Define the multiplicative Attention block --> Multi Head version (Transformer usual)
class Attention(nn.Module):
    dim: int
    num_heads: int
    use_bias: bool = False
    kernel_init: Callable = nn.initializers.xavier_uniform()

    def setup(self):
        # we define the base for equal-size Feed Forward output for the Query, Key and Value tensors
        self.linear1 = nn.Dense(features=self.dim*3, use_bias=self.use_bias, kernel_init=self.kernel_init)
        self.linear2 = nn.Dense(features=self.dim, use_bias=self.use_bias, kernel_init=self.kernel_init)

    def __call__(self, inputs):
        batch, height, width, channels = inputs.shape
        inputs = inputs.reshape(batch, height*width, channels) # reduce the shape of the tensor from 4 to 3 --> height * width = 
        batch, n, channels = inputs.shape
        scale = jnp.sqrt(self.dim // self.num_heads) # scale factor to improve stability of the gradient sqrt(d_k) in original paper
        qkv = self.linear1(inputs) # the output here is (batch_size, height*width, dim*3)
        qkv = jnp.reshape(qkv, (batch, n, 3, self.num_heads, channels // self.num_heads)) # from (batch, n, 3 times dimension of input tensor) to (batch, n, 3, self.num_heads, channels // self.num_heads)
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4)) # from (batch, n, 3, self.num_heads, channels // self.num_heads) to (3, batch, num_heads, n, dim // num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2] # define Query, Key and Value tensors --> each one with shape (batch_size, num_heads, height*width, channels//num_heads)
        # QV^T part
        attention = (q @ jnp.swapaxes(k, -2, -1)) * scale # this is the dot product between Query and the transpose of Value --> (batch_size, num_heads, height*width, height*width)
        attention = nn.softmax(attention, axis=-1) #force output to "probability matrix"
        # V weighted by probabilities 
        x = (attention @ v).swapaxes(1, 2).reshape(batch, n, channels) # dot product that makes possible the weighting --> plus it is reshaped to (batch, n, channels) --> (batch_size, height*width, channels)
        x = self.linear2(x) # shape --> (batch_size, height*width, channels*3)
        x = jnp.reshape(x, (batch, int(x.shape[1]** 0.5), int(x.shape[1]** 0.5), -1)) # reshape to (batch, height, width, channels*3)
        return x



# Define the block for the U-Net --> Not used here
class UNetBlock(nn.Module):
    channels : int 

    def __setup__(self):
        self.conv1 = nn.Conv(self.channels, kernel_size=(3, 3), padding="SAME")
        self.conv2 = nn.Conv(self.channels, kernel_size=(3, 3), padding="SAME")

    def __call__(self, inputs):
        x = self.conv1(x)
        x = nn.relu(x)
        x = self.conv2(x)
        x = nn.relu(x)

        return x



# CNN layer, both used in the U-Net for Up-Sampling and Down-Sampling. In addition it has a normalization and activation steps. It is customized for an Attention layer
class Block(nn.Module):
    channels : int 
    groups : int 

    def setup(self):
        self.conv = nn.Conv(self.channels, kernel_size=(3,3))
        self.g_norm = nn.GroupNorm(num_groups=self.groups)

    def __call__(self, inputs):
        conv = self.conv(inputs)
        norm = self.g_norm(conv) # here, differently from a normal implementation of the ResNet we allow for each head of the Transformer
        activation = nn.silu(norm)
        return activation # the shape of the tensor, using channels = 32 and groups = 8 is (batch_size, height, width, channel)
    
# Definition of the ResNet that will be the basic component of the U-Net --> sligthly change because here we allows as extra data the Time Embedding of data, that would not be normally considered in the usual U-Net
class ResNetBlock(nn.Module):
    """
    Case 1: If there is no time embedding provided as extra data, then it is a normal ResNet, where the CNN block describes the residual of the previous data.
    Case 2: Otherwise
    """
    channels : int 
    groups : int 

    def setup(self):
        self.block1 = Block(self.channels, self.groups)
        self.block2 = Block(self.channels, self.groups)
        self.conv = nn.Conv(self.channels, (1,1), padding="SAME")
        self.time_emb_projection = nn.Dense(self.channels)
        
        
    def __call__(self, inputs, time_embed = None):
        x = self.block1(inputs)
        
        if time_embed is not None:
            # broad casting operation --> done with FFN projection
            emb = self.time_emb_projection(time_embed)
            emb = emb[:, None, None, :]  # Shape: (batch_size, 1, 1, channels) --> here the batch_size means the sequence of timestep that are passed to the model
            emb = jnp.broadcast_to(emb, x.shape)  # Broadcast to (batch_size, height, width, channels)
            # add time info to image features
            x = x + emb
        
        x = self.block2(x)
        res_conv = self.conv(inputs) # this is the original data augmented in terms of channels.
        return x + res_conv # the output here is (batch_size, height, width, channel)