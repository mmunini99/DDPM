import math
import numpy as np
from typing import Callable

import jax
import jax.numpy as jnp
from jax import random

import flax
from flax import linen as nn
from flax.linen import initializers

from optax.losses import squared_error


# define the loss function simpliefied of DDPM
def loss_function(pred, true):
    loss = jnp.mean(squared_error(pred, true))
    return loss


# define the loss function indicated in the SDE paper
def loss_function_sde(score, marg, const):
    loss = (1/const)*jnp.sum((score-marg)**2)



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


    
# Definition of the ResNet that will be the basic component of the U-Net --> sligthly change because here we allows as extra data the Time Embedding of data, that would not be normally considered in the usual U-Net
class ResNetBlock(nn.Module):
    dim : int 
    out : int

    def setup(self):
        self.layer1= nn.Dense(features=self.dim, kernel_init=initializers.xavier_uniform())
        self.layer2 = nn.Dense(features=self.out, kernel_init=initializers.xavier_uniform())
        self.time_emb_projection = nn.Dense(features=self.dim)
        self.layer3 = nn.Dense(features=self.out, kernel_init=initializers.xavier_uniform())
        
        
    def __call__(self, inputs, time_embed = None):
        x = self.layer1(inputs)
        
        if time_embed is not None:
            # broad casting operation --> done with FFN projection
            emb = self.time_emb_projection(time_embed)
            # add time info to data features
            x = x + emb
        
        x = self.layer2(x)
        res_conv = self.layer3(inputs) # this is the original data augmented in terms of channels.
        return x + res_conv 
    



if __name__ == '__main__':
    key = jax.random.PRNGKey(0)
    model = PositionalEmbedding(dimension=128, expand_dimension=128*4)
    inputs = jnp.ones(1)
    inputs2 = jax.random.normal(key, 32)
    inputs3 = jax.random.normal(key, (1, 400))
    inputs4 = jax.random.normal(key, (32, 400))
    i = jax.random.normal(key, (1, 512))
    params = model.init(key, inputs)
    output = model.apply(params, inputs2) 
    print(output.shape)
    net = ResNetBlock(32, 64)
    p = net.init(key, inputs3, i)
    o = net.apply(p, inputs4, output)
    print(o.shape)

