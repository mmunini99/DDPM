import jax.numpy as jnp
from flax import linen as nn

class ConvBlock(nn.Module):
    '''
    Define the block for the beginning and the ending of the ResNet architecture.
    Basic kernel (3, 3)
    '''
    channels: int

    def setup(self):
        self.conv = nn.Conv(self.channels, (3,3), padding="SAME")

    def __call__(self, input):
        x = self.conv(input)

        return x
    

class DownConvPoolBlock(nn.Module):
    '''
    ConvBlock followed by avg pooliung for downsampling
    '''
    channels: int

    def setup(self):
        self.conv = nn.Conv(self.channels, (3,3), padding="SAME")
        self.pool = nn.avg_pool

    def __call__(self, input):
        x = self.conv(input)
        x = self.pool(x, window_shape=(2,2), strides=(2,2), padding='VALID')

        return x


class AtrousConvBlock(nn.Module):
    '''
    Define the block for the convolution layer with dilation/atrous.
    Basic kernel (3, 3)
    '''
    channels: int
    dil_int: int
    stride_int : int

    def setup(self):
        self.conv = nn.Conv(self.channels, (3,3), kernel_dilation=self.dil_int, padding="SAME", strides=(self.stride_int, self.stride_int)) # should be self.dil_int the padding

    def __call__(self, input):
        x = self.conv(input)

        return x    

class ResNetBlock(nn.Module):
    '''
    Building block of the ResNet
    Differnet configuration if it is used to downsample or keep dimensions as input one.
    Different configuration based on dilation or not in the convolution block
    To sum up --> 4 different blocks
    ''' 
    channels: int
    downsample: bool
    dil_flag: bool
    dil_int : int
    training : bool

    def setup(self):
        self.norm1 = nn.BatchNorm(use_running_average=not self.training)
        self.norm2 = nn.BatchNorm(use_running_average=not self.training)
        if self.downsample and self.dil_flag: # block for downsampling block with dilation
            self.conv1 = AtrousConvBlock(self.channels, self.dil_int, 2)
            self.conv2 = AtrousConvBlock(self.channels, self.dil_int, 1)
            self.down_x = AtrousConvBlock(self.channels, self.dil_int, 2)
        elif self.downsample and not self.dil_flag: # block for downsampling block without dilation
            self.conv1 = ConvBlock(self.channels)
            self.conv2 = DownConvPoolBlock(self.channels)
            self.down_x = DownConvPoolBlock(self.channels)
        elif not self.downsample and self.dil_flag: # block for dilation only
            self.conv1 = AtrousConvBlock(self.channels, self.dil_int, 1)
            self.conv2 = AtrousConvBlock(self.channels, self.dil_int, 1)   
        else: # normal ResNet CNN block 
            self.conv1 = ConvBlock(self.channels)
            self.conv2 = ConvBlock(self.channels) 

    def __call__(self, input):
        if self.downsample and self.dil_flag: # block for downsampling block with dilation
            z = self.down_x(input)

            x = self.norm1(input)
            x = nn.elu(x)
            x = self.conv1(x)
            x = self.norm2(x)
            x = nn.elu(x)
            x = self.conv2(x)

            return z + x # here we have reshaped the input for the new downsized dimension and larger feature channels

        elif self.downsample and not self.dil_flag: # block for downsampling block without dilation
            z = self.down_x(input)

            x = self.norm1(input)
            x = nn.elu(x)
            x = self.conv1(x)
            x = self.norm2(x)
            x = nn.elu(x)
            x = self.conv2(x)

            return z + x # here we have reshaped the input for the new downsized dimension and larger feature channels

        elif not self.downsample and self.dil_flag: # block for dilation only
            x = self.norm1(input)
            x = nn.elu(x)
            x = self.conv1(x)
            x = self.norm2(x)
            x = nn.elu(x)
            x = self.conv2(x)

            return input + x 
        else: # normal ResNet CNN block 
            x = self.norm1(input)
            x = nn.elu(x)
            x = self.conv1(x)
            x = self.norm2(x)
            x = nn.elu(x)
            x = self.conv2(x)

            return input + x

    

if __name__ == "__main__":
    import os
    import jax
    import jax.numpy as jnp
    model = ResNetBlock(32, True, False, None, False)
    p = model.init(jax.random.PRNGKey(0), jnp.ones((1, 32, 32, 1)))
    res = jnp.ones((128, 32, 32, 1))
    output = model.apply(p, res)
    
    print(output.shape)