import jax.numpy as jnp
import jax.image as jimg
from flax import linen as nn
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from .Utilities import ConditionalInstanceNorm2d as InstNorm


### REFINE NET BLOCK SECTION ###

# Residual Convolutional Unit --> RCU
class RCU(nn.Module):
    """
    Elu --> 3x3 Conv --> Elu --> Conv
    Resiudual Layer take as input --> like a Pre-activation of the ResNet block.
    """
    channels : int 

    def setup(self):
        self.conv1 = nn.Conv(self.channels, (3,3), padding="SAME")
        self.conv2 = nn.Conv(self.channels, (3,3), padding="SAME")

        self.inst_n1 = InstNorm()
        self.inst_n2 = InstNorm()
        
    def __call__(self, input): # here the input is the same passed from the resnet level
        res = input # for identity mapping
        x = nn.elu(input)
        x = self.inst_n1(input)
        x = self.conv1(x)        
        x = nn.elu(x)
        x = self.inst_n2(x)
        x = self.conv2(x)  
        return x + res # the output here is (batch_size, height, width, channel)


# Multi-Resolution Fusion
class MRF(nn.Module):
    '''
    Take input from the ResNet and the RefineNet (if it is not in the last section)
    Up-Sampling for ResNet in terms of channels
    Up-Sampling for RefineNet in terms of height and width
    ''' 

    channels_res : int 
    ref_presence : bool 


    def setup(self):
        # adapt the number of channels in RefineNet block(if present) to the one in ResNet --> feature maps of the same channels (the smallest one, so the one of ResNet block) 
        self.conv_ref = nn.Conv(self.channels_res, (3,3), padding="SAME")
        self.conv_res = nn.Conv(self.channels_res, (3,3), padding="SAME")     
        # normalizer
        self.inst_ref = InstNorm()
        self.inst_res = InstNorm() 
    
    def __call__(self, input):
        img_res, img_ref = input
        y = self.inst_res(img_res)
        y = self.conv_res(y)
        if not self.ref_presence:
            x = self.inst_ref(img_ref)
            x = self.conv_ref(x)
            x = jimg.resize(x, y.shape, method='bilinear')

            return x+y
        else:
            return y


# Chained Residual Pooling
class CRP(nn.Module):
    '''
    ResNet + Pooling
    '''
    channels : int

    def setup(self):
        # define the Convolutional layers : 3 layers
        self.conv1 = nn.Conv(self.channels, (3,3), padding="SAME")
        self.conv2 = nn.Conv(self.channels, (3,3), padding="SAME")  
        self.conv3 = nn.Conv(self.channels, (3,3), padding="SAME")     
        # normalizer
        self.inst1 = InstNorm()
        self.inst2 = InstNorm() 
        self.inst3 = InstNorm() 
        # pooling layer 5x5 : 3 layers
        self.max_pool1 = nn.max_pool
        self.max_pool2 = nn.max_pool
        self.max_pool3 = nn.max_pool

    def __call__(self, input):
        # first Residual Connection
        x = nn.elu(input)
        x = self.max_pool1(x, window_shape=(5, 5), strides=(1, 1), padding='SAME')
        x = self.inst1(x)
        x = self.conv1(x)
        # second Residual Connection
        y = self.max_pool1(x, window_shape=(5, 5), strides=(1, 1), padding='SAME')
        y = self.inst1(y)
        y = self.conv1(y)
        # third Residual Connection
        z = self.max_pool1(y, window_shape=(5, 5), strides=(1, 1), padding='SAME')
        z = self.inst1(z)
        z = self.conv1(z)
        # final 
        return x + y + z


class RefineNet(nn.Module):

    channels_res: int
    channels_ref: int
    flag_last_block : bool

    def setup(self):
        # define the required layer conditioned if the block is teh last one or not
        if self.flag_last_block: 
            # first 2 blocks for ResNet output 
            self.cru1 = RCU(self.channels_res)
            self.cru2 = RCU(self.channels_res)
            # last RCU block for output
            self.cru3 = RCU(self.channels_res)
            # define MRF block
            self.mrf = MRF(self.channels_res, self.flag_last_block)
            # define CRP block
            self.crp = CRP(self.channels_res)
        else:
            # first 2 blocks for both ResNet output and previous RefineNet blocks
            self.cru1_res = RCU(self.channels_res)
            self.cru2_res = RCU(self.channels_res)
            self.cru1_ref = RCU(self.channels_ref)
            self.cru2_ref = RCU(self.channels_ref)
            # last RCU block for output
            self.cru3 = RCU(self.channels_res)
            # define MRF block
            self.mrf = MRF(self.channels_res, self.flag_last_block)
            # define CRP block
            self.crp = CRP(self.channels_res)

    def __call__(self, inputs):
        if self.flag_last_block: 
            img_res, _ = inputs
            # adaptive convolution
            x = self.cru1(img_res)
            x = self.cru2(x)
            # multi resolution fusion
            x = self.mrf([x, _])
            # chained residual pooling
            x = self.crp(x)
            # final adaptive convolution
            x = self.cru3(x)

            return x

        else:
            img_res, img_ref = inputs
            # adaptive convolution
            x = self.cru1_res(img_res)
            x = self.cru2_res(x)
            y = self.cru1_ref(img_ref)
            y = self.cru2_ref(y)
            # multi resolution fusion
            z = self.mrf([x, y])
            # chained residual pooling
            z = self.crp(z)
            # final adaptive convolution
            z = self.cru3(z)

            return z


if __name__ == "__main__":
    import os
    from Dataset import load_data
    import jax
    import jax.numpy as jnp
    train_data, _ = load_data(os.path.abspath("C:/Users/matte/Documents/JAX Tutorial/NCSN/datset_MNIST/"), 32, 32, 32, False, 32)
    batch = next(iter(train_data))
    model = RefineNet(16, None, True)
    p = model.init(jax.random.PRNGKey(0), [jnp.ones((1, 32, 32, 16)),jnp.ones((1, 16, 16, 32))])
    res = jnp.ones((128, 32, 32, 16))
    ref = jnp.ones((128, 16, 16, 32))
    output = model.apply(p, [res, None])
    
    print(output.shape)
































