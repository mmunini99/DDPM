import jax.numpy as jnp
from flax import linen as nn

from RefineNet import RefineNet as RFN
from ResNet import ResNetBlock as RSB
from ResNet import ConvBlock as CB
from Utilities import ConditionalInstanceNorm2d as InstNorm


class NCSN(nn.Module):
    '''
    Architecture used as Score Matching Neural Network.
    This is 4-cascade RefineNet so, pass a tuple of length 4
    '''
    channels_init: int
    seq_dim_channels : tuple
    channels_out: int
    training: bool

    def setup(self):
        self.conv_ini = CB(self.channels_init)
        self.conv_end = CB(self.channels_out)
        self.norm = InstNorm()
        ### RES NET ###
        # First block --> 2 ResNet block without dilation and downsampling procedure
        self.resnet_11 = RSB(self.channels_init*self.seq_dim_channels[0], False, False, None, self.training)
        self.resnet_12 = RSB(self.channels_init*self.seq_dim_channels[0], False, False, None, self.training)
        # Secodn Block --> 2 ResNet block without dilation procedure, but with one downsampling block
        self.resnet_21 = RSB(self.channels_init*self.seq_dim_channels[1], True, False, None, self.training)
        self.resnet_22 = RSB(self.channels_init*self.seq_dim_channels[1], False, False, None, self.training)
        # Third Block --> 2 ResNet block with one downsampling procedure and two dilation blocks
        self.resnet_31 = RSB(self.channels_init*self.seq_dim_channels[2], True, True, 2, self.training)
        self.resnet_32 = RSB(self.channels_init*self.seq_dim_channels[2], False, True, 2, self.training)
        # Fourth Block --> 2 ResNet block without one downsampling procedure and two dilation blocks
        self.resnet_41 = RSB(self.channels_init*self.seq_dim_channels[3], True, True, 4, self.training)
        self.resnet_42 = RSB(self.channels_init*self.seq_dim_channels[3], False, True, 4, self.training)
        ### REFINE NET ###
        # Refine Block for 4th ResNet block --> No connection with previous Refine block
        self.refinet_4 = RFN(self.channels_init*self.seq_dim_channels[3], None, True)
        # Refine Block for 3rd ResNet block --> Connection with 4th Refine block
        self.refinet_3 = RFN(self.channels_init*self.seq_dim_channels[2], self.channels_init*self.seq_dim_channels[3], False)
        # Refine Block for 2nd ResNet block --> Connection with 3rd Refine block
        self.refinet_2 = RFN(self.channels_init*self.seq_dim_channels[1], self.channels_init*self.seq_dim_channels[2], False)
        # Refine Block for 1st ResNet block --> Connection with 2nd Refine block
        self.refinet_1 = RFN(self.channels_init*self.seq_dim_channels[0], self.channels_init*self.seq_dim_channels[1], False)        
        
        



    def __call__(self, inputs):
        input, sigma = inputs
        # INIT #
        x = self.conv_ini(input) # expected output --> (batch_size, height_image, width_image, channels_init)
        # RES NET BLOCK #
        x11 = self.resnet_11(x)
        x12 = self.resnet_12(x11) # expected output --> (batch_size, height_image, width_image, channels_init*seq_dim_channels[0])
        x21 = self.resnet_21(x12)
        x22 = self.resnet_22(x21) # expected output --> (batch_size, height_image/2, width_image/2, channels_init*seq_dim_channels[1])
        x31 = self.resnet_31(x22)
        x32 = self.resnet_32(x31) # expected output --> (batch_size, height_image/4, width_image/4, channels_init*seq_dim_channels[2])
        x41 = self.resnet_41(x32)
        x42 = self.resnet_42(x41) # expected output --> (batch_size, height_image/8, width_image/8, channels_init*seq_dim_channels[3])
        # REFINE NET #
        y4 = self.refinet_4([x42, None]) # expected output --> (batch_size, height_image/8, width_image/8, channels_init*seq_dim_channels[3])
        y3 = self.refinet_3([x32, y4]) # expected output --> (batch_size, height_image/4, width_image/4, channels_init*seq_dim_channels[2])
        y2 = self.refinet_2([x22, y3]) # expected output --> (batch_size, height_image/2, width_image/2, channels_init*seq_dim_channels[1])
        y1 = self.refinet_1([x12, y2]) # expected output --> (batch_size, height_image, width_image, channels_init*seq_dim_channels[0])
        # CLOSE #
        z = self.norm(y1)
        z = nn.elu(z)
        z = self.conv_end(z) # reshape with initial channels

        out = z/sigma # NCSN without noise conditioning --> Technique 3 (see NCSN paper 10/23/2020)

        return z
        

if __name__ == "__main__":
    import os
    from Dataset import load_data
    import jax
    import jax.numpy as jnp
    train_data, _ = load_data(os.path.abspath("C:/Users/matte/Documents/JAX Tutorial/NCSN/datset_MNIST/"), 32, 32, 32, False, 32)
    batch = next(iter(train_data))
    model = NCSN(128, (1, 2, 4, 8), 1, True)
    p = model.init(jax.random.PRNGKey(0), [jnp.ones((1, 32, 32, 1)), 1])
    res = jnp.ones((128, 32, 32, 1))
    output = model.apply(p, [res, 0.25], mutable=['batch_stats'])
    out, _ = output
    
    print(out.shape)