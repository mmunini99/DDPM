import jax.numpy as jnp
from jax import random, vmap
from scipy.stats import norm as Norm
from scipy.optimize import fsolve
from flax import linen as nn

def init_first_sigma(train_dataloader):
    '''
    Technique 1 (see NCSN paper 10/23/2020)
    sigma1 chosen by max L2 distance between all pairs of training images
    '''
    # convert batch data in data loader to jnp array
    list_data = [batch for batch in train_dataloader]
    list_data = jnp.array([item for sublist in list_data for item in sublist]) # remove batch grouping
    # define flatting operation function
    flat_oper = lambda img: img.reshape(-1)
    vec_flatting = vmap(flat_oper) # vectorize
    flat_data = vec_flatting(list_data) # matrix (n° data, height*width*channels)
    # change dimension for difference broadcasting operation
    diff = flat_data[:, None, :]-flat_data[None, :, :] # tensor shape (n° data, n° data, height*width*channels)
    # compute L2
    l2 = jnp.sqrt(jnp.sum(diff ** 2, axis=-1))
    # trick: fill with -inf the diagonal to avoid to pick self-pair element
    l2 = jnp.fill_diagonal(l2, -jnp.inf, inplace=False)

    return jnp.max(l2) # retrun the maximum value


def init_ratio_sigma(h, w):
    '''
    Technique 2 (see NCSN paper 10/23/2020)
    Common ratio gamma chosen by solving equation
    '''
    # define the space dimension of images
    d = h*w 
    # define the equation to solve
    def eq(x):
        arg1 = jnp.sqrt(2*d)*(x-1)+3*x
        arg2 = jnp.sqrt(2*d)*(x-1)-3*x
        return Norm.cdf(arg1)-Norm.cdf(arg2)-0.5
    # get solution --> interval R_o+ since variance > 0
    solution = fsolve(eq, [1.1])
    if jnp.abs(eq(solution[0])) > 1e-2:
        raise CustomError("The solver for finding the noise scale did not work. Please set noise_scale with a custom value.")
    
    return solution[0]

def config_eps_Langevin(T, last_sigma, gamma):
    '''
    Technique 4 (see NCSN paper 10/23/2020)
    Once chosen T (number of steps) find eps by solving equation
    '''
    # define the equation to solve
    s = last_sigma**2
    def eq(x):
        arg1 = (1-x/(s))**(2*T)
        arg2 = (gamma**2-(2*x)/((s-s*(1-x/s)**2)))
        arg3 = -(2*x)/((s-s*(1-x/s)**2))
        return arg1*arg2+arg3-1.0
    # get solution --> interval R_o+ since variance > 0
    solution = fsolve(eq, [1e-6])
    if jnp.abs(eq(solution[0])) > 1e-2:
        raise CustomError("The solver for finding the noise scale did not work. Please set eps with a custom value.")
    
    return solution[0]


def sequence_sigma(sigma1, sigmaL, length):
    '''
    It define the geometric sequence of sigmas, given the first, the last sigma and the length
    '''
    list_sigma = jnp.exp(jnp.linspace(jnp.log(sigma1), jnp.log(sigmaL), length)).astype(jnp.float32)

    return list_sigma


def normal_with_mean_and_stddev(mean=1.0, stddev=0.2):
    '''
    Define, as in the original paper, the normalizer from N(1,0.2) --> not implemented in Flax
    '''
    def initializer(key, shape, dtype=jnp.float32):
        return random.normal(key, shape, dtype) * stddev + mean
    return initializer


class CustomError(Exception):
    pass

class InstanceNorm2D(nn.Module):
    '''
    Instance 2D normalization --> compute mean and variance along dimension of height and width for each channel independently.
    Normalization technique for each channel matrix with the related mean and variance
    image in format --> (batch_size, height, width, channels)
    '''
    eps : float = 1e-5

    @nn.compact
    def __call__(self, image):
        # define the two learnable parameter vectors --> same dimensions of the channels
        gamma = self.param('gamma_inst_norm', normal_with_mean_and_stddev(mean=1.0, stddev=0.2), (1,1,1, image.shape[-1])) # Flax method to define a trainable param specifing the shape of it --> for broadcasting (1,1,1, channels)
        beta = self.param('beta_inst_norm', nn.initializers.zeros, (1,1,1, image.shape[-1])) # Flax method to define a trainable param specifing the shape of it --> for broadcasting (1,1,1, channels)
        # define the mean and the std on height and width for every channel singularly
        mean = jnp.mean(image, axis=(1,2), keepdims= True) # mean for each channel --> shape (batch_size, 1, 1, channels) for broadcasting the presence of (1,1)
        sd = jnp.std(image, axis=(1,2), keepdims= True, ddof=1) # mean for each channel --> shape (batch_size, 1, 1, channels) for broadcasting the presence of (1,1). ddof = 1 ensures unbiased estimator
        # define the normalized iamge
        out = beta + ((image-mean)/(jnp.sqrt(self.eps+sd**2)))*gamma

        return out # shape same of image


class ConditionalInstanceNorm2d(nn.Module):
    '''
    Conditional instance 2d normalization --> to avoid to remove information from channels

    Attention --> this formulation is defined conditioned on the application of technique 3 (see NCSN paper 10/23/2020) --> n° classes = 1
    '''
    eps : float = 1e-5

    @nn.compact
    def __call__(self,image):
        # call the instance normalizer fro 2D image and apply it
        img_norm = InstanceNorm2D()(image)
        # define another learnable parameter vector --> same dimensions of the channels
        alpha = self.param('alpha_inst_norm', normal_with_mean_and_stddev(mean=1.0, stddev=0.2), (1,1,1, image.shape[-1])) # Flax method to define a trainable param specifing the shape of it --> for broadcasting (1,1,1, channels)
        # define the mean on height and width for every channel --> shape (batch_size, 1, 1, channels) for broadcasting the presence of (1,1)
        mean = jnp.mean(image, axis=(1,2), keepdims= True)
        # define the mean and variance of the mean  height and width for every channel --> overall of all channels
        mean_scalar = jnp.mean(mean, axis=-1, keepdims=True) # --> scalar
        sd_scalar = jnp.std(mean, axis=-1, keepdims=True, ddof=1) # --> scalar
        # define the conditional normalized image
        out = ((mean-mean_scalar)/(jnp.sqrt(self.eps+sd_scalar**2)))*alpha + img_norm

        return out # shape same of image
    


    



# if __name__ == "__main__":
#     import os
#     from Dataset import load_data
#     import jax
#     import jax.numpy as jnp
#     train_data, _ = load_data(os.path.abspath("C:/Users/matte/Documents/JAX Tutorial/NCSN/datset_MNIST/"), 32, 32, 32, False, 32)
#     batch = next(iter(train_data))
#     model = ConditionalInstanceNorm2d()
#     p = model.init(jax.random.PRNGKey(0), jnp.ones((1, 32, 32, 16)))
#     res = jnp.ones((128, 32, 32, 16))
#     ref = jnp.ones((128, 16, 16, 32))
#     output = model.apply(p, res)
    
#     print(output.shape)

    


    