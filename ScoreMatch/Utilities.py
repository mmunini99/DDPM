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
    def grid_search(d, x_start=0.01, x_end=1.1, num_points=1000):
        x_grid = jnp.linspace(x_start, x_end, num_points)
        values = jnp.array([eq(x) for x in x_grid])
        best_x = x_grid[jnp.argmin(jnp.abs(values))]  # Find x where eq(x) is closest to 0
        return best_x
    gamma = grid_search(d)
    if jnp.abs(eq(gamma)) > 1e-2:
        raise CustomError("The solver for finding the noise scale did not work. Please set noise_scale with a custom value.")
    
    return gamma

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
    def grid_search(x_start=1e-7, x_end=1e-2, num_points=10000):
        x_grid = jnp.linspace(x_start, x_end, num_points)
        values = jnp.array([eq(x) for x in x_grid])
        best_x = x_grid[jnp.argmin(jnp.abs(values))]  # Find x where eq(x) is closest to 0
        return best_x
    eps = grid_search()
    print(eq(eps))
    # if jnp.abs(eq(eps)) > 1e-2:
    #     raise CustomError("The solver for finding the noise scale did not work. Please set eps with a custom value.")
    
    return eps


def sequence_sigma(sigma1, sigmaL, length):
    '''
    It define the geometric sequence of sigmas, given the first, the last sigma and the length
    '''
    list_sigma = jnp.exp(jnp.linspace(jnp.log(sigma1), jnp.log(sigmaL), length)).astype(jnp.float32)

    return jnp.array(list_sigma)


def normal_with_mean_and_stddev(mean=1.0, stddev=0.2):
    '''
    Define, as in the original paper, the normalizer from N(1,0.2) --> not implemented in Flax
    '''
    def initializer(key, shape, dtype=jnp.float32):
        return random.normal(key, shape, dtype) * stddev + mean
    return initializer


def loss_function_denoise(pred, noise_applied, sigma):
    # unique sigma
    unique_sigmas = jnp.unique(sigma)
    list_loss = []

    # groupby operation --> loss for each sigma
    for sg in unique_sigmas:
        # index of batches to select
        idx = jnp.where(sigma == sg)[0]

        # get relative pred and noise applied
        pred_select = pred[idx]
        pred_noise = noise_applied[idx]

        # compute loss
        loss = (sg**2)*(((pred_select+pred_noise/sg)**2).sum(axis=(1, 2, 3)))

        mean_loss = jnp.mean(loss)        

        list_loss.append(mean_loss)

    

    loss = 0.5*jnp.mean(jnp.array(list_loss))

    return loss


def loss_function_explicit(jacobian, score):
    loss_batch =  jnp.trace(jacobian) + 0.5*jnp.linalg.norm(score)

    return jnp.mean(loss_batch)



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
    


    



if __name__ == "__main__":
    import jax
    import jax.numpy as jnp
    from jax import random 
    import os
    import sys
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    # Add the parent directory to the Python path
    sys.path.append(parent_dir)
    from Dataset import load_data
    l = sequence_sigma(1,0.1,10)
    train_data, _ = load_data(os.path.abspath("C:/Users/matte/Documents/JAX Tutorial/NCSN/datset_MNIST/"), 32, 32, 32, False, 32*10)
    
    val = init_first_sigma(train_data)
    print(val)
    gamma = init_ratio_sigma(32,32)
    print(gamma)
    eps = config_eps_Langevin(1000, 0.01, gamma)
    print(eps)

    


    