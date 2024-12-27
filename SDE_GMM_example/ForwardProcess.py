import jax
import jax.numpy as jnp
from jax import random


class forward_process(object):
    '''
    T --> the number of times the forward process is applied to each image
    beta_lb --> lower bound for the alpha in forward process
    beta_ub --> upper bound for the alpha in forward process
    '''
    def __init__(self, N, beta_lb, beta_ub):
        self.N = N
        self.lb = beta_lb
        self.ub = beta_ub

    def __setting__(self):
        self.beta = jnp.linspace(self.lb/self.N, self.ub/self.N, self.N)
        self.alpha = 1-self.beta # defien a grid of values for the alpha, that represents the evolution in time of alpha
        self.alpha_hat = jnp.cumprod(self.alpha, 0) # make the cumlative moltiplication of the values in alpha from 0 to T
        self.alpha_mean = jnp.sqrt(self.alpha_hat) # compute the constants for the mean in the forward equations
        self.alpha_sd = jnp.sqrt(1-self.alpha_hat) # compute the constants for the variance in the forward equations

    def fp(self, rng_key, x0, idx):
        self.__setting__()
        gauss_noise = random.normal(rng_key, x0.shape) # define the noise to be applied to the data. The noise has mu = 0 and sd = 1
        param_mean = jnp.take(self.alpha_mean, idx)  # take select the alpha_mean according to the temporal step. 
        param_sd = jnp.take(self.alpha_sd, idx) # same as above --> only for variance
        # vector transpose and broadcast
        param_mean =  param_mean[:, jnp.newaxis]
        param_sd =  param_sd[:, jnp.newaxis]
        # new data : produced by DDPM forward process
        perturbed_x = x0*param_mean + gauss_noise*param_sd
        # the return is the perturbed image and the noise added at step t (this is important for the reverse process)
        return perturbed_x, gauss_noise 
    
    def get_params(self):
        '''
        Return the beta array, the alpha and the alpha hat (product) array
        '''
        self.__setting__()
        return self.beta

if __name__ == '__main__':
    from Dataset import build_dataset
    n_rows = 64  # Total number of rows in the dataset
    n_samples_per_row = 400  # Samples per row (from GMM)
    means = [0, 5, -5]  # Means of the Gaussian components
    variances = [1, 0.5, 0.78]  # Variances of the components
    weights = [0.4, 0.4, 0.2]  # Mixing coefficients (must sum to 1)
    key = random.PRNGKey(0)
    # Generate the dataset
    dataset, mean, var  = build_dataset(n_rows, n_samples_per_row, means, variances, weights, 32)
    data = next(iter(dataset))[0][None, :]
    print(data.shape)
    foo = forward_process(1000, 0.1, 20)
    foo.fp(key, data, 4)