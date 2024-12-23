from jax import random
from jax import value_and_grad
import jax.numpy as jnp
from jax import jit
from optax import adam, clip, chain
from optax.losses import squared_error
from flax.training.train_state import TrainState
import statistics as sts
from tqdm.notebook import tqdm
from scipy.integrate import solve_ivp

from ForwardProcess import forward_process
from Utilities import loss_function
from UNET import UNet


class TrainState(TrainState):
  batch_stats: any

class DiffusionModel(object):
    '''
    Define the configuration of the DDPM.
    setseed --> set the random numer generator
    UNET_scaling_tuple --> tuple for modelling the downsample and upsample phase in the U-Net 
    n_heads --> heads of the Multi-Head Attention mechanism
    lr --> Learning rate of the Adam optimizer
    nesterov_bool --> Boolean to use or not the Nestorov momentum

    Requirement:
    image shape data: (batch size, height, width, 1)
    '''

    def __init__(self, setseed, dimension, UNET_scaling_tuple, shape_data, lr, nesterov_bool, n_epoch, N, beta_lb, beta_ub, eps_lb, batch_size):
        self.setseed = setseed
        self.dimension = dimension
        self.UNET_scaling_tuple = UNET_scaling_tuple
        self.shape = shape_data
        self.lr = lr
        self.nesterov_bool = nesterov_bool
        self.n_epoch = n_epoch
        self.N = N
        self.beta_lb = beta_lb
        self.beta_ub = beta_ub
        self.eps_lb = eps_lb
        self.batch_size = batch_size

        # define the DDPM forward process --> init it!
        self.fw = forward_process(N = self.N, beta_lb = self.beta_lb, beta_ub = self.beta_ub)

        # list for storing data
        self.array_train_loss = []
        
    def __init_model__(self):
        # define the gloabl set seed
        self.rng = random.PRNGKey(self.setseed)
        # define the shape of a row
        sample_data_format = jnp.zeros([1, self.shape])
        # define the shape of the timestep passed to the U-Net blocks
        sample_timestep_format = jnp.zeros(1)

        # define the model
        self.model = UNet(self.dimension, self.UNET_scaling_tuple, self.shape)

        # Feed in a sample so it knows how the input is composed
        var = self.model.init(self.rng, [sample_data_format, sample_timestep_format], training=True) # here the shape of input is passed --> list of image batch and array of timestep
        params  = var['params']
        batch_stats = var['batch_stats']
        
        # define the optimizer of the model
        self.optimezer = chain(
            clip(1.0),
            adam(learning_rate=self.lr, nesterov=self.nesterov_bool)
            )

        # create the compelte training setup
        self.trainer = TrainState.create(apply_fn=self.model.apply,
                                    params=params,
                                    tx = self.optimezer,
                                    batch_stats=batch_stats)    
        # define the array of noise
        self.timestep_base = jnp.linspace(self.eps_lb/self.N, 1/self.N, self.N)
        self.timestep_base_rev = self.timestep_base[::-1]
        
    
    def __update__(self, perturbed_data, noise_applied, idx_array):
        timestep = jnp.take(self.timestep_base, idx_array)
        # first define the tool to allow JAX to compute the loss and gradient
        def loss_computation(params):
            # compute the predictions
            pred, updates = self.trainer.apply_fn({'params': params, 'batch_stats': self.trainer.batch_stats}, [perturbed_data, timestep], training = True, mutable=['batch_stats'])
            value_loss = loss_function(pred, noise_applied)
            return value_loss, updates
        # compute the loss and gradients --> forward prop.
        (loss, upt), grad = value_and_grad(loss_computation, has_aux = True)(self.trainer.params)
        # backward prop. --> optimization step of NN
        self.trainer = self.trainer.apply_gradients(grads=grad)
        self.trainer = self.trainer.replace(batch_stats=upt['batch_stats'])

        return loss
    
    def __single_step__(self, dataset):
        for i, batch_data in enumerate(tqdm(dataset)):
            # generate 2 sub-rng --> do not use original rng (good practice)
            self.rng, rng_timestep, rng_gauss_noise = random.split(self.rng, 3)
            # generate the array of idx using the random key generated above --> array of same lenght of the batch size of data
            idx_array = random.randint(rng_timestep, shape=(batch_data.shape[0],), minval=0, maxval=self.N)
            # get the actual data to pass to the U-Net: firstly batch of perturbed data; secondly the target noise
            # to do this we need to call the forward pocess of the fw class --> use the second rng key 
            batch_data_pert, batch_noise_applied = self.fw.fp(rng_gauss_noise, batch_data, idx_array)
            # back. prop., optimize and get the loss value
            loss_out = self.__update__(batch_data_pert, batch_noise_applied, idx_array)

            self.array_train_loss.append(loss_out) # append only for keeping track of results

        # cleaning procedure
        # del rng_timestep
        # del rng_gauss_noise
        # del idx_array
        # del batch_data_pert
        # del batch_noise_applied
        # compute the mean of the error
        avg_loss_epoch = jnp.mean(jnp.array(self.array_train_loss))
        # clean the list for storing loss values
        self.array_train_loss = []

        return avg_loss_epoch 
    

    def training(self, dataset):
        # create the model and complete the setup
        self.__init_model__()
        for idx in range(self.n_epoch):
            avg_loss_epoch = self.__single_step__(dataset)
            print("Loss at epoch ", idx, " is ", avg_loss_epoch)

    
    def sampling_DDPM(self):
        # define the beta used
        array_beta = self.fw.get_params()
        # generate a key for random noise --> gaussian
        self.rng, rng_sampling = random.split(self.rng, 2)
        # define a list to store the T transition
        list_transition = []
        # define the initial noise --> white noise --> make shape for model init
        data = random.normal(rng_sampling, (1, self.shape))
        # first saving
        list_transition.append(data)
        # define the recursive mechanism
        for t_idx in tqdm(range(0, len(self.timestep_base_rev)-1)):
            time = self.timestep_base_rev[t_idx]
            time_prev = self.timestep_base_rev[t_idx+1]
            # define the noise for RK45-Maruyama
            self.rng, rng_rev = random.split(self.rng, 2)
            z = random.normal(rng_rev, (1, self.shape))
            # define the DRIFT term
            dft = -0.5*(array_beta[0] + time*(array_beta[1]-array_beta[0]))
            # define the DIFFUSION term
            dff = jnp.sqrt(array_beta[0] + time*(array_beta[1]-array_beta[0]))
            # define the function to compute the deterministic part
            def ode_function(t, input):
                # make the prediction
                score = self.model.apply({'params': self.trainer.params, 'batch_stats': self.trainer.batch_stats}, [input, jnp.full((1,), t)], training = False,  mutable=False)

                return dft - (dff**2)*score
            # simulate the trajectory due to deterministic component
            det_comp = solve_ivp(ode_function, (time, time_prev), data.squeeze(), method='RK45')
            # transform data shape
            det_comp = det_comp.y[:, 1][jnp.newaxis, :]
            # compute Maruyama component / stochastic
            sto_comp = dff*jnp.sqrt(1/self.N)*z
            # obtain the denoised data
            data = data - det_comp + sto_comp

            
            # save it
            list_transition.append(data)


        return list_transition

   





if __name__ == '__main__':
    from Dataset import build_dataset
    n_rows = 64*2  # Total number of rows in the dataset
    n_samples_per_row = 400  # Samples per row (from GMM)
    means = [0, 5, -5]  # Means of the Gaussian components
    variances = [1, 0.5, 0.78]  # Variances of the components
    weights = [0.4, 0.4, 0.2]  # Mixing coefficients (must sum to 1)

    # Generate the dataset
    dataset, mean, var  = build_dataset(n_rows, n_samples_per_row, means, variances, weights, 32)
    model = DiffusionModel(42, 32, (6,4,2), 400, 1e-5, False, 4, 20,  0.1, 20, 1e-4, 32)
    model.training(dataset)
    l = model.sampling_DDPM()
    print(len(l))









    



