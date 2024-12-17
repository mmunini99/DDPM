from jax import random
from jax import value_and_grad
import jax.numpy as jnp
from jax import jacfwd
from optax import adam, clip, chain
from flax.training.train_state import TrainState
import statistics as sts
from tqdm.notebook import tqdm

from .NN_model import ScoreMatchNN as Net
from .Utilities import loss_function_denoise, sequence_sigma, loss_function_explicit


class TrainState(TrainState):
  batch_stats: any

class NCSN(object):
    '''
    Define the NCSN based on RefineNet network unconditioned by the stendard deviation.
    '''
    def __init__(self, denoise_sm, setseed, channels_ini, channels_out, seq_dim_channels, height, width, lr, nesterov_bool, sigma_ini, sigma_fin, L, gamma, T, n_epoch, batch_size):
        self.denoise_sm = denoise_sm
        self.setseed = setseed
        self.channels_ini = channels_ini
        self.channels_out = channels_out
        self.seq_dim_channels = seq_dim_channels
        self.sigma_ini = sigma_ini
        self.sigma_fin = sigma_fin
        self.L = L
        self.gamma = gamma
        self.T = T
        self.height = height
        self.width = width
        self.lr = lr
        self.nesterov_bool = nesterov_bool
        self.n_epoch = n_epoch
        self.batch_size = batch_size

        # list for storing data
        self.array_train_loss = []

    def __init_model__(self, training_flag):
        # define the gloabl set seed
        self.rng = random.PRNGKey(self.setseed)
        # define the shape of an image
        sample_image_format = jnp.zeros([1, self.height, self.width, 1])
        # define the shape of the sigma passed to the net
        sample_sigma_format = jnp.zeros(1)

        # define the model
        self.model = Net(self.channels_ini, self.seq_dim_channels, self.channels_out, training_flag)
        
        
        # Feed in a sample so it knows how the input is composed
        var = self.model.init(self.rng, [sample_image_format, sample_sigma_format]) # here the shape of input is passed --> list of image batch and array of sigma
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
        

        self.sigma_array = sequence_sigma(self.sigma_ini, self.sigma_fin, self.L)

    def __update_denoise__(self, perturbed_image, noise_applied, sigma):
        # first define the tool to allow JAX to compute the loss and gradient
        def loss_computation(params):
            # compute the predictions
            pred, updates = self.trainer.apply_fn({'params': params, 'batch_stats': self.trainer.batch_stats}, [perturbed_image, sigma], mutable=['batch_stats'])
            value_loss = loss_function_denoise(pred, noise_applied, sigma)
                       
            return value_loss, updates
        
        # compute the loss and gradients --> forward prop.
        (loss, upt), grad = value_and_grad(loss_computation, has_aux = True)(self.trainer.params)
        # backward prop. --> optimization step of NN
        self.trainer = self.trainer.apply_gradients(grads=grad)
        self.trainer = self.trainer.replace(batch_stats=upt['batch_stats'])

        return loss
    

    def __update_explicit__(self, batch_image):
        # first define the tool to allow JAX to compute the loss and gradient
        def loss_computation(params):
            # compute the predictions --> score
            pred, updates = self.trainer.apply_fn({'params': params, 'batch_stats': self.trainer.batch_stats}, [batch_image, None], mutable=['batch_stats'])
            # define the fucntion on which compute the jacobian
            def func(x):
                pred, _ = self.trainer.apply_fn({'params': params, 'batch_stats': self.trainer.batch_stats}, [batch_image, None], mutable=['batch_stats'])
                return pred
            # compute the jacobian
            jac = jacfwd(func)(batch_image)
            value_loss = loss_function_explicit(jac, pred)
            return value_loss, updates
        
        # compute the loss and gradients --> forward prop.
        (loss, upt), grad = value_and_grad(loss_computation, has_aux = True)(self.trainer.params)
        # backward prop. --> optimization step of NN
        self.trainer = self.trainer.apply_gradients(grads=grad)
        self.trainer = self.trainer.replace(batch_stats=upt['batch_stats'])

        return loss
    

    
    def __single_step_denoise__(self, dataset):
        for i, batch_data in enumerate(tqdm(dataset)):
            # generate 2 sub-rng --> do not use original rng (good practice)
            self.rng, rng_sigma, rng_gauss_noise = random.split(self.rng, 3)
            # generate the array of sigma using the random key generated above --> array of same lenght of the batch size of data
            sigma_idx = random.randint(rng_sigma, shape=(self.batch_size,), minval=0, maxval=len(self.sigma_array))
            sigma = self.sigma_array[sigma_idx] # Map indices to actual sigma values (shape: batch_size)
            # generate noise perturbation
            noise_applied = random.normal(rng_gauss_noise, shape=batch_data.shape)
            # generate data perturbed
            data_perturbed = batch_data+noise_applied
            # back. prop., optimize and get the loss value
            loss_out = self.__update_denoise__(data_perturbed, noise_applied, sigma)

            self.array_train_loss.append(loss_out) # append only for keeping track of results

        # cleaning procedure
        del sigma_idx
        del sigma
        del noise_applied
        del data_perturbed
        del batch_data
        # compute the mean of the error
        avg_loss_epoch = jnp.mean(jnp.array(self.array_train_loss))
        # clean the list for storing loss values
        self.array_train_loss = []

        return avg_loss_epoch 
    
    def __single_step_explicit__(self, dataset):
        for i, batch_data in enumerate(tqdm(dataset)):
            # back. prop., optimize and get the loss value
            loss_out = self.__update_explicit__(batch_data)

            self.array_train_loss.append(loss_out) # append only for keeping track of results

        # cleaning procedure
        del loss_out
        del batch_data
        # compute the mean of the error
        avg_loss_epoch = jnp.mean(jnp.array(self.array_train_loss))
        # clean the list for storing loss values
        self.array_train_loss = []

        return avg_loss_epoch
    

    def training(self, dataset):
        # create the model and complete the setup
        self.__init_model__(True)
        # choose how to train the model
        if self.denoise_sm:
            for idx in range(self.n_epoch):
                avg_loss_epoch = self.__single_step_denoise__(dataset)
                print("Loss at epoch ", idx, " is ", avg_loss_epoch)
        else:
            for idx in range(self.n_epoch):
                avg_loss_epoch = self.__single_step_explicit__(dataset)
                print("Loss at epoch ", idx, " is ", avg_loss_epoch)


    def annealed_L_sampling(self, eps):
        # recal the model 
        self.model_trained = Net(self.channels_ini, self.seq_dim_channels, self.channels_out, False)
        # define the key for initialize teh noisy image
        self.rng, key_x0 = random.split(self.rng, 2)
        # init the starting noisy image
        img = random.uniform(key_x0, shape=(1, self.height, self.width, 1), minval=0, maxval=1)
        for i in tqdm(range(self.L)):
            alpha = eps*self.sigma_array[i]/self.sigma_array[-1] # compute the step size of Langevin equation --> eps*sigma_i/sigma_L
            for time in range(self.T): # run the SDE for a period T with fixed sigma
                self.rng, key_noise = random.split(self.rng, 2)
                pred = self.model_trained.apply({'params': self.trainer.params, 'batch_stats': self.trainer.batch_stats}, [img, self.sigma_array[i]], mutable=False)
                z = random.normal(key_noise, shape=(1, self.height, self.width, 1)) # white noise
                img = img + alpha*pred + jnp.sqrt(2*alpha)*z
        
        pred_last = self.model_trained.apply({'params': self.trainer.params, 'batch_stats': self.trainer.batch_stats}, [img, self.sigma_array[-1]], mutable=False)
        # remove unwanted noise using Tweedie's formula
        img = img + (self.sigma_array[-1]**2)*pred_last

        return img







        



    
if __name__ == "__main__":
    import os
    from Dataset import load_data
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_debug_nans", False)
    train_data, _ = load_data(os.path.abspath("C:/Users/matte/Documents/JAX Tutorial/NCSN/datset_MNIST/"), 32, 32, 32, False, 64)
    model = NCSN(True, 42, 32, 1, (1,1,1,1), 32, 32, 1e-5, False, 1, 0.1, 10, None, 10, 2, 32)
    model.training(train_data)
    model.annealed_L_sampling(0.1)
