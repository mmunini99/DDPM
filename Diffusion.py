from jax import random
from jax import value_and_grad
import jax.numpy as jnp
from jax import jit
from optax import adam
from optax.losses import squared_error
from flax.training.train_state import TrainState
import statistics as sts
from tqdm.notebook import tqdm

from ForwardProcess import forward_process
from Utilities import PositionalEmbedding, ResNetBlock, Attention, loss_function
from UNET import UNet




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

    def __init__(self, setseed, dimension, UNET_scaling_tuple, n_heads, height, width, lr, nesterov_bool, n_epoch, T, beta_lb, beta_ub, batch_size):
        self.setseed = setseed
        self.dimension = dimension
        self.UNET_scaling_tuple = UNET_scaling_tuple
        self.n_heads = n_heads
        self.height = height
        self.width = width
        self.lr = lr
        self.nesterov_bool = nesterov_bool
        self.n_epoch = n_epoch
        self.T = T
        self.beta_lb = beta_lb
        self.beta_ub = beta_ub
        self.batch_size = batch_size

        # # define the DDPM forward process --> init it!
        self.fw = forward_process(T = self.T, beta_lb = self.beta_lb, beta_ub = self.beta_ub)      

        # list for storing data
        self.array_train_loss = []
        
    def __init_model__(self):
        # define the gloabl set seed
        self.rng = random.PRNGKey(self.setseed)
        # define the shape of an image
        sample_image_format = jnp.zeros([1, self.height, self.width, 1])
        # define the shape of the timestep passed to the U-Net blocks
        sample_timestep_format = jnp.zeros(1)

        # define the model
        self.model = UNet(self.dimension, self.UNET_scaling_tuple, self.n_heads)

        # Feed in a sample so it knows how the input is composed
        params = self.model.init(self.rng, [sample_image_format, sample_timestep_format]) # here the shape of input is passed --> list of image batch and array of timestep

        # define the optimizer of the model
        self.optimezer = adam(learning_rate=self.lr, nesterov=self.nesterov_bool)

        # create the compelte training setup
        self.trainer = TrainState.create(apply_fn=self.model.apply,
                                    params=params,
                                    tx = self.optimezer)   
    
    def __update__(self, perturbed_image, noise_applied, timestep):
        # first define the tool to allow JAX to compute the loss and gradient
        def loss_computation(params):
            # compute the predictions
            pred = self.trainer.apply_fn(params, [perturbed_image, timestep])
            value_loss = loss_function(pred, noise_applied)
            return value_loss
        # compute the loss and gradients --> forward prop.
        loss, grad = value_and_grad(loss_computation)(self.trainer.params)
        # backward prop. --> optimization step of NN
        self.trainer = self.trainer.apply_gradients(grads=grad)

        return loss
    
    def __single_step__(self, dataset):
        for i, batch_data in enumerate(tqdm(dataset)):
            # generate 2 sub-rng --> do not use original rng (good practice)
            self.rng, rng_timestep, rng_gauss_noise = random.split(self.rng, 3)
            # generate the array of tiemstep using the random key generated above --> array of same lenght of the batch size of data
            timestep = random.randint(rng_timestep, shape=(batch_data.shape[0],), minval=0, maxval=self.T)
            # get the actual data to pass to the U-Net: firstly batch of perturbed data; secondly the target noise
            # to do this we need to call the forward pocess of the fw class --> use the second rng key 
            batch_img_pert, batch_noise_applied = self.fw.fp(rng_gauss_noise, batch_data, timestep)
            # back. prop., optimize and get the loss value
            loss_out = self.__update__(batch_img_pert, batch_noise_applied, timestep)

            self.array_train_loss.append(loss_out) # append only for keeping track of results

        # cleaning procedure
        del rng_timestep
        del rng_gauss_noise
        del timestep
        del batch_img_pert
        del batch_noise_applied
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


    def __single_sampling_DDPM__(self, img_pert, pred, timestep):
        '''
        DDPM
        img_pert --> image passed. If single image : (1, height, width, 1)
        pred --> U-Net prediction based on batch_img_pert conditioned on timestep
        '''
        # load the data used in the forward denoising pass
        beta, alpha, alpha_hat = self.fw.get_params()
        # define the components of the denoised image by DDPM from t to t-1 : const1*(img - const2*pred) + sigma*gaussian_noise
        const1 = 1/(jnp.sqrt(jnp.take(alpha, timestep))) # 1/sqrt(alpha_t)
        const2 = (1-jnp.take(alpha, timestep))/(jnp.sqrt(1-jnp.take(alpha_hat, timestep))) # 1-alpha_t/sqrt(1-alpha_hat_t)
        # condition to apply white noise perturbation
        if timestep == 0:
            denoised_image = const1*(img_pert-const2*pred)
        else:
            # generate a key for random noise --> gaussian
            self.rng, rng_noise = random.split(self.rng, 2)
            gauss_noise = random.normal(key=rng_noise, shape=img_pert.shape)
            # define sigma_t = beta_t*((1-alpha_hat_(t-1))/(1-alpha_hat_(t)))
            sigma = jnp.take(beta, timestep)*((1-jnp.take(alpha_hat, timestep-1))/(1-jnp.take(alpha_hat, timestep)))
            denoised_image = const1*(img_pert-const2*pred) + sigma*gauss_noise

        return denoised_image
    
    def __single_sampling_DDIM(self, img_pert, pred, timestep, eta):
        '''
        DDIM if eta = 0
        if eta in [0;1[ --> trade-off between DDIM and DDPM.
        img_pert --> image passed. If single image : (1, height, width, 1)
        pred --> U-Net prediction based on batch_img_pert conditioned on timestep
        '''
        # load the data used in the forward denoising pass
        _, __, alpha_hat = self.fw.get_params()
        if timestep == 0:
            # define the components of the denoised image by DDIM from t to 0 : (img - const2*pred)/const3   
            const2 = jnp.sqrt(1-jnp.take(alpha_hat, timestep)) # sqrt(1-alpha_t)
            const3 = jnp.sqrt(jnp.take(alpha_hat, timestep)) # sqrt(alpha_t)
            denoised_image = (img_pert-const2*pred)/const3
        else:
            # define the components of the denoised image by DDIM from t to t-1 : const1*(img - const2*pred)/const3 + const4*pred + sigma*gaussian_noise  
            const1 = jnp.sqrt(jnp.take(alpha_hat, timestep-1)) # sqrt(alpha_t-1)
            const2 = jnp.sqrt(1-jnp.take(alpha_hat, timestep)) # sqrt(1-alpha_t)
            const3 = jnp.sqrt(jnp.take(alpha_hat, timestep)) # sqrt(alpha_t)
            sigma = eta*jnp.sqrt((1-jnp.take(alpha_hat, timestep-1))/(1-jnp.take(alpha_hat, timestep)))*jnp.sqrt(1-(jnp.take(alpha_hat, timestep))/(jnp.take(alpha_hat, timestep-1)))
            const4 = jnp.sqrt(1-jnp.take(alpha_hat, timestep-1)-sigma**2)
            # generate a key for random noise --> gaussian
            self.rng, rng_noise = random.split(self.rng, 2)
            gauss_noise = random.normal(key=rng_noise, shape=img_pert.shape)
            # compute denoised image at step t
            denoised_image = const1*((img_pert-const2*pred)/(const3))+const4*pred+sigma*gauss_noise
        
        return denoised_image


    def sampling_DDPM(self):
        # generate a key for random noise --> gaussian
        self.rng, rng_sampling = random.split(self.rng, 2)
        # define a list to store the T transition
        list_transition = []
        # define the initial noise --> white noise --> make shape for model init
        img = random.normal(rng_sampling, (1, self.height, self.width, 1))
        # first saving
        list_transition.append(img)
        # define the recursive mechanism
        for time in tqdm(reversed(range(0, self.T-1))):
            # make the prediction
            pred = self.model.apply({'params': self.trainer.params["params"]}, [img, jnp.full((1,), time)])
            # denoise the image --> replace img with the new image
            img = self.__single_sampling_DDPM__(img, pred, time)
            # save it
            list_transition.append(img)


        return list_transition

    def sampling_DDIM(self, num_step_inference, eta): # add method linear and quadratic
        # generate a key for random noise --> gaussian
        self.rng, rng_sampling = random.split(self.rng, 2)
        # define a list to store the T transition
        list_transition = []
        # define the initial noise --> white noise --> make shape for model init
        img = random.normal(rng_sampling, (1, self.height, self.width, 1))
        # first saving
        list_transition.append(img)
        # define the trajectory --> linear method
        grid_time =  range(0, self.T, self.T // (num_step_inference-1)) 
        param_c = self.T/len(grid_time)
        # define the backward step on trajectory 
        for time in tqdm(reversed(grid_time)):
            time_adj = int(param_c*time)
            # make the prediction  
            pred = self.model.apply({'params': self.trainer.params["params"]}, [img, jnp.full((1,), time_adj)])
            # denoise the image --> replace img with the new image
            img = self.__single_sampling_DDIM(img, pred, time, eta)
            # save it
            list_transition.append(img)


        return list_transition

















    



