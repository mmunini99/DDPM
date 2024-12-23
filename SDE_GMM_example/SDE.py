import jax
import jax.numpy as jnp
from jax import random
from scipy.integrate import solve_ivp
  

class SDE_DDPM(object):
    def __init__(self, beta_lb, beta_ub, N_step):
        self.lb = beta_lb
        self.ub = beta_ub
        self.N = N_step
        # the boundary in terms of time are eps and 1
        self.delta = self.ub - self.lb

    def __setting__(self):
        self.beta = jnp.linspace(self.lb, self.ub, self.N)
        self.beta_norm = jnp.linspace(self.lb/self.N, self.ub/self.N, self.N) # get the discretized betas for SDE

    def __get_drift_and_diff__(self, t):
        beta_t = self.lb + t*(self.delta)
        dft = -0.5*beta_t
        dff = jnp.sqrt(beta_t)

        return dft, dff
    
    def __ode_function__(self, t, input):
        drift, _ = self.__get_drift_and_diff__(t)
        return drift*input
    
    def ode_solver(self, input, t):
        t_span = (0, t)
        solution = solve_ivp(self.__ode_function__, t_span, [input], method='RK45')

        return solution
    


if __name__ == '__main__':
    import jax.numpy as jnp
    
    fp = SDE_DDPM(0.1, 20, 1000)
    a = fp.ode_solver(1, 10)
    print(a.y[0][-1])




