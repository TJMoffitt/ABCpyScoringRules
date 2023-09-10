import numpy as np
import torch
from abcpy.approx_lhd import SynLikelihood, EnergyScore, KernelScore
from abcpy.backends import BackendDummy
from abcpy.continuousmodels import Normal, LogNormal
from abcpy.inferences import adSGLD, SGLD
from abcpy.statistics import Identity
from Gaussian_model import Gaussian


def BetaNormNeg(x1, x2, beta):
    assert len(x2.shape) == 1, "x2 should be a 1D tensor"
    assert x1.shape[1:] == x2.shape, "The last dimensions of x1 and x2 should match"
    
    # Subtract x2 from all entries in x1 and compute the beta norm
    diff = x1 - x2
    norm_beta = torch.sum(torch.abs(diff).pow(2), dim=-1).pow(beta/2)
    return -1*norm_beta


# setup backend
dummy = BackendDummy()

# define a uniform prior distribution
mu = Normal([5, 1], name='mu')
sigma = LogNormal([1,1], name='sigma')
model = Gaussian([mu, sigma])

stat_calc = Identity(degree=2, cross=False)

kernelfunction = lambda x1,x2: BetaNormNeg(x1,x2,1.0)
dist_calc = KernelScore(stat_calc, model, kernelfunction)

y_obs = model.forward_simulate([6,1], 2, rng=np.random.RandomState(8))  # Correct

sampler = SGLD([model], [dist_calc], dummy, seed=1)

journal = sampler.sample([y_obs], 100, 4, 100, step_size=0.0001, w_val = 30, diffusion_factor=0.01, path_to_save_journal="tmp.jnl")

journal.plot_posterior_distr(path_to_save="posterior.png")
journal.traceplot()
