import numpy as np
from abcpy.approx_lhd import SynLikelihood, EnergyScore, KernelScore
from abcpy.backends import BackendDummy
from abcpy.continuousmodels import Normal, LogNormal
from abcpy.inferences import adSGLD, SGLD
from abcpy.statistics import Identity
from UserModels.IncreasingTimeSeries import IncreasingTimeSeries
import time
#from twodimgaussian import Gaussian

# setup backend
start = time.time()
dummy = BackendDummy()

# define a uniform prior distribution
A = Normal([9, 3], name='Rate')
#B = LogNormal([1,1], name='B')
model = IncreasingTimeSeries([A], 4)

stat_calc = Identity(degree=2, cross=False)

dist_calc = EnergyScore(stat_calc, model, 1)

y_obs = model.forward_simulate([6.0], 30, rng=np.random.RandomState(8))  # Correct
print(y_obs)

sampler = adSGLD([model], [dist_calc], dummy, seed=1)

journal = sampler.sample([y_obs], 1500, 100, 1500, step_size=0.0001, w_val = 150, path_to_save_journal="tmp.jnl")
end = time.time()
print(np.mean(y_obs))
print(np.std(y_obs))
print("Run Time:" + str(end - start))
journal.plot_posterior_distr(path_to_save="TimeSeries6Simulationadsgld.png")
journal.traceplot()
#journal.plot_posterior_distr(path_to_save="DeleteMe.png")
#journal.traceplot()