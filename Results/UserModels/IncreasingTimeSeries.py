import logging
from numbers import Number
import numpy as np
from abcpy.probabilisticmodels import ProbabilisticModel, Continuous, InputConnector
import torch

class IncreasingTimeSeries(ProbabilisticModel, Continuous):
    def __init__(self, parameters, timesteps, name='IncreasingTimeSeries'):
        # We expect input of type parameters = [rate]
        if not isinstance(parameters, list):
            raise TypeError('Input of IncreasingTimeSeries model is of type list')

        if len(parameters) != 1:
            raise RuntimeError('Input list must be of length 1, containing [rate].')

        input_connector = InputConnector.from_list(parameters)
        super().__init__(input_connector, name)
        self.ordered_transforms = [False]
        self.ordered_inverse_transforms = [False]

        self.timesteps = timesteps  # Setting the length of the time series

    def _check_input(self, input_values):
        if len(input_values) != 1:
            raise ValueError('Number of parameters of IncreasingTimeSeries model must be 1.')
        return True

    def forward_simulate(self, input_values, n, rng=np.random.RandomState()):
        rate = input_values[0]
        results = self.simulate_time_series(n, rate)
        return results

    def grad_forward_simulate(self, input_values, n, rng=np.random.RandomState()):
        rate = input_values[0]
        results = self.grad_simulate_time_series(n, rate)
        return results

    def simulate_time_series(self, n, rate):

        fullvalues = []
        
        for x in range(0,n):
            values = []
            current_value = 0
            for _ in range(self.timesteps):
                noise = torch.randn(1)  
                current_value += rate + noise
                values.append(current_value.item())
            fullvalues.append(values)
        return fullvalues

    def grad_simulate_time_series(self, n, rate):

        fullvalues = []
        fullgrad = []
        
        for x in range(0,n):
            rate_tensor = torch.tensor(rate, requires_grad=True)
            current_value = torch.tensor(0.0, requires_grad=True)
            values = []
            gradvalues = []
            for _ in range(self.timesteps):
                noise = torch.randn(1)
                new_value = current_value + rate_tensor + noise
                new_value.backward(retain_graph=True)
                
                values.append(new_value.item())
                gradvalues.append([rate_tensor.grad.item()])

                # Update current_value without modifying it in-place
                current_value = new_value.detach().clone().requires_grad_(True)
                
                # Zero out the gradients to ensure the accumulation does not happen
                rate_tensor.grad.zero_()
            fullvalues.append(values)
            fullgrad.append(gradvalues)
            
        return fullvalues + fullgrad


    def _check_output(self, values):
        # Check that the output is a list (or iterable)
        if not isinstance(values, (list, tuple, np.ndarray)):
            return False
        return True

    def get_output_dimension(self):
        return self.timesteps

    def jacobian_list(self):
        return self.ordered_transforms

    def transform_list(self):
        return self.ordered_transforms

    def inverse_transform_list(self):
        return self.ordered_inverse_transforms

simulatorclass = IncreasingTimeSeries([1.5], 10)
print(simulatorclass.forward_simulate([4.0], 10))
