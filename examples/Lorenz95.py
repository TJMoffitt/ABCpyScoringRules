import logging
from numbers import Number
import numpy as np
import scipy
from abcpy.probabilisticmodels import ProbabilisticModel, Continuous, InputConnector
import torch

class Lorenz95(ProbabilisticModel, Continuous):
    def __init__(self, parameters, name='Lorenz95', N=10):
        if not isinstance(parameters, list):
            raise TypeError('Input of Lorenz95 model is of type list')

        if len(parameters) != 1:
            raise RuntimeError('Input list must be of length 1, containing [F].')

        input_connector = InputConnector.from_list(parameters)
        super().__init__(input_connector, name)
        self.N = N
        self.ordered_transforms = [False]
        self.ordered_inverse_transforms = [False]

    def _check_input(self, input_values):
        if len(input_values) != 1:
            raise ValueError('Number of parameters of Lorenz95 model must be 1.')
        # You can add more checks if necessary
        return True

    def forward_simulate(self, input_values, k, rng=np.random.RandomState()):
        F = input_values[0]
        result = self.lorenz95_model_pytorch(F, k)
        return result

    def grad_forward_simulate(self, input_values, k, rng=np.random.RandomState()):
        F = input_values[0]
        result = self.grad_lorenz95_model_pytorch(F, k)
        return result

    def lorenz95_model_pytorch(self, F, n, dt=0.01):
        values = []

        F_tensor = torch.tensor(F, requires_grad=True)
        initial_conditions = torch.ones(self.N, requires_grad=True)

        for _ in range(n):
            dXdt = torch.zeros(self.N)
            for i in range(self.N):
                dXdt[i] = (initial_conditions[(i + 1) % self.N] - initial_conditions[i - 2]) * initial_conditions[i - 1] - initial_conditions[i] + F_tensor

            initial_conditions = initial_conditions + dt * dXdt
            values.append(initial_conditions.detach().numpy())
            
        return values

    def grad_lorenz95_model_pytorch(self, F, n, dt=0.01):
        values = []
        gradvalues = []

        F_tensor = torch.tensor(F, requires_grad=True)
        initial_conditions = torch.ones(self.N, requires_grad=True)

        for _ in range(n):
            dXdt = torch.zeros(self.N)
            for i in range(self.N):
                dXdt[i] = (initial_conditions[(i + 1) % self.N] - initial_conditions[i - 2]) * initial_conditions[i - 1] - initial_conditions[i] + F_tensor

            initial_conditions = initial_conditions + dt * dXdt
            values.append(initial_conditions.detach().numpy())
            
            initial_conditions.sum().backward(retain_graph=True)
            gradvalues.append(F_tensor.grad.item())
            F_tensor.grad.zero_()

        return values + gradvalues

    def _check_output(self, values):
        # Adapt this method based on the output type and constraints of the Lorenz95 model.
        return True

    def get_output_dimension(self):
        return self.N

    def jacobian_list(self):
        return self.ordered_transforms

    def transform_list(self):
        return self.ordered_transforms

    def inverse_transform_list(self):
        return self.ordered_inverse_transforms
    
classvar = Lorenz95([8.0])
print(classvar.grad_forward_simulate([8.0],5))

