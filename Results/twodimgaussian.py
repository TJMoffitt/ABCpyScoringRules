import logging
from numbers import Number
import numpy as np
import scipy
from abcpy.probabilisticmodels import ProbabilisticModel, Continuous, InputConnector
import torch
from torch.autograd.functional import jacobian

class Gaussian(ProbabilisticModel, Continuous):
    def __init__(self, parameters, name='Gaussian'):
        # We expect input of type parameters = [mu, sigma]
        if not isinstance(parameters, list):
            raise TypeError('Input of Normal model is of type list')

        if len(parameters) != 2:
            raise RuntimeError('Input list must be of length 2, containing [mu, sigma].')

        input_connector = InputConnector.from_list(parameters)
        super().__init__(input_connector, name)
        self.ordered_transforms = [False, torch.exp]
        self.ordered_inverse_transforms = [False, torch.log]

    def _check_input(self, input_values):
        # Check whether input has correct type or format
        if len(input_values) != 2:
            raise ValueError('Number of parameters of Normal model must be 2.')
        # Check whether input is from correct domain
        mu = input_values[0]
        sigma = input_values[1]
        if sigma < 0:
            return False
        return True

    def forward_simulate(self, input_values, k, rng=np.random.RandomState()):
        # Extract the input parameters
        result = self.normal_model_pytorch([float(input_value) for input_value in input_values], k)#[np.array([x]) for x in vector_of_k_samples]
        return result

    def normal_model_pytorch(self, input_values, n, return_grad = False):
        values = []
        for n in range(0,n):
            mu = torch.tensor(input_values[0], requires_grad = True)
            sigma = torch.tensor(input_values[1], requires_grad = True)
            variables = [mu,sigma]
            yval1 = torch.randn(1)*sigma + mu
            yval2 = torch.randn(1)*sigma + mu
            value = np.array([yval1.item(),yval2.item()])
            values.append(value)
        return values

    def grad_forward_simulate(self, input_values, k, rng=np.random.RandomState()):
        # Takes input in the form:  [a,....,z]
        # Outputs: array: [x1, x2, ...... ,xn, [dx1/dtheta1, dx1/dtheta2], ...... [dxn/dtheta1, dxn/dtheta2],]
        result = self.grad_normal_model_pytorch([float(input_value) for input_value in input_values], k)#[np.array([x]) for x in vector_of_k_samples]
        return result

    def grad_normal_model_pytorch(self, input_values, n, return_grad = False):
        values = []
        gradvalues = []
        for n in range(0,n):
            mu1 = torch.tensor(input_values[0], requires_grad = True)
            sigma1 = torch.tensor(input_values[1], requires_grad = True)
            z1 = torch.randn(1)
            variables1 = [mu1,sigma1]
            yval1 = z1*sigma1 + mu1
            yval1.backward()

            mu2 = torch.tensor(input_values[0], requires_grad = True)
            sigma2 = torch.tensor(input_values[1], requires_grad = True)
            z2 = torch.randn(1)
            variables2 = [mu2,sigma2]
            yval2 = z2*sigma2 + mu2
            yval2.backward()

            values.append([yval1.item(),yval2.item()])

            gradvalue1 = []
            for var in variables1:
                gradvalue1.append(var.grad.item())

            gradvalue2 = []
            for var in variables2:
                gradvalue2.append(var.grad.item())

            gradvalues.append([gradvalue1, gradvalue2])
        return values + gradvalues

    def _check_output(self, values):
        if not isinstance(values, Number):
            raise ValueError('Output of the normal distribution is always a number.')
        # At this point values is a number (int, float); full domain for Normal is allowed
        return True

    def get_output_dimension(self):
        return 1

    def jacobian_list(self):
        return self.ordered_transforms

    def transform_list(self):
        return self.ordered_transforms

    def inverse_transform_list(self):
        return self.ordered_inverse_transforms
    
