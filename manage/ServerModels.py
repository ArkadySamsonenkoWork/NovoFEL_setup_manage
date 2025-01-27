import gpytorch
import torch
import math

class ExperementalModel_1:
    def __init__(self):
        self.Yvar = torch.tensor([1.1])
        
    def exact_function(self, x):
        return x**2 + torch.cos(x**2)
    
    def __call__(self, X):
        size = (*list(self.Yvar.shape), *list(X.shape))
        return self.exact_function(X) + self.Yvar * torch.randn(size)

class ExperementalModel_2:
    def __init__(self, Yvar):
        self.Yvar = Yvar

    def _f1(self, X):
        return (X ** 2 + torch.cos(X ** 2)).squeeze()

    def _f2(self, X):
        return (X ** 2 + torch.sin(X ** 2)).squeeze()

    def _f3(self, X):
        return (X + torch.sin(torch.cos(X ** 2))).squeeze()

    def exact_function(self, X):
        return torch.stack((self._f1(X), self._f2(X), self._f3(X)), dim=-1)

    def __call__(self, X):
        # Ensure the shape of noise matches the output of `exact_function`
        size = (X.shape[0], self.Yvar.shape[0])  # [num_points, 3]
        noise = self.Yvar.sqrt().unsqueeze(0) * torch.randn(size)
        return self.exact_function(X) + noise


class ExperementalModel_3:
    def __init__(self, Yvar):
        self.Yvar = Yvar

    def _f1(self, X):
        return ((X[:, 0] + X[:, 1]) ** 2 + torch.cos(X[:, 0] ** 2)).squeeze()

    def _f2(self, X):
        return (X[:, 0] ** 2 + torch.sin(X[:, 1] ** 2)).squeeze()

    def _f3(self, X):
        return (X[:, 1] + torch.sin(torch.cos(X[:, 1] ** 2))).squeeze()

    def exact_function(self, X):
        return torch.stack((self._f1(X), self._f2(X), self._f3(X)), dim=-1)

    def __call__(self, X):
        # Ensure the shape of noise matches the output of `exact_function`
        size = (X.shape[0], self.Yvar.shape[0])  # [num_points, 3]
        noise = self.Yvar.sqrt().unsqueeze(0) * torch.randn(size)
        return self.exact_function(X) + noise


class ExperementalModels:
    def experemental_model_1(self):
        return ExperementalModel_1()

    def experemental_model_2(self):
        return ExperementalModel_2()


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)