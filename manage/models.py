from typing import Optional, Tuple

import torch

import gpytorch
from gpytorch.likelihoods import GaussianLikelihood

from botorch.utils.transforms import t_batch_mode_transform
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.optim import optimize_acqf
from botorch.models.multitask import KroneckerMultiTaskGP
from botorch.models.gp_regression import SingleTaskGP
from botorch.acquisition import AcquisitionFunction
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.input import Normalize


class SquareMinimaFunction(AcquisitionFunction):
    """Custom acquisition function to compute a weighted combination of mean and variance."""

    def __init__(self, model, alpha: float = 0):
        """
        Args:
            model (BotorchBaseModel): The model used for acquisition.
            alpha (float): Weighting factor for the mean in the acquisition function.
        """
        super().__init__(model)
        self.alpha = torch.tensor([alpha])

    def _mean_and_sigma(
        self, X: torch.Tensor, compute_sigma: bool = True, min_var: float = 1e-12
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Computes the mean and variance of the model's posterior.

        Args:
            X (torch.Tensor): Input tensor of shape `(batch_shape x q x d)`.
            compute_sigma (bool): Whether to compute variance. Defaults to True.
            min_var (float): Minimum variance value for stability. Defaults to 1e-12.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: Mean and standard deviation tensors.
        """
        self.to(device=X.device)
        posterior = self.model.posterior(X=X)
        mean = posterior.mean.squeeze(-2).squeeze(-1)  # Remove redundant dimensions
        if not compute_sigma:
            return mean, None
        sigma = posterior.variance.clamp_min(min_var).sqrt().view(mean.shape)
        return mean, sigma

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the acquisition function on candidate set X.

        Args:
            X (torch.Tensor): Input tensor of shape `(batch_shape x 1 x d)`.

        Returns:
            torch.Tensor: Acquisition values of shape `(batch_shape)`.
        """
        mean, variance = self._mean_and_sigma(X)
        acquisition_value = self.alpha * mean**2 - (1 - self.alpha) * variance
        return -torch.sum(acquisition_value, dim=-1)


class MultiOutputModel:
    """MultiTask GP model for Bayesian optimization."""

    def __init__(
            self,
            train_X: torch.Tensor,
            train_Y: torch.Tensor,
            Yvar: torch.Tensor,
            bounds: torch.Tensor,
            weights: torch.Tensor = torch.tensor([1.0, 1.0, 1.0]),
            enforce_positive_outputs: bool = False,
    ):
        """
        Args:
            train_X (torch.Tensor): Training inputs.
            train_Y (torch.Tensor): Training outputs.
            Yvar (torch.Tensor): Observation noise.
            enforce_positive_outputs (bool): Whether to enforce positivity of outputs. Defaults to False.
            weights (torch.Tensor): weights to calculate minima of the function
        """
        if enforce_positive_outputs:
            train_Y = train_Y.square()
            Yvar = Yvar.square()

        self.weights = weights.unsqueeze(0)
        self.Yvar = Yvar
        self.dtype = train_X.dtype
        self.enforce_positive_outputs = enforce_positive_outputs
        self.input_transform = self._get_input_transform(train_X.shape[1], bounds)
        self.gp_model = self._create_model(train_X, train_Y)
        self.training_data_X = train_X.tolist()
        self.training_data_Y = train_Y.tolist()
        self._best_X, self._best_f = self._calculate_best_point(train_X, train_Y)

    def __repr__(self):
        return (f"MultiOutputModel with the next parameters: "
                f"Yvar: {self.Yvar}, dtype: {self.dtype},"
                f"enforce_positive_outputs: {self.enforce_positive_outputs}")

    def _get_input_transform(self, d: int, bounds: torch.Tensor):
        return Normalize(d, bounds=bounds)

    def transform_inputs(self, X: torch.Tensor):
        return self.input_transform(X)

    def untransform_inputs(self, X: torch.Tensor):
        return self.input_transform.untransform(X)

    @property
    def best_X(self):
        return self._best_X

    @property
    def best_f(self):
        return self._best_f

    def _calculate_best_point(self, X: torch.Tensor, Y: torch.Tensor):
        statistics = torch.sum(Y ** 2 * self.weights, dim=-1)
        index = torch.argmin(statistics)
        best_f = statistics[index]
        best_X = X[index]
        return best_X, best_f

    def _updata_best_point(self, X: torch.Tensor, Y: torch.Tensor):
        cand_X, cand_f = self._calculate_best_point(X, Y)
        if cand_f < self._best_f:
            self._best_f = cand_f
            self._best_X = cand_X

    def _create_model(self, train_X, train_Y):
        raise NotImplementedError("Add _create_model_method")

    def reinitialize_model(self) -> None:
        """Re-initialize the model with updated training data."""
        X = torch.tensor(self.training_data_X, dtype=self.dtype)
        Y = torch.tensor(self.training_data_Y, dtype=self.dtype)
        gp_model = self._create_model(X, Y)
        gp_model.load_state_dict(self.gp_model.state_dict())
        self.gp_model = gp_model

    def add_training_points(self, X: torch.Tensor, Y: torch.Tensor, reinit: bool = True) -> None:
        """
        Add new points to the model.

        Args:
            X (torch.Tensor): New input data.
            Y (torch.Tensor): New output data.
            reinit (bool): Whether to reinitialize the model. Defaults to False.
        """
        self.training_data_X.append(X.squeeze(0).tolist())
        self.training_data_Y.append(
            Y.square().squeeze(0).tolist() if self.enforce_positive_outputs else Y.squeeze(0).tolist()
        )
        self._updata_best_point(X, Y.square() if self.enforce_positive_outputs else Y)

        if reinit:
            self.reinitialize_model()
        else:
            raise NotImplementedError("please, set reinit=True")

    def _get_marginallikehood(self):
        raise NotImplementedError("Add _create_model_method")

    def train(self) -> None:
        """Train the model using marginal log-likelihood optimization."""
        self.gp_model.train()
        mll = self._get_marginallikehood()
        fit_gpytorch_mll(mll)

    def observe_pred(self, X: torch.Tensor):
        """
        Predict the model's posterior at given points.

        Args:
            X (torch.Tensor): Input tensor.

        Returns:
            gpytorch.distributions.MultivariateNormal: Posterior distribution.
        """
        self.gp_model.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            return self.gp_model.posterior(self.transform_inputs(X), observation_noise=True)

    def set_train_data(self, train_X: torch.Tensor, train_Y: torch.Tensor) -> None:
        """Update the training data."""
        self.gp_model.set_train_data(self.transform_inputs(train_X), train_Y, strict=False)

    def _get_analytic_mean_acq_function(self, alpha: float) -> SquareMinimaFunction:
        """Return an instance of the custom acquisition function."""
        return SquareMinimaFunction(self.gp_model, alpha)

    def get_new_candidate_point(self, bounds: torch.Tensor, alpha: float = 0) -> torch.Tensor:
        """
        Optimize the acquisition function to get a new candidate point.

        Returns:
            torch.Tensor: Optimized candidate point.
        """
        acq_function = self._get_analytic_mean_acq_function(alpha)
        candidate_point, _ = optimize_acqf(
            acq_function=acq_function, q=1, num_restarts=3, raw_samples=64, bounds=self.transform_inputs(bounds)
        )
        return self.untransform_inputs(candidate_point)


class KroneckerModel(MultiOutputModel):
    """Encapsulation of a Kronecker MultiTask GP model for Bayesian optimization."""
    def __init__(
            self,
            train_X: torch.Tensor,
            train_Y: torch.Tensor,
            Yvar: torch.Tensor,
            bounds: torch.Tensor,
            weights: torch.Tensor = torch.tensor([1.0, 1.0, 1.0]),
            rank: int = 2,
            enforce_positive_outputs: bool = False,
    ):
        """
        Args:
            train_X (torch.Tensor): Training inputs.
            train_Y (torch.Tensor): Training outputs.
            Yvar (torch.Tensor): Observation noise.
            rank (int): Rank of the multitask covariance matrix. Defaults to 2.
            enforce_positive_outputs (bool): Whether to enforce positivity of outputs. Defaults to False.
            weights (torch.Tensor): weights to calculate minima of the function
        """
        self._rank = rank
        super().__init__(train_X, train_Y, Yvar, bounds, weights, enforce_positive_outputs)

    def __repr__(self):
        return (f"MultiListModel with the next parameters: "
                f"Yvar: {self.Yvar}, dtype: {self.dtype} \n  "
                f"enforce_positive_outputs: {self.enforce_positive_outputs}")

    @property
    def rank(self):
        return self._rank

    def _create_model(self, train_X: torch.Tensor, train_Y: torch.Tensor):
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=train_Y.size(-1), has_global_noise=False
        )
        likelihood.task_noises = self.Yvar
        likelihood.raw_task_noises.requires_grad = False

        return KroneckerMultiTaskGP(self.transform_inputs(train_X), train_Y, likelihood=likelihood,
                                    rank=self._rank)

    def _get_marginallikehood(self):
        return gpytorch.mlls.ExactMarginalLogLikelihood(self.gp_model.likelihood, self.gp_model)


class MultiListModel(MultiOutputModel):
    def __repr__(self):
        return (f"MultiListModel with the next parameters: "
                f"Yvar: {self.Yvar}, dtype: {self.dtype}"
                f"enforce_positive_outputs: {self.enforce_positive_outputs}")

    def _create_model(self, train_X: torch.Tensor, train_Y: torch.Tensor):
        dims = train_Y.shape[1]
        models = []
        for dim in range(dims):
            likelihood = GaussianLikelihood()
            likelihood.noise = self.Yvar[dim].unsqueeze(-1)
            likelihood.noise_covar.raw_noise.requires_grad = False

            model = SingleTaskGP(self.transform_inputs(train_X), train_Y[:, dim].unsqueeze(-1), likelihood=likelihood,
                                 outcome_transform=None)
            models.append(model)
        return ModelListGP(*models)

    def _get_marginallikehood(self):
        return gpytorch.mlls.SumMarginalLogLikelihood(self.gp_model.likelihood, self.gp_model)
