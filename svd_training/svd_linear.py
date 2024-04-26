import torch
import logging

_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class SVDLinear(torch.nn.modules.module.Module):
    U: torch.Tensor
    sigma: torch.Tensor
    V: torch.Tensor
    weight: torch.Tensor

    def __init__(
        self,
        U: torch.Tensor,
        sigma: torch.Tensor,
        V: torch.Tensor,
        weight: torch.Tensor,
    ):
        super().__init__()
        self.U = torch.nn.Parameter(U, requires_grad=False)
        self.sigma = torch.nn.Parameter(sigma, requires_grad=True)
        self.V = torch.nn.Parameter(V, requires_grad=False)
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.U.data = self.U.data.contiguous()
        self.sigma.data = self.sigma.data.contiguous()
        self.V.data = self.V.data.contiguous()
        self.weight.data = self.weight.data.contiguous()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input @ self._get_svd_weight().T

    def get_merged_linear(self):
        linear = torch.nn.Linear(self.weight.shape[1], self.weight.shape[0], bias=False)
        linear.weight = torch.nn.Parameter(self._get_svd_weight(), requires_grad=True)
        return linear

    def _get_svd_weight(self):
        return self.weight + self.U @ torch.diag(self.sigma) @ self.V.T

    @staticmethod
    def create_from_weight(weight, rank_fraction=0.1, niter=2):
        max_rank = min(weight.shape)
        q = int(max_rank * rank_fraction)
        U, sigma, V = torch.svd_lowrank(weight, q=q, niter=niter)
        new_weight = weight - U @ torch.diag(sigma) @ V.T
        return SVDLinear(U, sigma, V, new_weight)
