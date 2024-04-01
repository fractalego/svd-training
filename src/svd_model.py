import logging
import torch

from transformers import MistralForCausalLM

_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class SVDLinear(torch.nn.modules.module.Module):
    U: torch.Tensor
    sigma: torch.Tensor
    V: torch.Tensor
    weight: torch.Tensor

    def __init__(self, U, sigma, V, weight):
        super().__init__()
        self.U = torch.nn.Parameter(U, requires_grad=False)
        self.sigma = torch.nn.Parameter(sigma, requires_grad=True)
        self.V = torch.nn.Parameter(V, requires_grad=False)
        self.weight = torch.nn.Parameter(weight, requires_grad=False)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input @ self._get_svd_weight()

    def get_merged_linear(self):
        linear = torch.nn.Linear(self.weight.shape[1], self.weight.shape[0], bias=False)
        linear.weight = self.weight + self._get_svd_weight()
        return linear

    def _get_svd_weight(self):
        return self.U @ torch.diag(self.sigma) @ self.V.T

    @staticmethod
    def create_from_weight(weight, rank_fraction=0.1, niter=2):
        max_rank = min(weight.shape)
        q = int(max_rank * rank_fraction)
        U, sigma, V = torch.svd_lowrank(weight, q=q, niter=niter)
        new_weight = weight - U @ torch.diag(sigma) @ V.T
        return SVDLinear(U, sigma, V, new_weight)


class SVDMistralForCausalLM(MistralForCausalLM):
    mlp_names = [
        "mlp.down_proj",
        "mlp.up_proj",
        "mlp.gate_proj",
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
    ]
    norm_names = [
        "input_layernorm",
        "post_attention_layernorm",
    ]

    def __init__(self, model, rank_fraction):
        super().__init__(model.config)
        self._build_from_model(model, rank_fraction)

    def merge_all(self):
        for layer_index in range(len(self.model.layers)):
            for mlp_name in self.mlp_names:
                exec(
                    f"self.model.layers[layer_index].{mlp_name} = self.model.layers[layer_index].{mlp_name}.get_merged_linear()"
                )
        self.lm_head = self.lm_head.get_merged_linear()
        self.model.embed_tokens = self.model.embed_tokens.get_merged_linear()

    def _build_from_model(self, model, rank_fraction):
        _logger.info(f"Building SVD model with rank_fraction={rank_fraction}")
        _logger.info(f"lm_head and embed_tokens are substituted with rank_fraction={rank_fraction}")
        self.lm_head = SVDLinear.create_from_weight(model.lm_head.weight, rank_fraction)
        self.model.embed_tokens = SVDLinear.create_from_weight(
            model.model.embed_tokens.weight, rank_fraction
        )
        for layer_index in range(len(model.base_model.layers)):
            for mlp_name in self.mlp_names:
                exec(f"weight = model.model.layers[layer_index].{mlp_name}.weight")
                exec(
                    f"self.model.layers[layer_index].{mlp_name} = SVDLinear.create_from_weight(weight, rank_fraction)"
                )
                _logger.info(
                    f"Layer {layer_index} on {mlp_name} with rank_fraction={rank_fraction} is substituted"
                )

        _logger.info(f"Copying norms from the original model to the new one")
        self.model.norm.weight = model.model.norm.weight
        for layer_index in range(len(model.base_model.layers)):
            for norm_name in self.norm_names:
                exec(
                    f"self.model.layers[layer_index].{norm_name} = model.model.layers[layer_index].{norm_name}"
                )
