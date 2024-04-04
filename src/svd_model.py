import torch
import logging

from functools import lru_cache
from typing import Mapping, Any
from transformers import MistralForCausalLM, MistralConfig
from src.variables import get_mlp_names, get_norm_names

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
        self.U.data = self.U.data.contiguous()
        self.sigma.data = self.sigma.data.contiguous()
        self.V.data = self.V.data.contiguous()
        self.weight.data = self.weight.data.contiguous()
        self._svd_linear_weight = self.weight + self.U @ torch.diag(self.sigma) @ self.V.T

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input @ self._svd_linear_weight.T

    def get_merged_linear(self):
        linear = torch.nn.Linear(self.weight.shape[1], self.weight.shape[0], bias=False)
        linear.weight = torch.nn.Parameter(self._svd_linear_weight, requires_grad=True)
        return linear

    @staticmethod
    def create_from_weight(weight, rank_fraction=0.1, niter=2):
        max_rank = min(weight.shape)
        q = int(max_rank * rank_fraction)
        U, sigma, V = torch.svd_lowrank(weight, q=q, niter=niter)
        new_weight = weight - U @ torch.diag(sigma) @ V.T
        return SVDLinear(U, sigma, V, new_weight)


class SVDMistralForCausalLM(MistralForCausalLM):
    mlp_names = get_mlp_names()
    norm_names = [
        "input_layernorm",
        "post_attention_layernorm",
    ]

    def __init__(self, model):
        super().__init__(model.config)
        self.model = model.model
        self.lm_head = model.lm_head

    def merge_all(self):
        for layer_index in range(len(self.model.layers)):
            for mlp_name in self.mlp_names:
                exec(
                    f"self.model.layers[layer_index].{mlp_name} = self.model.layers[layer_index].{mlp_name}.get_merged_linear()"
                )
        self.lm_head = self.lm_head.get_merged_linear()

    @staticmethod
    def create_from_state_dict(state_dict: Mapping[str, Any]):
        model_name = "mistralai/Mistral-7B-Instruct-v0.1"
        config = MistralConfig.from_pretrained(model_name)
        model = MistralForCausalLM(config)
        model.model.norm.weight = torch.nn.Parameter(state_dict["model.norm.weight"])
        model.model.norm.weight = torch.nn.Parameter(state_dict["model.norm.weight"])
        model.model.embed_tokens.weight = torch.nn.Parameter(
            state_dict["model.embed_tokens.weight"]
        )
        for layer_index in range(len(model.model.layers)):
            for mlp_name in get_mlp_names():
                weight = state_dict[f"model.layers.{layer_index}.{mlp_name}.weight"]
                U = state_dict[f"model.layers.{layer_index}.{mlp_name}.U"]
                sigma = state_dict[f"model.layers.{layer_index}.{mlp_name}.sigma"]
                V = state_dict[f"model.layers.{layer_index}.{mlp_name}.V"]
                exec(
                    f"model.model.layers[layer_index].{mlp_name} = SVDLinear(U, sigma, V, weight)"
                )
            for norm_name in get_norm_names():
                weight_name = f"model.layers.{layer_index}.{norm_name}.weight"
                exec(
                    f"model.model.layers[layer_index].{norm_name}.weight = torch.nn.Parameter(state_dict['{weight_name}'])"
                )
        weight = state_dict[f"lm_head.weight"]
        U = state_dict[f"lm_head.U"]
        sigma = state_dict[f"lm_head.sigma"]
        V = state_dict[f"lm_head.V"]
        model.lm_head = SVDLinear(U, sigma, V, weight)
        return SVDMistralForCausalLM(model)

    @staticmethod
    def create_from_model(model, rank_fraction):
        _logger.info(f"Building SVD model with rank_fraction={rank_fraction}")
        _logger.info(f"lm_head is substituted with rank_fraction={rank_fraction}")
        model.lm_head = SVDLinear.create_from_weight(model.lm_head.weight, rank_fraction)
        for layer_index in range(len(model.base_model.layers)):
            for mlp_name in get_mlp_names():
                exec(f"weight = model.model.layers[layer_index].{mlp_name}.weight")
                exec(
                    f"model.model.layers[layer_index].{mlp_name} = SVDLinear.create_from_weight(weight, rank_fraction)"
                )
                _logger.info(
                    f"Layer {layer_index} on {mlp_name} with rank_fraction={rank_fraction} is substituted"
                )

        return model
