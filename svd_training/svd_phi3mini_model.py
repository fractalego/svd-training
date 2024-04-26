import logging
from typing import Mapping, Any

import torch
from transformers import Phi3Config, Phi3ForCausalLM

from svd_training.model_keys import get_mlp_names, get_norm_names
from svd_training.svd_linear import SVDLinear

_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class SVDPhi3MiniForCausalLM(Phi3ForCausalLM):
    def __init__(self, model):
        super().__init__(model.config)
        self.model = model.model
        self.lm_head = model.lm_head

    def merge_all(self):
        for layer_index in range(len(self.model.layers)):
            for mlp_name in get_mlp_names("microsoft/Phi-3-mini-4k-instruct"):
                exec(
                    f"self.model.layers[layer_index].{mlp_name} = self.model.layers[layer_index].{mlp_name}.get_merged_linear()"
                )
        self.lm_head = self.lm_head.get_merged_linear()

    @staticmethod
    def create_from_state_dict(state_dict: Mapping[str, Any]):
        model_name = "microsoft/Phi-3-mini-4k-instruct"
        config = Phi3Config.from_pretrained(model_name)
        model = Phi3ForCausalLM(config)
        model.model.norm.weight = torch.nn.Parameter(state_dict["model.norm.weight"])
        model.model.norm.weight = torch.nn.Parameter(state_dict["model.norm.weight"])
        model.model.embed_tokens.weight = torch.nn.Parameter(
            state_dict["model.embed_tokens.weight"]
        )
        for layer_index in range(len(model.model.layers)):
            for mlp_name in get_mlp_names(model_name):
                weight = state_dict[f"model.layers.{layer_index}.{mlp_name}.weight"]
                U = state_dict[f"model.layers.{layer_index}.{mlp_name}.U"]
                sigma = state_dict[f"model.layers.{layer_index}.{mlp_name}.sigma"]
                V = state_dict[f"model.layers.{layer_index}.{mlp_name}.V"]
                exec(
                    f"model.model.layers[layer_index].{mlp_name} = SVDLinear(U, sigma, V, weight)"
                )
            for norm_name in get_norm_names(model_name):
                weight_name = f"model.layers.{layer_index}.{norm_name}.weight"
                exec(
                    f"model.model.layers[layer_index].{norm_name}.weight = torch.nn.Parameter(state_dict['{weight_name}'])"
                )
        weight = state_dict[f"lm_head.weight"]
        U = state_dict[f"lm_head.U"]
        sigma = state_dict[f"lm_head.sigma"]
        V = state_dict[f"lm_head.V"]
        model.lm_head = SVDLinear(U, sigma, V, weight)
        return SVDPhi3MiniForCausalLM(model)

    @staticmethod
    def create_from_model(model, rank_fraction):
        _logger.info(f"Building SVD model with rank_fraction={rank_fraction}")
        _logger.info(f"lm_head is substituted with rank_fraction={rank_fraction}")
        model.lm_head = SVDLinear.create_from_weight(
            model.lm_head.weight, rank_fraction
        )
        for layer_index in range(len(model.base_model.layers)):
            for mlp_name in get_mlp_names("microsoft/Phi-3-mini-4k-instruct"):
                exec(f"weight = model.model.layers[layer_index].{mlp_name}.weight")
                exec(
                    f"model.model.layers[layer_index].{mlp_name} = SVDLinear.create_from_weight(weight, rank_fraction)"
                )
                _logger.info(
                    f"Layer {layer_index} on {mlp_name} with rank_fraction={rank_fraction} is substituted"
                )

        return SVDPhi3MiniForCausalLM(model)
