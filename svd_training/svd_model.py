import logging

from transformers import PreTrainedModel
from svd_training.model_keys import get_mlp_names
from svd_training.svd_linear import SVDLinear

_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class SVDForCausalLM(PreTrainedModel):
    def merge(self):
        for layer_index in range(len(self.model.layers)):
            for mlp_name in get_mlp_names():
                try:
                    exec(
                        f"self.model.layers[layer_index].{mlp_name} = self.model.layers[layer_index].{mlp_name}.get_merged_linear()"
                    )
                except AttributeError:
                    continue

        self.lm_head = self.lm_head.get_merged_linear()

    @staticmethod
    def create_from_model(model, rank_fraction):
        _logger.info(f"Building SVD model with rank_fraction={rank_fraction}")
        _logger.info(f"lm_head is substituted with rank_fraction={rank_fraction}")
        model.lm_head = SVDLinear.create_from_weight(
            model.lm_head.weight, rank_fraction
        )
        for layer_index in range(len(model.base_model.layers)):
            for mlp_name in get_mlp_names():
                try:
                    exec(f"weight = model.model.layers[layer_index].{mlp_name}.weight")
                    exec(
                        f"model.model.layers[layer_index].{mlp_name} = SVDLinear.create_from_weight(weight, rank_fraction)"
                    )
                except AttributeError:
                    continue

                _logger.info(
                    f"Layer {layer_index} on {mlp_name} with rank_fraction={rank_fraction} is substituted"
                )
        _logger.info("Adding merge method to the model.")
        model.merge = SVDForCausalLM.merge.__get__(model)
        return model
