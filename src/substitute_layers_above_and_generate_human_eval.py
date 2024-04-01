import torch

from collections import OrderedDict
from human_eval.data import read_problems, write_jsonl
from tqdm import tqdm

from transformers import AutoTokenizer, MistralForCausalLM, MistralConfig

model_name = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = MistralConfig.from_pretrained(model_name)


class SVDLinear(torch.nn.modules.module.Module):
    U: torch.Tensor
    sigma: torch.Tensor
    V: torch.Tensor

    def __init__(self, U, sigma, V):
        super().__init__()
        self.U = torch.nn.Parameter(U)
        self.sigma = torch.nn.Parameter(sigma)
        self.V = torch.nn.Parameter(V)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input @ (self.U @ torch.diag(self.sigma) @ self.V.T).T


class SVDMistralForCausalLM(MistralForCausalLM):
    def __init__(self, state_dict):
        super().__init__(MistralConfig.from_pretrained(model_name))
        mlp_names = [
            "mlp.down_proj",
            "mlp.up_proj",
            "mlp.gate_proj",
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "input_layernorm",
            "post_attention_layernorm",
        ]
        self.lm_head.weight = torch.nn.Parameter(state_dict["lm_head.weight"])
        self.model.norm.weight = torch.nn.Parameter(state_dict["model.norm.weight"])
        self.model.embed_tokens.weight = torch.nn.Parameter(
            state_dict["model.embed_tokens.weight"]
        )
        for layer_index in range(len(self.model.layers)):
            for mlp_name in mlp_names:
                weight_name = f"model.layers.{layer_index}.{mlp_name}.weight"
                if weight_name in state_dict:
                    exec(
                        f"self.model.layers[layer_index].{mlp_name}.weight = torch.nn.Parameter(state_dict['{weight_name}'])"
                    )
                    continue

                U = state_dict[f"model.layers.{layer_index}.{mlp_name}.U"]
                sigma = state_dict[f"model.layers.{layer_index}.{mlp_name}.sigma"]
                V = state_dict[f"model.layers.{layer_index}.{mlp_name}.V"]
                exec(
                    f"self.model.layers[layer_index].{mlp_name} = SVDLinear(U, sigma, V)"
                )


def merge_state_dicts(original_state_dict, new_state_dict, threshold_layer):
    state_dict = OrderedDict()
    for original_key in original_state_dict.keys():
        if "model.layers" in original_key:
            layer_index = int(original_key.split(".")[2])
            if layer_index <= threshold_layer:
                state_dict[original_key] = original_state_dict[original_key]
        else:
            state_dict[original_key] = original_state_dict[original_key]

    for new_key in new_state_dict.keys():
        if "model.layers" in new_key:
            layer_index = int(new_key.split(".")[2])
            if layer_index > threshold_layer:
                state_dict[new_key] = new_state_dict[new_key]

    return state_dict


def filter_layers(state_dict, threshold_layer):
    new_state_dict = OrderedDict()
    for key in state_dict.keys():
        if "model.layers" in key:
            layer_index = int(key.split(".")[2])
            if layer_index > threshold_layer:
                new_state_dict[key] = state_dict[key]

    return new_state_dict


def generate_one_completion(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(
        input_ids.cuda(),
        max_length=1024,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(output[0]).replace("<s>", "").replace("</s>", "").strip()


if __name__ == "__main__":
    problems = read_problems()
    lowest_threshold_layer = 27
    new_model_state_dict = filter_layers(
        torch.load("../mistral-7b-instruct-laser-distil-backward-0.25"),
        lowest_threshold_layer,
    )
    original_model_state_dict = torch.load("../original_state_dict")
    for threshold_layer in reversed(range(lowest_threshold_layer, 32)):
        print(f"Threshold layer: {threshold_layer}")
        state_dict = merge_state_dicts(
            original_model_state_dict,
            new_model_state_dict,
            threshold_layer=threshold_layer,
        )
        model = SVDMistralForCausalLM(state_dict)
        model.half().cuda()
        num_samples_per_task = 1
        samples = [
            dict(
                task_id=task_id,
                completion=generate_one_completion(problems[task_id]["prompt"]),
            )
            for task_id in tqdm(problems)
            for _ in range(num_samples_per_task)
        ]
        del model
        filename = f"samples-threshold_layer-{threshold_layer}.jsonl"
        write_jsonl(filename, samples)
