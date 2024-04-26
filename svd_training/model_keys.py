from typing import List


def get_mlp_names(model_name: str) -> List[str]:
    if model_name == "mistralai/Mistral-7B-Instruct-v0.1":
        return [
            "mlp.down_proj",
            "mlp.up_proj",
            "mlp.gate_proj",
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
        ]

    if "microsoft/Phi-3-mini" in model_name:
        return [
            "mlp.down_proj",
            "mlp.gate_up_proj",
            "self_attn.qkv_proj",
            "self_attn.o_proj",
        ]


def get_norm_names(model_name) -> List[str]:
    if model_name == "mistralai/Mistral-7B-Instruct-v0.1":
        return [
            "input_layernorm",
            "post_attention_layernorm",
        ]

    if "microsoft/Phi-3-mini" in "microsoft/Phi-3-mini":
        return [
            "input_layernorm",
            "post_attention_layernorm",
        ]


def supported_models() -> List[str]:
    return [
        "mistralai/Mistral-7B-Instruct-v0.1",
        "microsoft/Phi-3-mini-128k-instruct",
        "microsoft/Phi-3-mini-4k-instruct",
    ]
