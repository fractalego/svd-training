from typing import List


def get_mlp_names() -> List[str]:
    return [
        "mlp.down_proj",
        "mlp.up_proj",
        "mlp.gate_proj",
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_up_proj",
        "self_attn.qkv_proj",
    ]



def get_norm_names() -> List[str]:
    return [
        "input_layernorm",
        "post_attention_layernorm",
    ]
