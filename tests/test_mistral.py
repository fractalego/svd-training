import os
import unittest
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

from src.svd_model import SVDMistralForCausalLM

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
original_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
_filename = "mistral_svd_model.psd"
if os.path.exists(_filename):
    del model
    svd_model = SVDMistralForCausalLM.create_from_state_dict(torch.load(_filename))

else:
    svd_model = SVDMistralForCausalLM.create_from_model(model, rank_fraction=0.25)
    torch.save(svd_model.state_dict(), _filename)
    del model


class TestMistral(unittest.TestCase):
    def test_1_mistral_can_be_loaded(self):
        self.assertIsNotNone(svd_model)

    def test_2_trainable_parameters_count(self):
        total_params = sum(p.numel() for p in svd_model.parameters() if p.requires_grad)
        expected_params = 131_519_488
        self.assertEqual(expected_params, total_params)

    def test_3_generation(self):
        input_ids = tokenizer.encode("Hello, my name is", return_tensors="pt")
        output = svd_model.generate(input_ids, max_length=12)
        print(tokenizer.decode(output[0]))
        self.assertIsNotNone(output)

    def test_4_merge(self):
        svd_model.merge_all()
        merged_trainable_params = sum(p.numel() for p in svd_model.parameters() if p.requires_grad)
        self.assertEqual(original_trainable_params, merged_trainable_params)
