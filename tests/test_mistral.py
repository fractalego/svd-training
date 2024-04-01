import unittest

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.svd_model import SVDMistralForCausalLM

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
svd_model = SVDMistralForCausalLM(model, rank_fraction=0.25)


class TestMistral(unittest.TestCase):
    def test_mistral_can_be_loaded(self):
        self.assertIsNotNone(svd_model)

    def test_trainable_parameters_count(self):
        total_params = sum(p.numel() for p in svd_model.parameters() if p.requires_grad)
        expected_params = 448_512
        self.assertEqual(expected_params, total_params)

    def test_generation(self):
        input_ids = tokenizer.encode("Hello, my name is")
        output = svd_model.generate(input_ids, max_length=20)
        print(tokenizer.decode(output[0]))
        self.assertIsNotNone(output)

    def test_merge(self):
        original_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        merged_model = svd_model.merge_all()
        merged_trainable_params = sum(p.numel() for p in merged_model.parameters() if p.requires_grad)
        self.assertEqual(original_trainable_params, merged_trainable_params)

