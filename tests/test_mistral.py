import unittest

from transformers import AutoModelForCausalLM, AutoTokenizer
from svd_training.svd_model import SVDForCausalLM

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
original_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad
)
svd_model = SVDForCausalLM.create_from_model(model, rank_fraction=0.25)


class TestMistral(unittest.TestCase):
    def test_1_mistral_can_be_loaded(self):
        self.assertIsNotNone(svd_model)

    def test_2_trainable_parameters_count(self):
        total_params = sum(p.numel() for p in svd_model.parameters() if p.requires_grad)
        expected_params = 131_519_488
        self.assertEqual(expected_params, total_params)

    def test_2_trainable_parameters_count_after_training(self):
        svd_model.train()
        total_params = sum(p.numel() for p in svd_model.parameters() if p.requires_grad)
        expected_params = 131_519_488
        self.assertEqual(expected_params, total_params)

    def test_3_generation(self):
        input_ids = tokenizer.encode("Hello, my name is", return_tensors="pt")
        svd_model.half().cuda()
        output = svd_model.generate(
            input_ids.cuda(),
            max_length=12,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        print(tokenizer.decode(output[0]))
        self.assertIsNotNone(output)

    def test_4_merge(self):
        svd_model.merge()
        merged_trainable_params = sum(
            p.numel() for p in svd_model.parameters() if p.requires_grad
        )
        self.assertEqual(original_trainable_params, merged_trainable_params)
