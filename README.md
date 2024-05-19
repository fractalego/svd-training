# Using Singular Value Decomposition (SVD) to reduce the dimensionality of the trainable parameters in a neural network

## Introduction
TBD

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from svd_training.svd_model import SVDForCausalLM

filename = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(filename)
model = AutoModelForCausalLM.from_pretrained(filename)

svd_model = SVDForCausalLM.create_from_model(model, rank_fraction=0.1) # Create the SVD model

### Train the model using your favourite training loop
...
###

svd_model.merge()  # Merge the SVD layers back into the model
svd_model.save_pretrained("svd_model/")  # Save the model
```