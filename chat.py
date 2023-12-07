# PLEASE DON'T CHANGE THE CODE UNLESS YOU'RE FAMILIAR WITH PROGRAMMING. IT MIGHT CAUSE ERRORS OTHERWISE.
# IMPORT NECESSARY MODULES 
import torch
import transformers
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# DEFINE FUNCTIONS
def get_path():
    return input("Enter model path: ")
    
def get_length():
    return input("Enter response length: ")

def load_model(model_path):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    return model

def load_tokenizer(model_path):
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    return tokenizer

def generate_text(model, tokenizer, sequence, max_length):
    inputs = tokenizer(sequence, return_tensors="pt")
    input_ids = inputs.input_ids
    final_outputs = model.generate(
        input_ids,
        do_sample=True,
        max_length=max_length,
        pad_token_id=model.config.eos_token_id,
        top_k=50,
        top_p=0.95,
    )
    return tokenizer.decode(final_outputs[0], skip_special_tokens=True)

# GET OUTPUT
model_path = get_path()
max_len = int(get_length())

model = load_model(model_path)
tokenizer = load_tokenizer(model_path)

sequence = input("Chat to model: ")
result = generate_text(model, tokenizer, sequence, max_len)
print(result)