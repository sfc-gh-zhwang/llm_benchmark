import numpy as np
from scipy import stats


def generate_input(tokenizer, token_num):
    if token_num <= 1:
        return ''
    with open('prompts/climate-6640.txt', 'r') as file:
        # Read the entire file content into a string
        sentence = file.read()
    tokens = tokenizer(sentence)['input_ids'][:token_num]
    sentence = tokenizer.decode(tokens, skip_special_tokens=True)
    return sentence


def generate_inputs(tokenizer, token_num, batch_size):
    return [generate_input(tokenizer, token_num) for _ in range(batch_size)]


def calculate_mean(data):
    # Calculate the sample statistics
    sample_mean = np.mean(data)
    sample_std = np.std(data, ddof=1)  # ddof=1 for sample standard deviation
    return sample_mean, sample_std/sample_mean*100.
