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


def calculate_mean(data, confidence_level=0.95):
    # Calculate the sample statistics
    sample_size = len(data)
    if sample_size == 0:
        raise ValueError("input is empty")
    if sample_size == 1:
        return data[0], data[0], data[0]
    mean = np.mean(data)
    sample_std = np.std(data, ddof=1)  # ddof=1 for sample standard deviation

    # Calculate the critical value from the t-distribution (two-tailed)
    t_critical = stats.t.ppf((1 + confidence_level) / 2, df=sample_size - 1)

    # Calculate the margin of error
    margin_of_error = t_critical * (sample_std / np.sqrt(sample_size))

    # Calculate the confidence interval
    lb = mean - margin_of_error
    up = mean + margin_of_error
    return mean, lb, up, f'{mean:.4f}[{lb:.4f}, {up:.4f}]'
