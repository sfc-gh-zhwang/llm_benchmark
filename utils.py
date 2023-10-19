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
    sample_mean = np.mean(data)
    sample_std = np.std(data, ddof=1)  # ddof=1 for sample standard deviation
    sample_size = len(data)

    # Calculate the critical value from the t-distribution (two-tailed)
    t_critical = stats.t.ppf((1 + confidence_level) / 2, df=sample_size - 1)

    # Calculate the margin of error
    margin_of_error = t_critical * (sample_std / np.sqrt(sample_size))

    # Calculate the confidence interval
    lower_bound = sample_mean - margin_of_error
    upper_bound = sample_mean + margin_of_error
    return sample_mean, lower_bound, upper_bound
