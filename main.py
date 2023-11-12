from utils import get_prompts, calculate_stats
from transformers import AutoTokenizer


def get_mean(tokenizer, r=1.0):
    prompts = get_prompts(1024, r)
    tokens = []
    for p in prompts:
        tokens.append(len(tokenizer.tokenize(p)))
    print(calculate_stats(tokens))


tokenizer = AutoTokenizer.from_pretrained('/Users/zhwang/models/llama-2-70b-chat-hf')
get_mean(tokenizer, 0.079)
get_mean(tokenizer, 0.075)
