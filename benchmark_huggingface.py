import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import time
import argparse
import torch


def warmup(model, tokenizer):
    input = ['']*10
    tokens = tokenizer(input, return_tensors='pt')
    tokens = tokens.to('cuda')
    model.generate(**tokens)


def benchmark_huggingface(
    model_path,
    max_output_len,
    batch_size,
    streaming=False):
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 device_map='auto',
                                                 torch_dtype=torch.float16)
    print(f'model intialized in {model.device}')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print('warm up')
    warmup(model=model, tokenizer=tokenizer)
    print('warm up done')
    print('start benchmarking')
    start_time = time.time()
    input = ['']*batch_size
    tokens = tokenizer(input, return_tensors='pt')
    tokens = tokens.to('cuda')
    if streaming:
        pass
    else:
        new_tokens = model.generate(**tokens, max_new_tokens=max_output_len,
                                    use_cache=True)
        print('generate done')
        for t in new_tokens:
            tokenizer.decode(t, skip_special_tokens=True)
        print('tokenizer done')
        end_time = time.time()
        print('latency: ', end_time - start_time)


parser = argparse.ArgumentParser(description="Benchmark")

# Add arguments to the parser
parser.add_argument("--model_path", type=str, default='/notebooks/llama2-7B-hf')
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--max_output_len", type=int, default=32)

# Parse the command-line arguments
args = parser.parse_args()

benchmark_huggingface(model_path=args.model_path,
                      max_output_len=args.max_output_len,
                      batch_size=args.batch_size)
