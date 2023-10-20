import argparse
import time
from typing import List, Optional

import torch
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          TextIteratorStreamer)

from utils import calculate_mean, generate_inputs
import deepspeed

class BatchTextIteratorStreamer(TextIteratorStreamer):
    def __init__(self, batch_size: int, tokenizer: "AutoTokenizer", skip_prompt: bool = False, timeout: Optional[float] = None, **decode_kwargs):
        super().__init__(tokenizer, skip_prompt, timeout, **decode_kwargs)
        self.batch_size = batch_size
        self.token_cache = [[] for _ in range(batch_size)]
        self.print_len = [0 for _ in range(batch_size)]
        self.generate_exception = None
        self.tokens = 0
        self.first_token_time = None

    def put(self, value):
        if len(value.shape) != 2:
            value = torch.reshape(value, (self.batch_size, value.shape[0] // self.batch_size))

        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        printable_texts = list()
        self.tokens += self.batch_size
        if self.first_token_time is None:
            self.first_token_time = time.time()
        for idx in range(self.batch_size):
            self.token_cache[idx].extend(value[idx].tolist())
            text = self.tokenizer.decode(self.token_cache[idx], **self.decode_kwargs)

            if text.endswith("\n"):
                printable_text = text[self.print_len[idx] :]
                self.token_cache[idx] = []
                self.print_len[idx] = 0
                # If the last token is a CJK character, we print the characters.
            elif len(text) > 0 and self._is_chinese_char(ord(text[-1])):
                printable_text = text[self.print_len[idx] :]
                self.print_len[idx] += len(printable_text)
            else:
                printable_text = text[self.print_len[idx] : text.rfind(" ") + 1]
                self.print_len[idx] += len(printable_text)
            printable_texts.append(printable_text)

        self.on_finalized_text(printable_texts)

    def end(self):
        printable_texts = list()
        for idx in range(self.batch_size):
            if len(self.token_cache[idx]) > 0:
                text = self.tokenizer.decode(self.token_cache[idx], **self.decode_kwargs)
                printable_text = text[self.print_len[idx] :]
                self.token_cache[idx] = []
                self.print_len[idx] = 0
            else:
                printable_text = ""
            printable_texts.append(printable_text)

        self.next_tokens_are_prompt = True
        self.on_finalized_text(printable_texts, stream_end=True)

    def on_finalized_text(self, texts: List[str], stream_end: bool = False):
        self.text_queue.put(texts, timeout=self.timeout)
        if stream_end:
            self.text_queue.put(self.stop_signal, timeout=self.timeout)


def warmup(model, tokenizer):
    input = ['hello world this is to warm up']*16
    tokens = tokenizer(input, return_tensors='pt')
    tokens = tokens.to('cuda')
    model.generate(**tokens, max_new_tokens=64)


def benchmark_huggingface(
    model_path,
    max_output_len,
    batch_size,
    input_len,
    streaming,
    n):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 device_map='auto',
                                                 torch_dtype=torch.float16)
    print(f'model intialized in {model.device}')
    warmup(model, tokenizer)
    print('start benchmarking')
    prompts = generate_inputs(tokenizer, input_len, batch_size)
    print(f"Prompt: {prompts[0][:32]}..{prompts[0][-32:]}")
    if streaming:
        first_token_latency = [0]*n
        throughput = [0]*n
        latency = [0]*n
        for i in tqdm(range(n)):
            start_time = time.time()
            tokens = tokenizer(prompts, return_tensors='pt')
            tokens = tokens.to(model.device)
            streamer = BatchTextIteratorStreamer(batch_size=batch_size, tokenizer=tokenizer, skip_prompt=True)
            model.generate(**tokens, streamer=streamer,
                           max_new_tokens=max_output_len,
                           use_cache=True)
            end_time = time.time()
            latency[i] = end_time - start_time
            first_token_latency[i] = streamer.first_token_time - start_time
            throughput[i] = (input_len * batch_size + streamer.tokens)/latency[i]
        print('first_token_latency: ', calculate_mean(first_token_latency))
        print('latency', calculate_mean(latency))
        print('throughput: ', calculate_mean(throughput))
        return

    # Non-streaming
    model = deepspeed.init_inference(
         model=model, mp_size=4, dtype=torch.float16, replace_method="auto", replace_with_kernel_inject=True)
    latency = []
    print('warming up')
    tokens = tokenizer(prompts, return_tensors='pt')
    tokens = tokens.to('cuda')
    new_tokens = model.generate(**tokens, max_new_tokens=max_output_len)
    print('done warm up')
    for i in tqdm(range(n)):
        start_time = time.time()
        tokens = tokenizer(prompts, return_tensors='pt')
        tokens = tokens.to('cuda')
        new_tokens = model.generate(**tokens, max_new_tokens=max_output_len,
                                    use_cache=True)
        new_tokens = new_tokens[:, input_len:]
        for t in new_tokens:
            generated_text = tokenizer.decode(t, skip_special_tokens=True)
        end_time = time.time()
        latency.append(end_time-start_time)
    tokens = tokenizer.encode(generated_text)
    print('output_tokens:', len(tokens))
    print(f"Generated text: {generated_text[:32]}..{generated_text[-32:]}")
    print(f'latency: {calculate_mean(latency)}')


parser = argparse.ArgumentParser(description="Benchmark")

# Add arguments to the parser
parser.add_argument("--model_path", type=str, default='/notebooks/llama2-7B-hf')
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--max_output_len", type=int, default=32)
parser.add_argument("--input_len", type=int, default=1)
parser.add_argument("--use_cache", action='store_false', default=True, help="Whether or not to use cache")
parser.add_argument("--streaming", action='store_true', default=False, help="Whether or not to stream")
parser.add_argument("--n", type=int, default=50)

# Parse the command-line arguments
args = parser.parse_args()

print('\n=============== Argument ===============')
for key in vars(args):
    print('{}: {}'.format(key, vars(args)[key]))
print('========================================')

benchmark_huggingface(model_path=args.model_path,
                      max_output_len=args.max_output_len,
                      batch_size=args.batch_size,
                      input_len=args.input_len,
                      streaming=args.streaming,
                      n=args.n)
