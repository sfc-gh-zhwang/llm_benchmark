import argparse
import random
import time
from typing import List, Optional

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          TextIteratorStreamer)

long_sentence = "A path from a point approximately 330 metres east of the most south westerly corner of 17 Batherton Close, Widnes and approximately 208 metres east-south-east of the most southerly corner of Unit 3 Foundry Industrial Estate, Victoria Street, Widnes, proceeding in a generally east-north-easterly direction for approximately 28 metres to a point approximately 202 metres east-south-east of the most south-easterly corner of Unit 4 Foundry Industrial Estate, Victoria Street, and approximately 347 metres east of the most south-easterly corner of 17 Batherton Close, then proceeding in a generally northerly direction for approximately 21 metres to a point approximately 210 metres east of the most south-easterly corner of Unit 5 Foundry Industrial Estate, Victoria Street, and approximately 202 metres east-south-east of the most north-easterly corner of Unit 4 Foundry Industrial Estate, Victoria Street, then proceeding in a generally east-north-east direction for approximately 64 metres to a point approximately 282 metres east-south-east of the most easterly corner of Unit 2 Foundry Industrial Estate, Victoria Street, Widnes and approximately 259 metres east of the most southerly corner of Unit 4 Foundry Industrial Estate, Victoria Street, then proceeding in a generally east-north-east direction for approximately 350 metres to a point approximately 3 metres west-north-west of the most north westerly corner of the boundary fence of the scrap metal yard on the south side of Cornubia Road, Widnes, and approximately 47 metres west-south-west of the stub end of Cornubia Road be diverted to a 3 metre wide path from a point approximately " \
    "183 metres east-south-east of the most easterly corner of Unit 5 Foundry Industrial Estate, Victoria Street and approximately 272 metres east of the most north-easterly corner of 26 Ann Street West, Widnes, then proceeding in a generally north easterly direction for approximately 58 metres to a point approximately 216 metres east-south-east of the most easterly corner of Unit 4 Foundry Industrial Estate, Victoria Street and approximately 221 metres east of the most southerly corner of Unit 5 Foundry Industrial Estate, Victoria Street, then proceeding in a generally easterly direction for approximately 45 metres to a point approximately 265 metres east-south-east of the most north-easterly corner of Unit 3 Foundry Industrial Estate, Victoria Street and approximately 265 metres east of the most southerly corner of Unit 5 Foundry Industrial Estate, Victoria Street, then proceeding in a generally east-south-east direction for approximately 102 metres to a point approximately 366 metres east-south-east of the most easterly corner of Unit 3 Foundry Industrial Estate, Victoria Street and approximately 463 metres east of the most north easterly corner of 22 Ann Street West, Widnes, then proceeding in a generally north-north-easterly direction for approximately 19 metres to a point approximately 368 metres east-south-east of the most easterly corner of Unit 3 Foundry Industrial Estate, Victoria Street and approximately 512 metres east of the most south easterly corner of 17 Batherton Close, Widnes then proceeding in a generally east-south, easterly direction for approximately 16 metres to a point approximately 420 metres east-south-east of the " \
    "most southerly corner of Unit 2 Foundry Industrial Estate, Victoria Street and approximately 533 metres east of the most south-easterly corner of 17 Batherton Close, then proceeding in a generally east-north-easterly direction for approximately 240 metres to a point approximately 606 metres east of the most northerly corner of Unit 4 Foundry Industrial Estate, Victoria Street and approximately 23 metres south of the most south westerly corner of the boundary fencing of the scrap metal yard on the south side of Cornubia Road, Widnes, then proceeding in a generally northern direction for approximately 44 metres to a point approximately 3 metres west-north-west of the most north westerly corner of the boundary fence of the scrap metal yard on the south side of Cornubia Road and approximately 47 metres west-south-west of the stub end of Cornubia Road."

words = long_sentence.split(' ')
num_words = len(words)


# generate a random sentence with token number = token_num
def generate_input(tokenizer, token_num):
    if token_num <= 1:
        return ''
    sentence = ''
    for i in range(token_num * 3):
        word_index = random.randint(0, num_words-1)
        word = words[word_index]
        if sentence == '':
            sentence = word
        else:
            sentence += ' ' + word
    tokens = tokenizer(sentence)['input_ids'][:token_num]
    sentence = tokenizer.decode(tokens, skip_special_tokens=True)
    print(f'generated a random sentence with {token_num} tokens, first 32 charactors are {sentence[:32]}')
    return sentence


def generate_inputs(tokenizer, token_num, batch_size):
    # make the rng deterministic
    random.seed(42)
    return [generate_input(tokenizer, token_num) for _ in range(batch_size)]



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
            print('first token generated')
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
        print(self.tokens)

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
    streaming=False):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 device_map='auto',
                                                 torch_dtype=torch.float16)
    print(f'model intialized in {model.device}')
    print('warm up')
    warmup(model=model, tokenizer=tokenizer)
    print('warm up done')
    print('start benchmarking')
    prompts = generate_inputs(tokenizer, input_len, batch_size)
    start_time = time.time()
    tokens = tokenizer(prompts, return_tensors='pt')
    tokens = tokens.to('cuda')
    if streaming:
        streamer = BatchTextIteratorStreamer(batch_size=batch_size, tokenizer=tokenizer, skip_prompt=True)
        model.generate(**tokens, streamer=streamer,
                       max_new_tokens=max_output_len,
                       use_cache=True)
        end_time = time.time()
        streaming_duration = end_time - streamer.first_token_time
        print('\nfirst_token_latency: ', streamer.first_token_time-start_time)
        print('total duration', end_time - start_time)
        print('total tokens generated: ', streamer.tokens, 'throughput',
              streamer.tokens/streaming_duration)
    else:
        new_tokens = model.generate(**tokens, max_new_tokens=max_output_len,
                                    use_cache=True)
        print('generate done', new_tokens.shape)
        generated_texts = []
        for t in new_tokens:
            generated_texts.append(tokenizer.decode(t, skip_special_tokens=True))
        print('tokenizer done')
        end_time = time.time()
        print('latency: ', end_time - start_time)
        for prompt, generated_text in zip(prompts, generated_texts):
            generated_text = generated_text[len(prompt):]
            print(f"Generated text: {generated_text[:32]}..{generated_text[-32:]}")


parser = argparse.ArgumentParser(description="Benchmark")

# Add arguments to the parser
parser.add_argument("--model_path", type=str, default='/notebooks/llama2-7B-hf')
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--max_output_len", type=int, default=32)
parser.add_argument("--input_len", type=int, default=1)
parser.add_argument("--use_cache", action='store_false', default=True, help="Whether or not to use cache")
parser.add_argument("--streaming", action='store_true', default=False, help="Whether or not to stream")

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
                      streaming=args.streaming)
