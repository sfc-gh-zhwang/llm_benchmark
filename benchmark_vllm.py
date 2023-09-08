# Modify from https://github.com/vllm-project/vllm/blob/852ef5b4f5481ce526c804ea234d1de0df91f48d/benchmarks/benchmark_throughput.py
import argparse
import random
import time

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

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


def warmup(llm):
    for i in range(10):
        sampling_params = SamplingParams(
            n=1,
            temperature=1.0,
            top_p=1.0,
            use_beam_search=False,
            ignore_eos=True,
            max_tokens=32,
        )
        # FIXME(woosuk): Do not use internal method.
        llm._add_request(
            prompt='hello world, this is to warm up',
            prompt_token_ids=None,
            sampling_params=sampling_params,
        )

    llm._run_engine(use_tqdm=True)



def benchmark_vllm(
    model_path,
    max_output_len,
    batch_size,
    input_len,
        ):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print('init llm')
    llm = LLM(model=model_path,
              tokenizer=model_path,
              tensor_parallel_size=8,
              max_num_seqs=args.batch_size,
              max_num_batched_tokens=batch_size * (input_len+max_output_len)
              )
    print('done init llm')

    print('warm up')
    warmup(llm)
    print('done warm up')

    inputs = generate_inputs(tokenizer, input_len, batch_size)
    # Add the requests to the engine.
    for i in range(batch_size):
        sampling_params = SamplingParams(
            n=1,
            temperature=1.0,
            top_p=1.0,
            use_beam_search=False,
            ignore_eos=True,
            max_tokens=max_output_len,
        )
        # FIXME(woosuk): Do not use internal method.
        llm._add_request(
            prompt=inputs[i],
            prompt_token_ids=None,
            sampling_params=sampling_params,
        )

    start = time.time()
    # FIXME(woosuk): Do use internal method.
    llm._run_engine(use_tqdm=True)
    end = time.time()

    elapsed_time = end - start
    total_num_tokens = batch_size * max_output_len
    print(f"Throughput: {total_num_tokens / elapsed_time} tokens/s")
    print(f"Total latency: {elapsed_time}")


parser = argparse.ArgumentParser(description="Benchmark")

# Add arguments to the parser

parser.add_argument("--model_path", type=str, default='/models/llama2-70B-hf')
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--max_output_len", type=int, default=32)
parser.add_argument("--input_len", type=int, default=1)
parser.add_argument("--streaming", action='store_true', default=False, help="Whether or not to stream")

args = parser.parse_args()

benchmark_vllm(model_path=args.model_path,
               max_output_len=args.max_output_len,
               batch_size=args.batch_size,
               input_len=args.input_len)
