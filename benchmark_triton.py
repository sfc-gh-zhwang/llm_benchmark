import argparse
import multiprocessing as mp
import random
import time
from functools import partial

import numpy as np
import tritonclient.grpc as grpcclient
from transformers import AutoTokenizer
from tritonclient.utils import *

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


def _input(name: str, data: np.ndarray) -> grpcclient.InferInput:
    t = grpcclient.InferInput(name, data.shape, np_to_triton_dtype(data.dtype))
    t.set_data_from_numpy(data)
    return t


def warmup(model_name, client):
    batch_size = 10
    inputs = [
        _input("text", np.array(['hello world, this is to warm up']*batch_size,
                                dtype=object).reshape(-1, 1)),
        _input("max_output_len", np.array([[32]]*batch_size, dtype=np.int32))
    ]
    outputs = [grpcclient.InferRequestedOutput("output")]
    client.infer(model_name, inputs, outputs=outputs)


start_time = None
first_token_time = None
end_time = None
output = None
printed = ''


def stream_callback(a, result, error):
    # print('stream_callback')
    global first_token_time
    global end_time
    global output
    global printed
    if error:
        raise error
    end_time = time.time()
    if first_token_time is None:
        first_token_time = end_time
    output = result.as_numpy('output')
    decoded = output[0][0].decode()
    print(decoded[len(printed):], end='')
    printed = decoded


def benchmark_triton(
    model_name,
    tokenizer_path,
    max_output_len,
    batch_size,
    input_len,
    streaming,
    addr="localhost:8001",
        ):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    prompts = generate_inputs(tokenizer, input_len, batch_size)
    print(prompts[0])
    inputs = [
        _input("text", np.array(prompts, dtype=object).reshape(-1, 1)),
        _input("max_output_len", np.array([[max_output_len]]*batch_size, dtype=np.int32)),
        _input("random_seed",  np.array([[0]]*batch_size, dtype=np.uint64))
    ]
    if streaming:
        with grpcclient.InferenceServerClient(addr, verbose=False) as client:
            result_queue = mp.Queue()
            start_time = time.time()
            client.start_stream(callback=partial(stream_callback, result_queue))
            client.async_stream_infer(model_name, inputs)

        first_token_latency = first_token_time - start_time
        total_duration = end_time - start_time
        streaming_duration = end_time - first_token_time
        tokens = 0
        for i in output:
            tokens += len(tokenizer.encode(i[0].decode())) - 1 # get rid of the start token.

        print('\nfirst_token_latency: ', first_token_latency)
        print('total duration', total_duration)
        print('total tokens generated: ', tokens, 'throughput', tokens/streaming_duration)
        return

    with grpcclient.InferenceServerClient(addr, verbose=False) as client:
        print('warm up')
        warmup(model_name, client)
        print('done warm up')
        outputs = [grpcclient.InferRequestedOutput("output")]
        start_time = time.time()
        response = client.infer(model_name, inputs, outputs=outputs)
        end_time = time.time()
        outputs = response.as_numpy("output")
        for prompt, output in zip(prompts, outputs):
            generated_text = output[0].decode()
            # Print the output to compare with each framework
            print(f"Generated text: {generated_text[:32]}..{generated_text[-32:]}")
        tokens = tokenizer.encode(outputs[0][0].decode())
        print('output_tokens:', len(tokens))
        print('total latency: ', end_time-start_time)

parser = argparse.ArgumentParser(description="Benchmark")

# Add arguments to the parser
parser.add_argument("--model_name", type=str, default='llama-2-70b-chat-hf-ft-streaming')
parser.add_argument("--tokenizer_path", type=str, default='/models/triton/llama-2-70b-hf-ft_tokenizer/1/')
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--max_output_len", type=int, default=32)
parser.add_argument("--input_len", type=int, default=1)
parser.add_argument("--streaming", action='store_true', default=False, help="Whether or not to stream")

# Parse the command-line arguments
args = parser.parse_args()

print('\n=============== Argument ===============')
for key in vars(args):
    print('{}: {}'.format(key, vars(args)[key]))
print('========================================')

benchmark_triton(model_name=args.model_name,
                 tokenizer_path=args.tokenizer_path,
                 max_output_len=args.max_output_len,
                 input_len=args.input_len,
                 batch_size=args.batch_size,
                 streaming=args.streaming)

## python3 b.py --model_name llama-2-70b-hf-ft --input_len 1 --batch_size 1 --max_output_len 2048
## python3 b.py --model_name llama-2-70b-hf-ft --input_len 1024 --max_output_len 1024  --batch_size 32
