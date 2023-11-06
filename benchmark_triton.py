import argparse
import multiprocessing as mp
import time
from functools import partial

import numpy as np
import tritonclient.grpc as grpcclient
from tqdm import tqdm
from transformers import AutoTokenizer
from tritonclient.utils import *

from utils import calculate_mean, generate_inputs


def _input(name: str, data: np.ndarray) -> grpcclient.InferInput:
    t = grpcclient.InferInput(name, data.shape, np_to_triton_dtype(data.dtype))
    t.set_data_from_numpy(data)
    return t


def warmup(model_name, client):
    batch_size = 1
    inputs = [
        _input("text", np.array(['hello world, this is to warm up']*batch_size,
                                dtype=object).reshape(-1, 1)),
        _input("max_output_len", np.array([[32]]*batch_size, dtype=np.int32))
    ]
    outputs = [grpcclient.InferRequestedOutput("output")]
    client.infer(model_name, inputs, outputs=outputs)


def stream_callback(index, result, error):
    print("index:", index)
    # print('stream_callback')
    global first_token_time
    global end_time
    global output
    if error:
        raise error
    end_time = time.time()
    output = result.as_numpy('output')
    if first_token_time is None:
        first_token_time = end_time


first_token_latency = None
first_token_time = None
latency = None
throughput = None
output = None
start_time = None
output_tokens = None
end_time = None

def start_stream(addr, model_name, inputs, index):
    print("start_stream index:", index)
    with grpcclient.InferenceServerClient(addr, verbose=False) as client:
        first_token_time[index] = None
        start_time = time.time()
        client.start_stream(callback=partial(stream_callback, index))
        client.async_stream_infer(model_name, inputs)

    global end_time
    first_token_latency[i] = first_token_time - start_time
    latency[i] = end_time - start_time
    tokens = 0
    for ot in output:
        output_len = len(tokenizer.encode(ot[0].decode())) - 1
        output_tokens += output_len
        tokens += input_len + output_len    # get rid of the start token.
    throughput[i] = tokens/latency[i]


def benchmark_triton(
    model_name,
    tokenizer_path,
    max_output_len,
    batch_size,
    input_len,
    streaming,
    parallelism,
    n,
    addr="localhost:8001",
        ):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    prompts = generate_inputs(tokenizer, input_len, batch_size)
    print(f"Prompt: {prompts[0][:32]}..{prompts[0][-512:]}")
    inputs = [
        _input("text", np.array(prompts, dtype=object).reshape(-1, 1)),
        _input("max_output_len", np.array([[max_output_len]]*batch_size, dtype=np.int32)),
    ]
    if streaming:
        print("parallelism:", parallelism)
        global first_token_latency, first_token_time, latency, throughput, output, start_time, end_time
        first_token_latency = [0]*n*parallelism
        first_token_time = [0]*n*parallelism
        latency = [0]*n*parallelism
        throughput = [0]*n*parallelism
        output = [0]*n*parallelism
        start_time = [0]*n*parallelism
        end_time = [0]*n*parallelism
        output_tokens = 0
        for i in tqdm(range(n)):
            for p in range(parallelism):
                start_stream(addr=addr,
                             model_name=model_name,
                             inputs=inputs,
                             index=i*parallelism+p)
            for ot in output:
                output_len = len(tokenizer.encode(ot[0].decode())) - 1
                output_tokens += output_len
                tokens += input_len + output_len    # get rid of the start token.
            throughput[i] = tokens/latency[i]

        print('first_token_latency: ', calculate_mean(first_token_latency))
        print('avg_output_len: ', int(output_tokens/n))
        print('latency', calculate_mean(latency))
        print('throughput: ', calculate_mean(throughput))
        return

    with grpcclient.InferenceServerClient(addr, verbose=False) as client:
        print('warm up')
        warmup(model_name, client)
        print('done warm up')
        latency = [0]*n
        for i in tqdm(range(n)):
            start_time = time.time()
            response = client.infer(model_name, inputs, outputs=[grpcclient.InferRequestedOutput("output")])
            end_time = time.time()
            latency[i] = end_time-start_time
        outputs = response.as_numpy("output")
        generated_text = outputs[0][0].decode()
        # Print the output to compare with each framework
        #print(f"Generated text: {generated_text[:32]}..{generated_text[-32:]}")
        print(f"Generated text: {generated_text[:512]}")
        tokens = tokenizer.encode(outputs[0][0].decode())
        print('output_tokens:', len(tokens))
        print(f'latency: {calculate_mean(latency)}')

parser = argparse.ArgumentParser(description="Benchmark")

# Add arguments to the parser
parser.add_argument("--model_name", type=str, default='llama-2-70b-chat-hf-ft-streaming')
parser.add_argument("--tokenizer_path", type=str, default='/models/triton/llama-2-70b-hf-ft_tokenizer/1/')
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--max_output_len", type=int, default=32)
parser.add_argument("--input_len", type=int, default=1)
parser.add_argument("--n", type=int, default=50)
parser.add_argument("--parallelism", type=int, default=1)
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
                 streaming=args.streaming,
                 parallelism=args.parallelism,
                 n=args.n)

## python3 b.py --model_name llama-2-70b-hf-ft --input_len 1 --batch_size 1 --max_output_len 2048
## python3 b.py --model_name llama-2-70b-hf-ft --input_len 1024 --max_output_len 1024  --batch_size 32
