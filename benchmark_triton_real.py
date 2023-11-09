import argparse
import multiprocessing
import time
from functools import partial

import numpy as np
import tritonclient.grpc as grpcclient
from tqdm import tqdm
from transformers import AutoTokenizer
from tritonclient.utils import *

from utils import calculate_mean, generate_inputs, get_prompts


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


def send_batch(client, model_name, n_requests, batch_size, max_output_len):
    prompts = get_prompts(n_requests)
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        # print(batch)
        inputs = [
            _input("text", np.array(batch,
                                    dtype=object).reshape(-1, 1)),
            _input("max_output_len", np.array([[max_output_len]]*len(batch), dtype=np.int32))
        ]
        resp = client.infer(model_name, inputs)
        print(resp)


def benchmark_triton_real(
    model_name,
    max_output_len,
    batch_size,
    n_requests,
    n,
    addr="localhost:8001",
        ):
    with grpcclient.InferenceServerClient(addr, verbose=False) as client:
        print('warm up')
        warmup(model_name, client)
        print('done warm up')
        latency = [0]*n
        for i in tqdm(range(n)):
            start_time = time.time()
            send_batch(client, model_name, n_requests, batch_size, max_output_len)
            end_time = time.time()
            latency[i] = end_time-start_time
            print(latency)
            print(f'latency: {calculate_mean(latency[:i+1])}')
        print(f'latency: {calculate_mean(latency)}')


parser = argparse.ArgumentParser(description="Benchmark")

# Add arguments to the parser
parser.add_argument("--model_name", type=str, default='llama-2-70b-chat-hf-tp-8')
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--requests", type=int, default=64)
parser.add_argument("--max_output_len", type=int, default=512)
parser.add_argument("--n", type=int, default=50)

# Parse the command-line arguments
args = parser.parse_args()

print('\n=============== Argument ===============')
for key in vars(args):
    print('{}: {}'.format(key, vars(args)[key]))
print('========================================')

benchmark_triton_real(
                model_name=args.model_name,
                max_output_len=args.max_output_len,
                batch_size=args.batch_size,
                n_requests=args.requests,
                n=args.n)

## python3 b.py --model_name llama-2-70b-hf-ft --input_len 1 --batch_size 1 --max_output_len 2048
## python3 b.py --model_name llama-2-70b-hf-ft --input_len 1024 --max_output_len 1024  --batch_size 32
