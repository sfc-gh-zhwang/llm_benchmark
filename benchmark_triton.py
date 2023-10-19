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


def stream_callback(result, error):
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


def benchmark_triton(
    model_name,
    tokenizer_path,
    max_output_len,
    batch_size,
    input_len,
    streaming,
    n,
    addr="localhost:8001",
        ):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    prompts = generate_inputs(tokenizer, input_len, batch_size)
    print(f"Prompt: {prompts[0][:32]}..{prompts[0][-32:]}")
    inputs = [
        _input("text", np.array(prompts, dtype=object).reshape(-1, 1)),
        _input("max_output_len", np.array([[max_output_len]]*batch_size, dtype=np.int32)),
        _input("random_seed",  np.array([[0]]*batch_size, dtype=np.uint64))
    ]
    if streaming:
        first_token_latency = [0]*n
        throughput = [0]*n
        latency = [0]*n
        for i in tqdm(range(n)):
            with grpcclient.InferenceServerClient(addr, verbose=False) as client:
                global first_token_time
                first_token_time = None
                start_time = time.time()
                client.start_stream(callback=partial(stream_callback,))
                client.async_stream_infer(model_name, inputs)

            global end_time
            first_token_latency[i] = first_token_time - start_time
            latency[i] = end_time - start_time
            tokens = 0
            for ot in output:
                tokens += input_len + len(tokenizer.encode(ot[0].decode())) - 1 # get rid of the start token.
            throughput[i] = tokens/latency[i]

        print('first_token_latency: ', calculate_mean(first_token_latency))
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
        print(f"Generated text: {generated_text[:32]}..{generated_text[-32:]}")
        tokens = tokenizer.encode(outputs[0][0].decode())
        print('output_tokens:', len(tokens))
        _, _, _, latency = calculate_mean(latency)
        print(f'latency: {latency}')

parser = argparse.ArgumentParser(description="Benchmark")

# Add arguments to the parser
parser.add_argument("--model_name", type=str, default='llama-2-70b-chat-hf-ft-streaming')
parser.add_argument("--tokenizer_path", type=str, default='/models/triton/llama-2-70b-hf-ft_tokenizer/1/')
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--max_output_len", type=int, default=32)
parser.add_argument("--input_len", type=int, default=1)
parser.add_argument("--n", type=int, default=50)
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
                 n=args.n)

## python3 b.py --model_name llama-2-70b-hf-ft --input_len 1 --batch_size 1 --max_output_len 2048
## python3 b.py --model_name llama-2-70b-hf-ft --input_len 1024 --max_output_len 1024  --batch_size 32
