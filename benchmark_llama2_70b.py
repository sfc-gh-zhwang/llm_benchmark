import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import *
import multiprocessing as mp
from transformers import AutoTokenizer
import time
from functools import partial
import argparse

def _input(name: str, data: np.ndarray) -> grpcclient.InferInput:
    t = grpcclient.InferInput(name, data.shape, np_to_triton_dtype(data.dtype))
    t.set_data_from_numpy(data)
    return t


start_time = None
first_token_time = None
end_time = None
output = None


def stream_callback(a, result, error):
    # print('stream_callback')
    global first_token_time
    global end_time
    global output
    if error:
        raise error
    end_time = time.time()
    if first_token_time is None:
        first_token_time = end_time
    output = result.as_numpy('output')


def benchmark_triton(
    model_name,
    tokenizer_path,
    max_output_len,
    batch_size,
    verbose,
    addr = "localhost:8001"
    ):
    with grpcclient.InferenceServerClient(addr, verbose=verbose) as client:
        result_queue = mp.Queue()
        inputs = [
            _input("text", np.array([''] * batch_size, dtype=object).reshape(-1, 1)),
            _input("max_output_len", np.array([[max_output_len]]*batch_size, dtype=np.int32))
        ]
        start_time = time.time()
        client.start_stream(callback=partial(stream_callback, result_queue))
        client.async_stream_infer(model_name, inputs)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    first_token_latency = first_token_time - start_time
    total_duration = end_time - start_time
    streaming_duration = end_time - first_token_time
    tokens = 0
    for i in output:
        tokens += len(tokenizer.encode(i[0].decode()))

    print('first_token_latency: ', first_token_latency)
    print('total duration', total_duration)
    print('total tokens generated: ', tokens, 'throughput', tokens/streaming_duration)

parser = argparse.ArgumentParser(description="Benchmark")

# Add arguments to the parser
parser.add_argument("--model_name", type=str, default='llama-2-70b-chat-hf-ft-streaming')
parser.add_argument("--tokenizer_path", type=str, default='/models/triton/llama-2-70b-chat-hf-ft-streaming_tokenizer/1/')
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--max_output_len", type=int, default=32)
parser.add_argument("--verbose", type=bool, action="store_true", default=False)

# Parse the command-line arguments
args = parser.parse_args()

benchmark_triton(model_name=args.model_name,
                 tokenizer_path=args.tokenizer_path,
                 max_output_len=args.max_output_len,
                 batch_size=args.batch_size)
