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

long_sentence = "A path from a point approximately 330 metres east of the most south westerly corner of 17 Batherton Close, Widnes and approximately 208 metres east-south-east of the most southerly corner of Unit 3 Foundry Industrial Estate, Victoria Street, Widnes, proceeding in a generally east-north-easterly direction for approximately 28 metres to a point approximately 202 metres east-south-east of the most south-easterly corner of Unit 4 Foundry Industrial Estate, Victoria Street, and approximately 347 metres east of the most south-easterly corner of 17 Batherton Close, then proceeding in a generally northerly direction for approximately 21 metres to a point approximately 210 metres east of the most south-easterly corner of Unit 5 Foundry Industrial Estate, Victoria Street, and approximately 202 metres east-south-east of the most north-easterly corner of Unit 4 Foundry Industrial Estate, Victoria Street, then proceeding in a generally east-north-east direction for approximately 64 metres to a point approximately 282 metres east-south-east of the most easterly corner of Unit 2 Foundry Industrial Estate, Victoria Street, Widnes and approximately 259 metres east of the most southerly corner of Unit 4 Foundry Industrial Estate, Victoria Street, then proceeding in a generally east-north-east direction for approximately 350 metres to a point approximately 3 metres west-north-west of the most north westerly corner of the boundary fence of the scrap metal yard on the south side of Cornubia Road, Widnes, and approximately 47 metres west-south-west of the stub end of Cornubia Road be diverted to a 3 metre wide path from a point approximately " \
    "183 metres east-south-east of the most easterly corner of Unit 5 Foundry Industrial Estate, Victoria Street and approximately 272 metres east of the most north-easterly corner of 26 Ann Street West, Widnes, then proceeding in a generally north easterly direction for approximately 58 metres to a point approximately 216 metres east-south-east of the most easterly corner of Unit 4 Foundry Industrial Estate, Victoria Street and approximately 221 metres east of the most southerly corner of Unit 5 Foundry Industrial Estate, Victoria Street, then proceeding in a generally easterly direction for approximately 45 metres to a point approximately 265 metres east-south-east of the most north-easterly corner of Unit 3 Foundry Industrial Estate, Victoria Street and approximately 265 metres east of the most southerly corner of Unit 5 Foundry Industrial Estate, Victoria Street, then proceeding in a generally east-south-east direction for approximately 102 metres to a point approximately 366 metres east-south-east of the most easterly corner of Unit 3 Foundry Industrial Estate, Victoria Street and approximately 463 metres east of the most north easterly corner of 22 Ann Street West, Widnes, then proceeding in a generally north-north-easterly direction for approximately 19 metres to a point approximately 368 metres east-south-east of the most easterly corner of Unit 3 Foundry Industrial Estate, Victoria Street and approximately 512 metres east of the most south easterly corner of 17 Batherton Close, Widnes then proceeding in a generally east-south, easterly direction for approximately 16 metres to a point approximately 420 metres east-south-east of the " \
    "most southerly corner of Unit 2 Foundry Industrial Estate, Victoria Street and approximately 533 metres east of the most south-easterly corner of 17 Batherton Close, then proceeding in a generally east-north-easterly direction for approximately 240 metres to a point approximately 606 metres east of the most northerly corner of Unit 4 Foundry Industrial Estate, Victoria Street and approximately 23 metres south of the most south westerly corner of the boundary fencing of the scrap metal yard on the south side of Cornubia Road, Widnes, then proceeding in a generally northern direction for approximately 44 metres to a point approximately 3 metres west-north-west of the most north westerly corner of the boundary fence of the scrap metal yard on the south side of Cornubia Road and approximately 47 metres west-south-west of the stub end of Cornubia Road."

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
    printed=decoded


def benchmark_triton(
    model_name,
    tokenizer_path,
    max_output_len,
    batch_size,
    input_len,
    streaming,
    addr = "localhost:8001"
    ):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if input_len == 1:
        text = ''
    else:
        tokens = tokenizer(long_sentence)['input_ids'][:input_len]
        text = tokenizer.decode(tokens)
        print('input token len: ', len(tokens))
    inputs = [
        _input("text", np.array([text] * batch_size, dtype=object).reshape(-1, 1)),
        _input("max_output_len", np.array([[max_output_len]]*batch_size, dtype=np.int32))
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
        outputs = [grpcclient.InferRequestedOutput("output")]
        start_time = time.time()
        response = client.infer(model_name, inputs, outputs=outputs)
        end_time = time.time()
        outputs = response.as_numpy("output")
        tokens = tokenizer.encode(outputs[0][0].decode())
        print('output_tokens:', len(tokens))
        print('total latency: ', end_time-start_time)

parser = argparse.ArgumentParser(description="Benchmark")

# Add arguments to the parser
parser.add_argument("--model_name", type=str, default='llama-2-70b-chat-hf-ft-streaming')
parser.add_argument("--tokenizer_path", type=str, default='/models/triton/llama-2-70b-chat-hf-ft-streaming_tokenizer/1/')
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--max_output_len", type=int, default=32)
parser.add_argument("--input_len", type=int, default=1)
parser.add_argument("--streaming", action='store_true', default=False, help="Whether or not to stream")

# Parse the command-line arguments
args = parser.parse_args()

benchmark_triton(model_name=args.model_name,
                 tokenizer_path=args.tokenizer_path,
                 max_output_len=args.max_output_len,
                 input_len=args.input_len,
                 batch_size=args.batch_size,
                 streaming=args.streaming)
