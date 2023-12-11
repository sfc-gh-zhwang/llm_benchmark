import argparse
import asyncio
import gc
import json
import os
import queue
import random
import time
from typing import Iterable, List
import numpy as np
import requests
from transformers import AutoTokenizer
from benchmark_tools import Benchmark, Query, summarize_chat_benchmarks
import threading
import multiprocessing
import multiprocessing as mp
from common_arg_types import list_of_floats, list_of_ints
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException, np_to_triton_dtype
from functools import partial

from prompt_generator import PromptsGenerator

MAX_SEQUENCE_LENGTH = 4096

def _input(name: str, data: np.ndarray) -> grpcclient.InferInput:
    t = grpcclient.InferInput(name, data.shape, np_to_triton_dtype(data.dtype))
    t.set_data_from_numpy(data)
    return t

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark inference")
    parser.add_argument("-k",
                        "--max_new_tokens",
                        type=int,
                        default=1024)
    parser.add_argument("-w",
                        "--warmup",
                        type=int,
                        help="number of queries for warming up",
                        default=128)
    parser.add_argument("-l",
                        "--prompt_length",
                        help="average number of tokens each prompt.",
                        type=list_of_ints,
                        default="512,1024,1536,2048,2560")
    parser.add_argument("-tp",
                        "--tensor_parallel",
                        type=int,
                        help="Tensor parallelism",
                        default='1')
    parser.add_argument("-c",
                        "--client_num",
                        type=int,
                        help="Number of clients",
                        default=64)
    parser.add_argument("-t",
                        "--use_thread",
                        action="store_true",
                        help="use thread for clients, else multiprocessing",
                        default=False)
    parser.add_argument("-qps",
                        "--queries_per_second",
                        type=list_of_floats,
                        help="List of queries per second",
                        default="0.5,1.0,1.5,2.0")
    parser.add_argument('--model', type=str, required=True, help="path to the model")

    args, _ = parser.parse_known_args()
    return args


class CallbackObject:
    def __init__(self):
        self.start_time = time.time()
        self.responses = []
        self.first = True
        self.first_token_time = 0.0


def benchmark_vllm(
    model: str,
    prompts: List[str],
    max_new_tokens: int,
    query: Query,
) -> List[Benchmark]:

    def get_streaming_response(response: requests.Response, time_last_token) -> Iterable[List[str]]:
        for chunk in response.iter_lines(chunk_size=8192, decode_unicode=False,
                                        delimiter=b"\0"):
            if chunk:
                data = json.loads(chunk.decode("utf-8"))
                output = data["text"][0]
                time_now = time.time()
                yield output, time_now - time_last_token
                time_last_token = time_now

    def stream_callback(a, result, error):
        if error:
            print(error)
        a.put((result.as_numpy('sequence_length').reshape(-1,)[0], time.time(), result.as_numpy('output_ids').reshape(-1,)[0]))

    tokenizer = AutoTokenizer.from_pretrained(model)
    input_id = tokenizer(prompts, padding=False).input_ids
    input_lengths = []
    for i in input_id:
        input_lengths.append(len(i))
    # with multiprocessing.Manager() as manager:
    #     def send(client, input_id, input_length, i, shared_list):
    inputs = [
        _input("input_ids", np.array(input_id, dtype=np.int32).reshape(1, -1)),
        _input("input_lengths", np.array(input_lengths, dtype=np.int32).reshape(1, -1)),
        _input("request_output_len", np.array([max_new_tokens], dtype=np.uint32).reshape(1, -1)),
        _input("end_id", np.array([2], dtype=np.uint32).reshape(1, -1)),
        _input("streaming", np.array([True], dtype=np.bool_).reshape(1, -1)),
    ]
    # inputs = [
    #     _input("text", np.array(query.prompt, dtype=object).reshape(1, -1)),
    #     _input("max_output_len", np.array([max_new_tokens], dtype=np.uint32).reshape(1, -1)),
    #     _input("end_id", np.array([2], dtype=np.uint32).reshape(1, -1)),
    #     _input("streaming", np.array([True], dtype=np.bool_).reshape(1, -1)),
    # ]
    result_queue = mp.Queue()
    start = time.time()
    with grpcclient.InferenceServerClient("localhost:8001", verbose=False) as client:
        client.start_stream(callback=partial(stream_callback, result_queue))
        client.async_stream_infer('tensorrt_llm', inputs)
        # client.stop_stream()
    token_gen_time = []
    output_length = 0
    while True:
        try:
            a = result_queue.get(block=False)
        except Exception:
            break
        output_length += a[0]
        token_gen_time.append(a[1])
        id = a[2]
        if output_length >= max_new_tokens or id == 2:
            break
    time_to_first_token = token_gen_time[0] - query.start_time
    latency = time.time() - query.start_time
    input_length = [query.input_tokens]
    benchmarks = ([
        Benchmark(
            framework='vllm',
            input_length=input_length,
            output_length=[int(output_length)],
            time_to_first_token=time_to_first_token,
            latency=latency,
            tensor_parallel=8,
        )
    ])
    print(benchmarks)

    return benchmarks


def _run_vllm_parallel(
    model,
    barrier,
    query_queue,
    result_queue,
    max_new_tokens,
    client_num,
):
    pid = os.getpid()
    session_id = f"test_session_p{pid}_t{threading.get_ident()}"
    event_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(event_loop)

    barrier.wait()

    # Warmup
    try:
        while True:
            query = query_queue.get(timeout=1)
            print(f"warmup queue size: {query_queue.qsize()} ({pid})", flush=True)
            benchmark_vllm(model=model, prompts=[query.prompt], max_new_tokens=max_new_tokens, query=query)
    except queue.Empty:
        pass

    barrier.wait()

    time.sleep(random.uniform(0, client_num) * 0.01)
    while True:
        try:
            query = query_queue.get(timeout=30)
            if len(query.prompt) == 0:
                break
            benchmarks = benchmark_vllm(model=model, prompts=[query.prompt], max_new_tokens=max_new_tokens, query=query)
            [result_queue.put(benchmark) for benchmark in benchmarks]
        except queue.Empty:
            pass

    print(f"Worker ({pid}) finished. session_id: {session_id}")


def run_vllm_benchmarks(
    client_num: int,
    use_thread: bool,
    model: str,
    queries_per_second_list: List[float],
    prompt_length_list: List[int],
    max_new_tokens: int,
    warmup: int,
) -> List[Benchmark]:
    try:
        # Start threads/processes for # of clients
        if use_thread:
            runnable_cls = threading.Thread
            barrier_cls = threading.Barrier
            queue_cls = queue.Queue
        else:
            runnable_cls = multiprocessing.Process
            barrier_cls = multiprocessing.Barrier
            queue_cls = multiprocessing.Queue

        barrier = barrier_cls(client_num + 1)
        query_queue = queue_cls()
        result_queue = queue_cls()

        processes = []
        for _ in range(client_num):
            processes.append(
                runnable_cls(
                    target=_run_vllm_parallel,
                    args=(model, barrier, query_queue, result_queue, max_new_tokens, client_num)
                )
            )
        for p in processes:
            p.start()

        prompt_generator = PromptsGenerator(tokenizer_path=model)

        # Generate warmup prompts. This will generate n * len(prompt_lengths) warmup queries
        prompts = (
            prompt_generator.generate(
                average_token=2560,
                variance=2560*0.3,
                max_token=MAX_SEQUENCE_LENGTH-max_new_tokens,
                n=warmup,
                show_progress=True,
            )
        )
        [query_queue.put(Query(prompt)) for prompt in prompts]

        # Barrier to wait for all clients to initialized
        barrier.wait()
        # Barrier for all clients to finish warmup
        barrier.wait()

        time.sleep(5)

        summarization_results = []
        for prompt_length in prompt_length_list:
            for queries_per_second in queries_per_second_list:
                print(f"benchmarking {prompt_length} prompt length at {queries_per_second} qps")
                # Generate prompts to run benchmark on
                prompts = (
                    prompt_generator.generate(
                        average_token=prompt_length,
                        variance=prompt_length*0.3,
                        max_token=MAX_SEQUENCE_LENGTH-max_new_tokens,
                        n=100,
                        show_progress=True,
                    )
                )

                # For 5 minutes, send a query every 1/qps
                i = 0
                total_queries_sent = 0
                time_start = time.time()
                while time.time() - time_start < 300:
                    if i >= len(prompts):
                        i = 0
                    query_queue.put(Query(prompts[i]))
                    i += 1
                    total_queries_sent += 1
                    time.sleep(1/queries_per_second)

                benchmarks = []
                while len(benchmarks) < total_queries_sent:
                    res = result_queue.get(block=True)
                    benchmarks.append(res)

                summarization_results.append(summarize_chat_benchmarks(
                    framework="vllm",
                    token_input=prompt_length,
                    queries_per_second=queries_per_second,
                    clients=args.client_num,
                    benchmarks=sorted(benchmarks),
                ))

        for _ in range(client_num):
            query_queue.put(Query(("", 0)))

        for summarization_result in summarization_results:
            print(summarization_result)

    except Exception as e:
        print(f"error: {repr(e)}")


if __name__ ==  "__main__":
    args = parse_args()
    print('\n=============== Argument ===============')
    for key in vars(args):
        print('{}: {}'.format(key, vars(args)[key]))
    print('========================================')

    benchmarks = run_vllm_benchmarks(
        client_num=args.client_num,
        use_thread=args.use_thread,
        model=args.model,
        queries_per_second_list=args.queries_per_second,
        prompt_length_list=args.prompt_length,
        max_new_tokens=args.max_new_tokens,
        warmup=args.warmup,
    )
