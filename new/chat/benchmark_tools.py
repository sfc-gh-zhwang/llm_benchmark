from functools import total_ordering
from typing import List
import time

class Query:
    def __init__(self, prompt):
        self.prompt = prompt[0]
        self.input_tokens = prompt[1]
        self.start_time = time.time()


def avg_int(lt):
    return sum(lt) // len(lt)

def avg_float(lt):
    return sum(lt) / len(lt)

@total_ordering
class Benchmark:
    def __init__(self, framework, input_length, output_length, time_to_first_token, latency, tensor_parallel):

        self.avg_input = avg_int(input_length)

        self.framework = framework

        self.max_input = max(input_length)
        self.min_input = min(input_length)

        self.avg_output = avg_int(output_length)
        self.max_output = max(output_length)
        self.min_output = min(output_length)

        self.tensor_parallel = tensor_parallel
        self.throughput = (sum(input_length)+sum(output_length))/latency
        self.latency = latency
        self.time_to_first_token = time_to_first_token

    def __str__(self):
        return f'{self.framework}' \
            f', {self.max_input}' \
            f', {self.max_output}' \
            f', {self.time_to_first_token: .3f}' \
            f', {self.latency: .2f}' \
            f', {self.throughput: .2f}' \
            f', {self.tensor_parallel}'

    def __lt__(self, other):
        if self.avg_input != other.avg_input:
            return self.avg_input < other.avg_input
        if self.tensor_parallel != other.tensor_parallel:
            return self.tensor_parallel < other.tensor_parallel
        if self.framework != other.framework:
            return self.framework < other.framework


def summarize_chat_benchmarks(
    framework: str,
    token_input: int,
    queries_per_second: int,
    clients: int,
    benchmarks: List[Benchmark],
) -> str:
    min_token_input = min([benchmark.max_input for benchmark in benchmarks])
    avg_token_input = avg_int([benchmark.max_input for benchmark in benchmarks])
    max_token_input = max([benchmark.max_input for benchmark in benchmarks])

    min_token_output = min([benchmark.max_output for benchmark in benchmarks])
    avg_token_output = avg_int([benchmark.max_output for benchmark in benchmarks])
    max_token_output = max([benchmark.max_output for benchmark in benchmarks])

    min_time_to_first_token = min([benchmark.time_to_first_token for benchmark in benchmarks])
    avg_time_to_first_token = avg_float([benchmark.time_to_first_token for benchmark in benchmarks])
    max_time_to_first_token = max([benchmark.time_to_first_token for benchmark in benchmarks])

    latencies = sorted([benchmark.latency for benchmark in benchmarks])
    min_latency = min(latencies)
    avg_latency = avg_float(latencies)
    max_latency = max(latencies)

    p50_latency = latencies[int(0.5 * len(latencies))]
    p90_latency = latencies[int(0.9 * len(latencies))]
    p95_latency = latencies[int(0.95 * len(latencies))]
    p99_latency = latencies[int(0.99 * len(latencies))]

    # print('!!!---Printing results---!!!')
    # # Output results as a csv
    # print('framework, input, output, time_to_first_token, latency(s), throughput, tensor_parallel')
    # for i in benchmarks:
    #     print(i)
    
    print(f"token_input: {token_input}")
    print(f"queries_per_second: {queries_per_second}")
    print(f"clients: {clients}")

    print(f"min_token_input: {min_token_input}")
    print(f"avg_token_input: {avg_token_input}")
    print(f"max_token_input: {max_token_input}")

    print(f"min_token_output: {min_token_output}")
    print(f"avg_token_output: {avg_token_output}")
    print(f"max_token_output: {max_token_output}")

    print(f"min_time_to_first_token: {min_time_to_first_token}")
    print(f"avg_time_to_first_token: {avg_time_to_first_token}")
    print(f"max_time_to_first_token: {max_time_to_first_token}")

    print(f"min_latency: {min_latency}")
    print(f"avg_latency: {avg_latency}")
    print(f"max_latency: {max_latency}")
    print(f"p50_latency: {p50_latency}")
    print(f"p90_latency: {p90_latency}")
    print(f"p95_latency: {p95_latency}")
    print(f"p99_latency: {p99_latency}")

    summarization_results = f"{framework}" \
        f", {queries_per_second: .2f}" \
        f", {avg_token_input}" \
        f", {avg_token_output}" \
        f", {avg_time_to_first_token: .3f}" \
        f", {avg_latency: .2f}" \
        f", {p50_latency: .2f}" \
        f", {p90_latency: .2f}" \
        f", {p95_latency: .2f}" \
        f", {p99_latency: .2f}"
    print('framework, qps, avg_token_input, avg_token_output, avg_time_to_first_token, avg_latency, p50_latency, p90_latency, p95_latency, p99_latency')
    print(summarization_results)
    return summarization_results
