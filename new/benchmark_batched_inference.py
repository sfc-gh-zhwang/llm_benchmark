from prompt_generator import PromptsGenerator
import argparse
import time
import torch
import gc
import json
import os
from functools import total_ordering


LLAMA2_MAX_SEQUENCE_LENGTH = 4096


def list_of_ints(arg):
    return list(map(int, arg.split(',')))


def list_of_strings(arg):
    return arg.split(',')


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark inference")
    parser.add_argument("-k",
                        "--max_new_tokens",
                        type=int,
                        default=1024)
    parser.add_argument("-n",
                        "--num_queries",
                        type=list_of_ints,
                        help="number of queries to run",
                        default='10')
    parser.add_argument("-w",
                        "--warmup",
                        type=int,
                        help="number of queries for warming up",
                        default=64)
    parser.add_argument("-l",
                        "--prompt_length",
                        help="average number of tokens each prompt.",
                        type=list_of_ints,
                        default='1024')
    parser.add_argument('--framework',
                        required=True,
                        type=list_of_strings,
                        default='vllm,mii')
    parser.add_argument("-tp",
                        "--tensor_parallel",
                        type=int,
                        help="Tensor parallelism",
                        default='1')
    parser.add_argument('--model', type=str, required=True, help="path to the model")

    args = parser.parse_args()
    return args


@total_ordering
class Benchmark:
    def __init__(self, framework, num_queries, input_length, output_length, latency, tensor_parallel):
        self.num_queries = num_queries

        def _avg(lt):
            return sum(lt) // len(lt)

        self.avg_input = _avg(input_length)

        self.framework = framework

        self.max_input = max(input_length)
        self.min_input = min(input_length)

        self.avg_output = _avg(output_length)
        self.max_output = max(output_length)
        self.min_output = min(output_length)

        self.tensor_parallel = tensor_parallel
        self.throughput = (sum(input_length)+sum(output_length))/latency
        self.latency = latency

    def __str__(self):
        return f'{self.framework}' \
            f', {self.num_queries}' \
            f', {self.avg_input}, {self.min_input}, {self.max_input}' \
            f', {self.avg_output}, {self.min_output}, {self.max_output}' \
            f', {self.latency: .2f}' \
            f', {self.throughput: .2f}' \
            f', {self.tensor_parallel}'

    def __lt__(self, other):
        if self.num_queries != other.num_queries:
            return self.num_queries < other.num_queries
        if self.avg_input != other.avg_input:
            return self.avg_input < other.avg_input
        if self.tensor_parallel != other.tensor_parallel:
            return self.tensor_parallel < other.tensor_parallel
        if self.framework != other.framework:
            return self.framework < other.framework


def benchmark_mii(model, tensor_parallel, num_queries, warmup, prompt_lengths, max_new_tokens):
    import mii
    from deepspeed.inference import RaggedInferenceEngineConfig, DeepSpeedTPConfig
    from deepspeed.inference.v2.ragged import DSStateManagerConfig

    tp_config = DeepSpeedTPConfig(tp_size=tensor_parallel)
    mgr_config = DSStateManagerConfig(max_ragged_batch_size=128*tensor_parallel,
                                      max_ragged_sequence_count=128*tensor_parallel)
    inference_config = RaggedInferenceEngineConfig(tensor_parallel=tp_config,
                                                   state_manager=mgr_config)
    start = time.time()
    llm = mii.serve(
        model,
        deployment_name='mii',
        tensor_parallel=tensor_parallel,
        inference_engine_config=inference_config,
        replica_num=1,
        task='text-generation'
    )
    print('took ' + "{:.2f}".format(time.time()-start) + " seconds to start llm engine")

    prompt_generator = PromptsGenerator(tokenizer_path=model)
    if warmup > 0:
        print('warmming up...')
        warmup_prompts = prompt_generator.generate(1024, 1024*0.3, 2048, warmup)
        llm.generate(warmup_prompts, max_new_tokens=max_new_tokens)
        print('warm up finished')

    benchmarks = []
    for prompt_length in prompt_lengths:
        for num_query in num_queries:
            prompt_generator.reset()
            prompts = prompt_generator.generate(average_token=prompt_length,
                                                variance=prompt_length*0.3,
                                                max_token=LLAMA2_MAX_SEQUENCE_LENGTH-max_new_tokens,
                                                n=num_query,
                                                show_progress=True)
            print('generating...')
            start = time.time()
            outputs = llm.generate(prompts,
                                   do_sample=False,
                                   top_p=1.0,
                                   max_new_tokens=max_new_tokens)
            latency = time.time() - start
            print('generating finished')

            input_lengths = []
            output_lengths = []

            for output in outputs:
                input_lengths.append(output.prompt_length)
                output_lengths.append(output.generated_length)
            benchmarks.append(Benchmark(framework='mii',
                                        num_queries=num_query,
                                        input_length=input_lengths,
                                        output_length=output_lengths,
                                        latency=latency,
                                        tensor_parallel=tensor_parallel))
            for i in benchmarks:
                print(i)

    llm.terminate_server()
    return benchmarks


def benchmark_vllm(model, tensor_parallel, num_queries, warmup, prompt_lengths, max_new_tokens):
    from vllm import LLM, SamplingParams
    from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel


    # Create an LLM.
    start = time.time()
    llm = LLM(model=model, tensor_parallel_size=tensor_parallel)
    print('took ' + "{:.2f}".format(time.time()-start) + " seconds to start llm engine")

    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0,  # get rid of nondeterminism.
                                     top_p=1.0,
                                     top_k=-1,
                                     max_tokens=max_new_tokens)

    prompt_generator = PromptsGenerator(tokenizer_path=model)
    if warmup > 0:
        print('warmming up...')
        warmup_prompts = prompt_generator.generate(1024, 1024*0.3, 2048, warmup)
        llm.generate(warmup_prompts, sampling_params)
        print('warm up finished')

    benchmarks = []
    for prompt_length in prompt_lengths:
        for num_query in num_queries:
            prompt_generator.reset()
            prompts = prompt_generator.generate(average_token=prompt_length,
                                                variance=prompt_length*0.3,
                                                max_token=LLAMA2_MAX_SEQUENCE_LENGTH-max_new_tokens,
                                                n=num_query,
                                                show_progress=True)
            start = time.time()
            outputs = llm.generate(prompts, sampling_params)
            latency = time.time() - start

            input_lengths = []
            output_lengths = []

            for output in outputs:
                input_lengths.append(len(output.prompt_token_ids))
                output_lengths.append(len(output.outputs[0].token_ids))

            benchmarks.append(Benchmark(framework='vllm',
                                        num_queries=num_query,
                                        input_length=input_lengths,
                                        output_length=output_lengths,
                                        latency=latency,
                                        tensor_parallel=tensor_parallel))
            for i in benchmarks:
                print(i)

    # Destroy
    # destroy_model_parallel()
    # del llm
    # gc.collect()
    # torch.cuda.empty_cache()
    # torch.distributed.destroy_process_group()
    return benchmarks


def benchmark_trtllm(model, tensor_parallel, num_queries, warmup, prompt_lengths, max_new_tokens):
    import tensorrt_llm
    import tensorrt_llm.profiler as profiler
    from tensorrt_llm.logger import logger
    from tensorrt_llm.quantization import QuantMode

    def TRTLLaMA(engine_dir):
        config_path = os.path.join(engine_dir, 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        dtype = config['builder_config']['precision']
        tp_size = config['builder_config']['tensor_parallel']
        pp_size = config['builder_config']['pipeline_parallel']
        world_size = tp_size * pp_size

        assert world_size == tensorrt_llm.mpi_world_size(), \
            f'Engine world size ({world_size}) != Runtime world size ({tensorrt_llm.mpi_world_size()})'

        num_heads = config['builder_config']['num_heads'] // tp_size
        hidden_size = config['builder_config']['hidden_size'] // tp_size
        vocab_size = config['builder_config']['vocab_size']
        num_layers = config['builder_config']['num_layers']
        use_gpt_attention_plugin = bool(
            config['plugin_config']['gpt_attention_plugin'])
        remove_input_padding = config['plugin_config']['remove_input_padding']
        num_kv_heads = config['builder_config'].get('num_kv_heads', num_heads)
        paged_kv_cache = config['plugin_config']['paged_kv_cache']
        tokens_per_block = config['plugin_config']['tokens_per_block']
        use_custom_all_reduce = config['plugin_config'].get('use_custom_all_reduce',
                                                            False)

        quant_mode = QuantMode(config['builder_config']['quant_mode'])
        if config['builder_config'].get('multi_query_mode', False):
            tensorrt_llm.logger.warning(
                "`multi_query_mode` config is deprecated. Please rebuild the engine."
            )
            num_kv_heads = 1
        num_kv_heads = (num_kv_heads + tp_size - 1) // tp_size

        model_config = tensorrt_llm.runtime.ModelConfig(
            vocab_size=vocab_size,
            num_layers=num_layers,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            hidden_size=hidden_size,
            paged_kv_cache=paged_kv_cache,
            tokens_per_block=tokens_per_block,
            gpt_attention_plugin=use_gpt_attention_plugin,
            remove_input_padding=remove_input_padding,
            use_custom_all_reduce=use_custom_all_reduce,
            dtype=dtype,
            quant_mode=quant_mode)

        runtime_rank = tensorrt_llm.mpi_rank()
        runtime_mapping = tensorrt_llm.Mapping(world_size,
                                            runtime_rank,
                                            tp_size=tp_size,
                                            pp_size=pp_size)
        torch.cuda.set_device(runtime_rank % runtime_mapping.gpus_per_node)

        engine_name = get_engine_name('llama', dtype, tp_size, pp_size,
                                    runtime_rank)
        serialize_path = os.path.join(engine_dir, engine_name)

        tensorrt_llm.logger.set_level('info')

        profiler.start('load tensorrt_llm engine')
        with open(serialize_path, 'rb') as f:
            engine_buffer = f.read()
        decoder = tensorrt_llm.runtime.GenerationSession(model_config,
                                                        engine_buffer,
                                                        runtime_mapping)
        profiler.stop('load tensorrt_llm engine')
        tensorrt_llm.logger.info(
            f'Load engine takes: {profiler.elapsed_time_in_sec("load tensorrt_llm engine")} sec'
        )
        return decoder

    tensorrt_llm_llama = TRTLLaMA('/models/trt_engines/llama-2-7b-chat-hf/1-gpu/')


if __name__ == "__main__":
    args = parse_args()
    print('\n=============== Argument ===============')
    for key in vars(args):
        print('{}: {}'.format(key, vars(args)[key]))
    print('========================================')

    benchmarks = []

    if 'mii' in args.framework:
        benchmarks += benchmark_mii(
                model=args.model,
                tensor_parallel=args.tensor_parallel,
                num_queries=args.num_queries,
                warmup=args.warmup,
                prompt_lengths=args.prompt_length,
                max_new_tokens=args.max_new_tokens)

    if 'vllm' in args.framework:
        benchmarks += benchmark_vllm(
                model=args.model,
                tensor_parallel=args.tensor_parallel,
                num_queries=args.num_queries,
                warmup=args.warmup,
                prompt_lengths=args.prompt_length,
                max_new_tokens=args.max_new_tokens)

    if 'trtllm' in args.framework:
        benchmarks += benchmark_trtllm(
                model=args.model,
                tensor_parallel=args.tensor_parallel,
                num_queries=args.num_queries,
                warmup=args.warmup,
                prompt_lengths=args.prompt_length,
                max_new_tokens=args.max_new_tokens)

    benchmarks = sorted(benchmarks)

    print('framework, num_prompts, avg_input, max_input, min_input, avg_output, max_output, min_output, latency(s), throughput, tensor_parallel')
    for i in benchmarks:
        print(i)
