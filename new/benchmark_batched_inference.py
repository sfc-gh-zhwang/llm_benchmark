from prompt_generator import PromptsGenerator
import argparse
import time

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
    parser.add_argument('--framework', required=True, choices=['vllm', 'mii', 'trtllm'])
    parser.add_argument("-tp",
                        "--tensor_parallel",
                        type=int,
                        help="Tensor parallelism",
                        default='1')
    parser.add_argument('--model', type=str, required=True, help="path to the model")

    args = parser.parse_args()
    return args


class Benchmark:
    def __init__(self, framework, num_queries, input_length, output_length, latency, tensor_parallel):
        self.framework = framework
        self.num_queries = num_queries
        self.input_length = input_length
        self.output_length = output_length
        self.latency = latency
        self.tensor_parallel = tensor_parallel

    def __str__(self):
        def _avg(lt):
            return sum(lt) // len(lt)

        return f'{self.framework}' \
            f', {self.num_queries}' \
            f', {_avg(self.input_length)}, {min(self.input_length)}, {max(self.input_length)}' \
            f', {_avg(self.output_length)}, {min(self.output_length)}, {max(self.output_length)}' \
            f', {self.latency: .2f}' \
            f', {(sum(self.input_length)+sum(self.output_length))/self.latency: .2f}' \
            f', {self.tensor_parallel}'


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

    llm.terminate_server()
    return benchmarks


def benchmark_vllm(model, tensor_parallel, num_queries, warmup, prompt_lengths, max_new_tokens):
    from vllm import LLM, SamplingParams

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

    return benchmarks


if __name__ == "__main__":
    args = parse_args()
    print('\n=============== Argument ===============')
    for key in vars(args):
        print('{}: {}'.format(key, vars(args)[key]))
    print('========================================')

    result = []
    if args.framework == 'vllm':
        benchmarks = benchmark_vllm(
            model=args.model,
            tensor_parallel=args.tensor_parallel,
            num_queries=args.num_queries,
            warmup=args.warmup,
            prompt_lengths=args.prompt_length,
            max_new_tokens=args.max_new_tokens)
    elif args.framework == 'mii':
        benchmarks = benchmark_mii(
            model=args.model,
            tensor_parallel=args.tensor_parallel,
            num_queries=args.num_queries,
            warmup=args.warmup,
            prompt_lengths=args.prompt_length,
            max_new_tokens=args.max_new_tokens)



            # result.append(
            #     f'{args.framework}, {num_queries}, '
            #     f'{_avg(input_lengths)}, {max(input_lengths)}, {min(input_lengths)}, '
            #     f'{_avg(output_lengths)}, {max(output_lengths)}, {min(output_lengths)}, '
            #     "{:.2f}".format(latency) + ', ' +
            #     "{:.2f}".format((sum(input_lengths)+sum(output_lengths))/latency) +
            #     f', {args.tensor_parallel}')

    print('framework, num_prompts, avg_input, max_input, min_input, avg_output, max_output, min_output, latency(s), throughput, tensor_parallel')
    for i in benchmarks:
        print(i)
    time.sleep(10000)


# # Sample prompts.
# model_path = "/models/llama-2-7b-chat-hf/"
# prompt_generator = PromptsGenerator(tokenizer_path=model_path)

# prompts = prompt_generator.generate(1024, 1024*0.3, 4096-1024, 32, show_progress=True)


# # Generate texts from the prompts. The output is a list of RequestOutput objects
# # that contain the prompt, generated text, and other information.
# outputs = llm.generate(prompts, sampling_params)
# # Print the outputs.
# for output in outputs:
#     prompt = output.prompt
#     generated_text = output.outputs[0].text
#     print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
