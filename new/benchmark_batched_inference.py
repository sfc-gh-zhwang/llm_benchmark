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
                        type=list_of_ints,
                        help="Tensor parallelism",
                        default='1')
    parser.add_argument('--model', type=list_of_strings, required=True, help="path to the model")

    args = parser.parse_args()
    return args


def benchmark_mii(model, tensor_parallel, num_queries, warmup, prompt_length, max_new_tokens):
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

    prompt_generator.reset()
    prompts = prompt_generator.generate(average_token=prompt_length,
                                        variance=prompt_length*0.3,
                                        max_token=LLAMA2_MAX_SEQUENCE_LENGTH-max_new_tokens,
                                        n=num_queries,
                                        show_progress=True)
    start = time.time()
    outputs = llm.generate(prompts,
                           do_sample=False,
                           top_p=1.0,
                           max_new_tokens=max_new_tokens)
    latency = time.time() - start
    llm.terminate_server()

    input_lengths = []
    output_lengths = []

    for output in outputs:
        input_lengths.append(output.prompt_length)
        output_lengths.append(output.generated_length)

    return latency, input_lengths, output_lengths


def benchmark_vllm(model, tensor_parallel, num_queries, warmup, prompt_length, max_new_tokens):
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

    prompt_generator.reset()
    prompts = prompt_generator.generate(average_token=prompt_length,
                                        variance=prompt_length*0.3,
                                        max_token=LLAMA2_MAX_SEQUENCE_LENGTH-max_new_tokens,
                                        n=num_queries,
                                        show_progress=True)
    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    latency = time.time() - start

    input_lengths = []
    output_lengths = []

    for output in outputs:
        input_lengths.append(len(output.prompt_token_ids))
        output_lengths.append(len(output.outputs[0].token_ids))

    return latency, input_lengths, output_lengths


if __name__ == "__main__":
    args = parse_args()
    print('\n=============== Argument ===============')
    for key in vars(args):
        print('{}: {}'.format(key, vars(args)[key]))
    print('========================================')

    result = []
    for model, tensor_parallel, num_queries, prompt_length in enumerate(args.tensor_parallel, args.num_queries, args.prompt_length):
        if args.framework == 'vllm':
            latency, input_lengths, output_lengths = benchmark_vllm(
                model=args.model,
                tensor_parallel=tensor_parallel,
                num_queries=num_queries,
                warmup=args.warmup,
                prompt_length=prompt_length,
                max_new_tokens=args.max_new_tokens)
        elif args.framework == 'mii':
            latency, input_lengths, output_lengths = benchmark_mii(
                model=args.model,
                tensor_parallel=tensor_parallel,
                num_queries=num_queries,
                warmup=args.warmup,
                prompt_length=args.prompt_length,
                max_new_tokens=args.max_new_tokens)

        def _avg(lt):
            return sum(lt) // len(lt)

        result.append(
            f'{args.framework}, {args.num_queries}, '
            f'{_avg(input_lengths)}, {max(input_lengths)}, {min(input_lengths)}, '
            f'{_avg(output_lengths)}, {max(output_lengths)}, {min(output_lengths)}, '
            "{:.2f}".format(latency) + ', ' +
            "{:.2f}".format((sum(input_lengths)+sum(output_lengths))/latency))

    print('framework, num_prompts, avg_input, max_input, min_input, avg_output, max_output, min_output, latency(s), throughput, tensor_parallel')
    for i in result:
        print(i)


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
