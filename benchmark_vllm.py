# Modify from https://github.com/vllm-project/vllm/blob/852ef5b4f5481ce526c804ea234d1de0df91f48d/benchmarks/benchmark_latency.py
import argparse
import time

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from utils import calculate_mean, generate_inputs


def warmup(llm):
    for i in range(10):
        sampling_params = SamplingParams(
            n=1,
            temperature=1.0,
            top_p=1.0,
            use_beam_search=False,
            ignore_eos=True,
            max_tokens=512,
        )
        # FIXME(woosuk): Do not use internal method.
        llm._add_request(
            prompt='hello world, this is to warm up',
            prompt_token_ids=None,
            sampling_params=sampling_params,
        )

    llm._run_engine(use_tqdm=True)



def benchmark_vllm(
    model_path,
    max_output_len,
    batch_size,
    input_len,
        ):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print('init llm')
    llm = LLM(model=model_path,
              tokenizer=model_path,
              tensor_parallel_size=8,
              max_num_seqs=args.batch_size,
              max_num_batched_tokens=max(batch_size * (input_len+max_output_len+128), 4096)
              )
    print('done init llm')

    print('warm up')
    warmup(llm)
    print('done warm up')
 
    sampling_params = SamplingParams(
            n=1,
            temperature=1.0,
            top_k=50,
            top_p=1.0,
            use_beam_search=False,
            max_tokens=max_output_len,
        )
    start = time.time()
    # FIXME(woosuk): Do not use internal method.
    outputs = llm.generate(prompts=generate_inputs(tokenizer, input_len, batch_size),
                 sampling_params=sampling_params,
                 use_tqdm=False)
    end = time.time()

    elapsed_time = end - start
    total_num_tokens = batch_size * max_output_len
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        # Print the output to compare with each framework
        print(f"Prompt: {prompt[:16]}, Generated text: {generated_text[:16]}..{generated_text[-16:]}")

    print(f"Throughput: {total_num_tokens / elapsed_time} tokens/s")
    print(f"Total latency: {elapsed_time}")


parser = argparse.ArgumentParser(description="Benchmark")

# Add arguments to the parser

parser.add_argument("--model_path", type=str, default='/models/llama2-70B-hf')
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--max_output_len", type=int, default=32)
parser.add_argument("--input_len", type=int, default=1)
parser.add_argument("--streaming", action='store_true', default=False, help="Whether or not to stream")

args = parser.parse_args()

print('\n=============== Argument ===============')
for key in vars(args):
    print('{}: {}'.format(key, vars(args)[key]))
print('========================================')

benchmark_vllm(model_path=args.model_path,
               max_output_len=args.max_output_len,
               batch_size=args.batch_size,
               input_len=args.input_len)
