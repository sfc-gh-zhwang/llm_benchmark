from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
prompts = prompts * 1000
# Create a sampling params object.
sampling_params = SamplingParams(temperature=1, max_tokens=512)

# Create an LLM.
llm = LLM(model="/models/llama-2-7b-chat-hf/", max_num_batched_tokens=4096)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
