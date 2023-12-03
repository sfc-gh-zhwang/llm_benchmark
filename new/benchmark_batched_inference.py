from vllm import LLM, SamplingParams
from prompt_generator import PromptsGenerator

# Sample prompts.
model_path = "/models/llama-2-7b-chat-hf/"
prompt_generator = PromptsGenerator(tokenizer_path=model_path)

prompts = prompt_generator.generate(1024, 1024*0.3, 4096-1024, 32, show_progress=True)

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0, max_tokens=1024)

# Create an LLM.
llm = LLM(model=model_path)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
