from transformers import AutoTokenizer
import csv
import numpy as np
from tqdm import tqdm


class PromptsGenerator:
    def __init__(self, tokenizer_path):
        self.tokenizer_path = tokenizer_path
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        self.texts = []
        with open('./arxiv.csv', newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                self.texts.append(row[0])
        self.prompt_index = 0

        prompt_template = '''[INST]<<SYS>>
Please summarize the text that is given. Return just the summary and no additional conversational dialog such as ""Sure, here is the summary of the text:"".
<</SYS>>  [/INST]'''
        self.prompt_template_length = len(tokenizer.encode(prompt_template))
        np.random.seed(37)

    def generate(self, average_token, variance, max_token, n, show_progress=False):
        if n <= 0:
            return []
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        prompts = []
        prompt_lengths = []
        for i in tqdm(range(n), disable=not show_progress, desc="Generating prompts"):
            prompt_length = min(int(np.random.normal(average_token, variance)), max_token)
            prompt_length = max(prompt_length-self.prompt_template_length, 16)  # avoid prompt too short.
            prompt = self.texts[self.prompt_index]
            self.prompt_index += 1
            if self.prompt_index >= len(self.texts):
                self.prompt_index = 0
            prompt_tokens = tokenizer.encode(prompt)[:prompt_length]
            prompt = tokenizer.decode(prompt_tokens, skip_special_tokens=True)
            prompt = f'''[INST]<<SYS>>
Please summarize the text that is given. Return just the summary and no additional conversational dialog such as ""Sure, here is the summary of the text:"".
<</SYS>> {prompt} [/INST]'''
            prompt_lengths.append(len(tokenizer.encode(prompt)))
            prompts.append(prompt)
        print(max(prompt_lengths), min(prompt_lengths), sum(prompt_lengths)//len(prompt_lengths))
        return prompts

    def reset(self):
        self.prompt_index = 0


if __name__ == "__main__":
    import torch
    model = '/Users/zhwang/models/llama-2-7b-chat-hf'
    tokenizer = AutoTokenizer.from_pretrained(model)
    pg = PromptsGenerator(model)
    prompts = pg.generate(2048, 2048*0.3, 4096-1024, 2, show_progress=True)
    # prompts = ['hello', 'hello world']
    # input_id = tokenizer(prompts,
    #                      padding=False)
    # print(input_id)
    l = []
    for i in prompts:
        l.append(len(tokenizer.encode(i)))

    print(len(l), max(l), min(l), sum(l)/len(l))
