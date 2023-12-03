from transformers import AutoTokenizer
import csv
import numpy as np


class PromptsGenerator:
    def __init__(self, tokenizer_path):
        self.tokenizer = AutoTokenizer.from_pretrained('/models/llama-2-7b-chat-hf')

        self.texts = []
        with open('/models/llm_benchmark/prompts/arxiv2.csv', newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                self.texts.append(row[0])
        self.prompt_index = 0

        prompt_template = '''<s>[INST]<<SYS>>
Please summarize the text that is given. Return just the summary and no additional conversational dialog such as ""Sure, here is the summary of the text:"".
<</SYS>> [/INST]'''
        self.prompt_template_length = len(self.tokenizer.encode(prompt_template))
        np.random.seed(37)

    def generate(self, average_token, variance, max_token, n):
        if n <= 0:
            return []

        prompts = []
        for i in range(n):
            prompt_length = min(int(np.random.normal(average_token, variance)), max_token)
            prompt_length -= self.prompt_template_length
            prompt = self.texts[self.prompt_index]
            self.prompt_index += 1
            if self.prompt_index >= len(self.texts):
                self.prompt_index = 0
            prompt_tokens = self.tokenizer.encode(prompt)[:prompt_length]
            prompt = self.tokenizer.decode(prompt_tokens)

            prompts.append(
                f'''<s>[INST]<<SYS>>
Please summarize the text that is given. Return just the summary and no additional conversational dialog such as ""Sure, here is the summary of the text:"".
<</SYS>> {prompt}[/INST]'''
            )
        return prompts

tokenizer = AutoTokenizer.from_pretrained('/models/llama-2-7b-chat-hf')
pg = PromptsGenerator(None)
prompts = pg.generate(512, 512*0.3, 4096-1024, 128)

l = []
for i in prompts:
    l.append(len(tokenizer.encode(i)))

print(max(l), min(l), sum(l)/len(l))
