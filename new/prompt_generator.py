from transformers import AutoTokenizer
import csv
import numpy as np
from tqdm import tqdm


class PromptsGenerator:
    def __init__(self, tokenizer_path):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        self.texts = []
        with open('./arxiv2.csv', newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                self.texts.append(row[0])
        self.prompt_index = 0

        prompt_template = '''<s>[INST]<<SYS>>
Please summarize the text that is given. Return just the summary and no additional conversational dialog such as ""Sure, here is the summary of the text:"".
<</SYS>>  [/INST]'''
        self.prompt_template_length = len(self.tokenizer.encode(prompt_template))
        np.random.seed(37)

    def generate(self, average_token, variance, max_token, n):
        if n <= 0:
            return []
        prompts = []
        prompt_lengths = np.clip(np.random.normal(average_token, variance, n).astype(int), a_min=16, a_max=max_token)
        for i in tqdm(range(n)):
            prompt_length = prompt_lengths[i]
            prompt_length -= self.prompt_template_length
            prompt = self.texts[self.prompt_index]
            self.prompt_index += 1
            if self.prompt_index >= len(self.texts):
                self.prompt_index = 0
            prompt_tokens = self.tokenizer.encode(prompt)[:prompt_length]
            prompt = self.tokenizer.decode(prompt_tokens, skip_special_tokens=True)

            prompts.append(
                f'''<s>[INST]<<SYS>>
Please summarize the text that is given. Return just the summary and no additional conversational dialog such as ""Sure, here is the summary of the text:"".
<</SYS>> {prompt} [/INST]'''
            )
        return prompts


tokenizer = AutoTokenizer.from_pretrained('/models/llama-2-7b-chat-hf')
pg = PromptsGenerator('/models/llama-2-7b-chat-hf')
prompts = pg.generate(1024, 1024*0.3, 4096-1024, 1024)

l = []
for i in prompts:
    #print(i)
    l.append(len(tokenizer.encode(i)))

print(len(l), max(l), min(l), sum(l)/len(l))
