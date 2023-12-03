from transformers import AutoTokenizer
import csv



tokenizer = AutoTokenizer.from_pretrained('/models/llama-2-7b-chat-hf')
prompts = []
with open('/models/llm_benchmark/prompts/arxiv2.csv', newline='') as file:
    reader = csv.reader(file)
    l = []
    for row in reader:
        l.append(len(tokenizer.encode(row[0])))
    print(min(l))
