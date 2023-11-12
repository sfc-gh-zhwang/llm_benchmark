import csv

from datasets import load_dataset
from utils import calculate_stats

ds = load_dataset("ccdv/arxiv-summarization")
print(ds)

l1 = []
l2 = []
print(len(ds['test']))
id = 0
data = []
for i in ds['test']:
    l1.append(len(i['article']))
    l2.append(len(i['abstract']))
    id += 1
    print(i['abstract'])
    data.append([i['abstract']])
    if id >= 2048:
        break
with open('prompts/arxiv.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerows(data)


print(calculate_stats(l1))
print(calculate_stats(l2))
