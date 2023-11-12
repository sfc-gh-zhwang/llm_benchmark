import csv

from datasets import load_dataset
#from utils import calculate_stats

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
    article = i['article'].split()
    article = ' '.join(article[:2000])
    data.append([article])
    if id >= 1024:
        break
with open('prompts/arxiv.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerows(data)

def calculate_stats(lt):
    _max = max(lt)
    _min = min(lt)
    mean = sum(lt)//len(lt)
    return f'min: {_min}, max: {_max}, mean: {mean}'

print(calculate_stats(l1))
print(calculate_stats(l2))
