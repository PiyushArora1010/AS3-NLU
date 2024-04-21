import torch
from tqdm import tqdm
from module.data import IndicHeadlineGenerationData, IndicTranslationData
from module.metrics import metricDic
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = 'surajp/gpt2-hindi'

tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')
model = AutoModelForCausalLM.from_pretrained(model_id).to('cuda')
data = input("Enter the dataname: ")
if data == 'IndicHeadline':
    _, _, testdata = IndicHeadlineGenerationData(tokenizer,100)
elif data == 'IndicTranslation':
    _, _, testdata = IndicTranslationData(tokenizer,100)

model = model.eval()
del metricDic['rouge']
resultMetric = {key: 0 for key in metricDic.keys()}
count = 0
with torch.no_grad():
    for item in tqdm(testdata):
        item = {key: val.unsqueeze(0).to('cuda') for key, val in item.items()}
        generated = model.generate(**item, max_length=256, num_return_sequences=1, temperature=0.7)
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        references = tokenizer.batch_decode(item['labels'], skip_special_tokens=True)
        count += 1
        for key, metric in metricDic.items():
            resultMetric[key] = (metric.compute(predictions=decoded, references=references)[key] - resultMetric[key]) / count + resultMetric[key] * (count - 1) / count

print(resultMetric)