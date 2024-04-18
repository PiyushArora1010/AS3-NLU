import evaluate

metricDic = {
    'rouge': evaluate.load('rouge'),
    'bleu': evaluate.load('bleu'),
    'meteor': evaluate.load('meteor'),
    'cider': evaluate.load('cider')
}

metric = metricDic['bleu']
metric = metricDic['rouge']
metric = metricDic['meteor']
metric = metricDic['cider']

print(metric)