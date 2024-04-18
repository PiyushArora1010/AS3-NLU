import evaluate

metricDic = {
    'rouge': evaluate.load('rouge'),
    'bleu': evaluate.load('bleu'),
    'meteor': evaluate.load('meteor'),
    'cider': evaluate.load('cider')
}
