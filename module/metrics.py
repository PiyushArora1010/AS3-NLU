import evaluate

metricDic = {
    'bleu': evaluate.load('bleu'),
    'meteor': evaluate.load('meteor'),
    'cider': evaluate.load('cider')
}
