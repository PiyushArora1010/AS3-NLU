import evaluate

metricDic = {
    'bleu': evaluate.load('bleu'),
    'cider': evaluate.load('cider')
}
