import evaluate
import nltk

nltk.download("punkt", quiet=True)

metricDic = {
    "bleu": evaluate.load('evaluate/metrics/bleu/bleu.py'),
    "rouge": evaluate.load('evaluate/metrics/rouge/rouge.py'),
} 