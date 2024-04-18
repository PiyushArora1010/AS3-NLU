import evaluate
import nltk

nltk.download("punkt", quiet=True)

metricDic = {
    "bleu": evaluate.load('bleu'),
    "rouge": evaluate.load('rouge'),
}