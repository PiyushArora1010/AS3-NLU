import datasets

metricDic = {
    "rouge": datasets.load_metric("rouge"),
    "bleu": datasets.load_metric("sacrebleu"),
    "meteor": datasets.load_metric("meteor"),
}