import datasets

metricDic = {
    "rouge": datasets.load_metric("rouge"),
    "bleu": datasets.load_metric("sacrebleu"),
    "meteor": datasets.load_metric("meteor"),
    "sari": datasets.load_metric("sari"),
    "bertscore": datasets.load_metric("bertscore"),
    "bleurt": datasets.load_metric("bleurt"),
    "indic": datasets.load_metric("indic")
}