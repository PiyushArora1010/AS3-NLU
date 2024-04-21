from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import EncoderDecoderModel


dicModels = {
    'bert-base': 'bert-base-multilingual-cased',
    'bart-base': 'Someman/bart-hindi',
    't5-base': 'google/mt5-base',
}

def getModel(name):
    global dicModels

    name = dicModels[name]
    if 'bert' in name:
        bert2bert = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-multilingual-cased", "bert-base-multilingual-cased")

    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(name)
    return model

def getTokenizer(name):
    global dicModels
    name = dicModels[name]
    return AutoTokenizer.from_pretrained(name, keep_accents=True)