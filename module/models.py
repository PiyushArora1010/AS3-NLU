from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import EncoderDecoderModel, BertGenerationEncoder, BertGenerationDecoder
from transformers import BertTokenizer


dicModels = {
    'bert-base': 'bert-base-multilingual-cased',
    'bart-base': 'Someman/bart-hindi',
    't5-base': 'google/mt5-base',
}

def getModel(name):
    global dicModels
    if 'bert' in name:
        name = dicModels[name]
        encoder = BertGenerationEncoder.from_pretrained(name)
        decoder = BertGenerationDecoder.from_pretrained(name)
        model = EncoderDecoderModel(encoder=encoder, decoder=decoder)
    else:
        name = dicModels[name]
        model = AutoModelForSeq2SeqLM.from_pretrained(name)
    return model

def getTokenizer(name):
    global dicModels
    name = dicModels[name]
    return AutoTokenizer.from_pretrained(name, keep_accents=True)
