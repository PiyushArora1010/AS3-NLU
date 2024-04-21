from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import EncoderDecoderModel, BertGenerationEncoder, BertGenerationDecoder

dicModels = {
    'bert-base': 'ai4bharat/indic-bert',
    'bart-base': 'Someman/bart-hindi',
    't5-base': 'google/mt5-base',
}

def getModel(name):
    global dicModels

    name = dicModels[name]
    if 'bert' in name:
        encoder = BertGenerationEncoder.from_pretrained(name)
        decoder = BertGenerationDecoder.from_pretrained(name, add_cross_attention=True, is_decoder=True)
        model = EncoderDecoderModel(encoder=encoder, decoder=decoder)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(name)
    return model

def getTokenizer(name):
    global dicModels
    name = dicModels[name]
    return AutoTokenizer.from_pretrained(name, keep_accents=True)