from datasets import load_dataset

def tokenize_data_headline(tokenizer, example):
    source = example['input']
    target = example['target']
    source = tokenizer(source)
    target = tokenizer(target)
    return {'input_ids': source.input_ids, 'attention_mask': source.attention_mask, 'labels': target.input_ids}


def IndicHeadlineGenerationData(tokenizer):
    # ['id', 'input', 'target', 'url']
    dataset = load_dataset("ai4bharat/IndicHeadlineGeneration","pa")

    dataset['train'] = dataset['train'].map(lambda x: tokenize_data_headline(tokenizer, x), batched=True)
    dataset['validation'] = dataset['validation'].map(lambda x: tokenize_data_headline(tokenizer, x), batched=True)
    dataset['test'] = dataset['test'].map(lambda x: tokenize_data_headline(tokenizer, x), batched=True)

    dataset['train'].remove_columns(['id', 'url', 'target', 'input'])
    dataset['validation'].remove_columns(['id', 'url', 'target', 'input'])
    dataset['test'].remove_columns(['id', 'url', 'target', 'input'])

    dataset['train'].set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    dataset['validation'].set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    dataset['test'].set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    return dataset['train'], dataset['validation'], dataset['test']
