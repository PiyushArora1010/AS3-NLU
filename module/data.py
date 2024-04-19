from datasets import load_dataset

def tokenize_data_headline(tokenizer, example):
    source = example['input']
    target = example['target']
    source = tokenizer(source, truncation=True, padding='max_length', max_length=256)
    target = tokenizer(target, truncation=True, padding='max_length', max_length=256)
    return {'input_ids': source.input_ids, 'attention_mask': source.attention_mask, 'labels': target.input_ids}


def IndicHeadlineGenerationData(tokenizer, samples=1000):
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

    dataset['train'] = dataset['train'].shuffle().select(range(samples))

    return dataset['train'], dataset['validation'], dataset['test']
