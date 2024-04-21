from datasets import load_dataset, DatasetDict

def tokenize_data_headline(tokenizer, example):
    source = example['input']
    target = example['target']
    source = tokenizer(source, truncation=True, padding='max_length', max_length=256)
    target = tokenizer(target, truncation=True, padding='max_length', max_length=256)
    return {'input_ids': source.input_ids.flatten(), 'attention_mask': source.attention_mask.flatten(), 'labels': target.input_ids.flatten()}


def IndicHeadlineGenerationData(tokenizer, samples=1000):
    # ['id', 'input', 'target', 'url']
    dataset = load_dataset("ai4bharat/IndicHeadlineGeneration","hi")

    dataset['train'] = dataset['train'].shuffle().select(range(samples))
    dataset['validation'] = dataset['validation'].shuffle().select(range(samples))
    dataset['test'] = dataset['test'].shuffle().select(range(samples))

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


def tokenize_data_samanantar(tokenizer, example):
    """
    Tokenizes the 'source' and 'target' columns of an example using the given tokenizer.
    """
    # Tokenize source and target texts
    tokenized_input = tokenizer(example['src'], padding='max_length', truncation=True, return_tensors='pt')
    tokenized_target = tokenizer(example['tgt'], padding='max_length', truncation=True, return_tensors='pt')

    # Prepare a dictionary with tokenized data
    return {
        'input_ids': tokenized_input['input_ids'].flatten(),
        'attention_mask': tokenized_input['attention_mask'].flatten(),
        'labels': tokenized_target['input_ids'].flatten()
    }

def IndicTranslationData(tokenizer, samples=1000):
    """
    Loads and processes the ai4bharat/samanantar dataset for the given language pair.
    """
    # Load the dataset for the specified language pair (e.g., 'hi-en' for Hindi-English)
    dataset_temp = load_dataset("ai4bharat/samanantar", "hi")

    # Split dataset into train (80%) and test (20%)
    train_test = dataset_temp['train'].train_test_split(test_size=0.2, seed=42)

    # Split the train set further into train (80%) and validation (20%)
    train_val = train_test['train'].train_test_split(test_size=0.2, seed=42)

    # Assign the train, validation, and test sets
    dataset = DatasetDict({
    'train': train_val['train'],
    'test': train_test['test'],
    'validation': train_val['test']})

    #sampling
    dataset['train'] = dataset['train'].shuffle().select(range(samples))
    dataset['validation'] = dataset['validation'].shuffle().select(range(samples))
    dataset['test'] = dataset['test'].shuffle().select(range(samples))

    # Apply tokenization to the train, validation, and test splits
    dataset['train'] = dataset['train'].map(lambda x: tokenize_data_samanantar(tokenizer, x), batched=True)
    dataset['validation'] = dataset['validation'].map(lambda x: tokenize_data_samanantar(tokenizer, x), batched=True)
    dataset['test'] = dataset['test'].map(lambda x: tokenize_data_samanantar(tokenizer, x), batched=True)

    # Remove unused columns
    dataset['train'].remove_columns(['src', 'tgt'])
    dataset['validation'].remove_columns(['src', 'tgt'])
    dataset['test'].remove_columns(['src', 'tgt'])

    # Set the format of the dataset to PyTorch tensors
    dataset['train'].set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    dataset['validation'].set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    dataset['test'].set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    # Shuffle and sample the specified number of examples from each split
    # dataset['train'] = dataset['train'].shuffle().select(range(samples))
    # dataset['validation'] = dataset['validation'].shuffle().select(range(samples))
    # dataset['test'] = dataset['test'].shuffle().select(range(samples))

    return dataset['train'], dataset['validation'], dataset['test']
    # return dataset['train']