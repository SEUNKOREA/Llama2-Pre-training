import jsonlines
import datasets as ds
from functools import partial
from check_gpu_memory import print_gpu_memory
from model import load_model
import json
import os

def make_HFdataset(data_path):
    print("Transforming jsonl -> hf datasets...")
    with jsonlines.open(data_path) as f:
        data = [l for l in f] 

    data_dict = {
        'text': [d['text'] for d in data]
    }

    raw_dataset = ds.Dataset.from_dict(data_dict)
    return raw_dataset

def split_chunk(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i+chunk_size]

def tokenize(batch, tokenizer):
    return tokenizer(
        batch['text'],
        truncation=False
    )

def preprocess_dataset(tokenizer, max_length, raw_dataset):
    print("Preprocessing datset...")

    _tokenize = partial(tokenize, tokenizer=tokenizer)
    tokenized_dataset = raw_dataset.map(
        _tokenize,
        batched=True,
        remove_columns=raw_dataset.column_names
    )

    input_ids = []
    for ids in tokenized_dataset['input_ids']:
        input_ids.extend(ids)
    
    input_batch = []
    for chunk in list(split_chunk(input_ids, chunk_size=max_length)):
        if len(chunk) == max_length:
            input_batch.append(chunk)
    temp = {"input_ids": input_batch}
    dataset = ds.Dataset.from_dict(temp)
    return dataset

if __name__ == '__main__':
    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true' ## avoid warning

    data_path = './news_data_2gb.jsonl'  ## num_rows: 897851
    raw_dataset = make_HFdataset(data_path)

    model_name = "meta-llama/Llama-2-7b-hf" 
    _, tokenizer = load_model(model_name)
    
    # data_chunk_size = get_max_length(model) # Llama2 4096 -> 다른 길이를 원한다면 직접 지정
    data_chunk_size = 1024
    dataset = preprocess_dataset(tokenizer, data_chunk_size, raw_dataset)

    # 허깅페이스 업로드
    dataset.push_to_hub("leeseeun/tokenzied_news_2gb_data")

    

