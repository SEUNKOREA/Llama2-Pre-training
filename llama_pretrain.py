from transformers import (
    AutoConfig, AutoModelForCausalLM, AutoTokenizer, AutoModel, AutoModelForPreTraining,
    LlamaTokenizer, LlamaForCausalLM, # linear head on top of LlamaModel
    LlamaModel, # inherits PreTrainedModel
    DataCollatorForLanguageModeling,
    Trainer, TrainingArguments
)
import torch
import jsonlines
import datasets as ds
import os
import deepspeed

data_path = "/home/ubuntu/Llama2_PT/news_data_en_short.jsonl"
deepspeed_config_path = "/home/ubuntu/Llama2_PT/zero3_config.json"

def prepare_dataset(tokenizer, chunk_size):
    with jsonlines.open(data_path) as f:
        data = [l for l in f]   
    
    data_dict = {
        "text": [d["text"] for d in data]
    }

    raw_dataset = ds.Dataset.from_dict(data_dict)

    def split_chunks(lst):
        for i in range(0, len(lst), chunk_size):
            yield lst[i:i+chunk_size]
            
    def tokenize(element):
        ## 배치 속 문서를 토큰화 한 후 합쳐서 청크 나눔

        outputs = tokenizer(
            element["text"],
            truncation = False
        ) # tokenizer should automatically add bos/eos tokens

        input_ids = []
        for ids in outputs["input_ids"]:
            input_ids.extend(ids)

        input_batch = []
        for chunk in list(split_chunks(input_ids)):
            if len(chunk) == chunk_size:
                input_batch.append(chunk)

        return {
            "input_ids": input_batch
        }

    tokenized_dataset = raw_dataset.map(
        tokenize,
        batched=True,
        remove_columns = raw_dataset.column_names
    )

    return tokenized_dataset
    

if __name__ == "__main__":
    context_length = 2048 # llama2 4096
    base_model = "meta-llama/Llama-2-7b-hf"
    output_dir = "/home/ubuntu/llama-pretrain-b64"
    
    os.makedirs(output_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("device: ", device)
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false" # avoid warning: avoid using tokenizer before the fork

    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        add_bos_token=True,
        add_eos_token=True,
        token="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"
        )
    
    train_dataset = prepare_dataset(tokenizer, context_length)
    
    # zero3 필요
    with deepspeed.zero.Init():
        config, unused_kwargs = AutoConfig.from_pretrained(
            base_model,
            return_unused_kwargs=True,
            max_position_embeddings = context_length,
            vocab_size = len(tokenizer),
    ### below are attention is all you need transformer base parameters
            # hidden_size = 4096,
            # intermediate_size = 4096,
            # num_hidden_layers = 6,
            # num_attention_heads = 8,
            # num_key_value_heads = 8,
        )
        """
        if num_key_value_heads=num_attention_heads: MHA
        if num_key_value_heads=1: MQA
        otherwise: GQA
        """
    
        print("unused kwargs: ", unused_kwargs)
        print("config", config)
        model = LlamaForCausalLM(config)
    
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(
        tokenizer, mlm=False
    )
    
    args = TrainingArguments(
        output_dir = output_dir, # should be cloned repo if push_to_hub=True
        overwrite_output_dir = True,
    ### args to check for memory optimization START
        deepspeed = deepspeed_config_path,
        fp16 = True,
        per_device_train_batch_size = 4,    
        gradient_checkpointing=True,
        gradient_accumulation_steps=4,
    ### args to check for memory optimization END
        # max_steps=1, # overrides num_train_epochs
        num_train_epochs = 1,
        optim="adamw_torch", # default adamw_torch
        weight_decay = 0.1,
        warmup_ratio=0.01, # overrides warmup_steps
        lr_scheduler_type = "cosine",
        learning_rate = 5e-5, # default 5e-5
        max_grad_norm = 1.0, # default 1.0 for gradient clipping
        report_to="none", # disable wandb
        save_strategy="steps",
        save_total_limit=1,
        # push_to_hub = True,
        # hub_model_id = "lectura/llama-pretrain-b64",
        # hub_private_repo = True
    )

    import json
    with open(output_dir+ "/args.json", "w") as f:
        json.dump(args.to_dict(), f, ensure_ascii = False)
        
    
    model.train()
    
    trainer = Trainer(
        model = model,
        tokenizer = tokenizer,
        args = args,
        data_collator = data_collator,
        train_dataset = train_dataset
    )

    trainer.train()
    trainer.save_model()