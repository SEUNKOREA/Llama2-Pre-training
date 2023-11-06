from model import load_model, get_max_length
from datasets import load_dataset
import os
import torch
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from check_gpu_memory import print_gpu_memory
import json
from preprocess import make_HFdataset, preprocess_dataset
# pip install -U tokenizers
# pip install datasets transformers jsonlines
# pip install nvidia-ml-py3
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# pip install accelerate

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true' ## avoid warning

model_name = "meta-llama/Llama-2-7b-hf" 
model, tokenizer = load_model(model_name)

## 토크나이징된 데이터 불러오기(해당경로에는 2gb를 미리 토크나이징한 데이터가 있당.)
dataset = load_dataset("leeseeun/tokenized_news_2gb_4096") # 영어 뉴스데이터 2gb 청크사이즈 4096 경로
dataset = dataset['train']
"""
DatasetDict({
    train: Dataset({
        features: ['input_ids'],
        num_rows: 544042
    })
})
"""

### 직접 토크나이징 하는 경우
# data_path = "./news_data_en_short.jsonl"
# raw_dataset = make_HFdataset(data_path)
# max_length =  get_max_length(model) # Llama2 4096
# print(f"max length: {max_length}")
# dataset = preprocess_dataset(tokenizer, max_length, raw_dataset)


print(tokenizer)
print(model)
print(model.config)


if torch.cuda.is_available():
    device = torch.device("cuda")
    n_gpus = torch.cuda.device_count()
    print("device: ", device, n_gpus)
else:
    device = torch.device("cpu")
    print("device: ", device)

# print_gpu_memory()

output_dir = "./Llama2_PT/result"
os.makedirs(output_dir, exist_ok=True)

deepspeed_config_path = './Llama2_PT/ds_config.json'
args = TrainingArguments(
      deepspeed = deepspeed_config_path,
      fp16=False,
      fp16_full_eval=False,
      bf16 = True,
      bf16_full_eval=True,
    ## eval plan 기준 Pretrain args(eval_plan 설정값에 따름) START
      optim="adamw_torch", # default adamw_torch, eval plan 설정값 1-bit Adam
      per_device_train_batch_size=4, # gpu 8개 기준 batch_size=128로 맞추기 위함
      gradient_accumulation_steps=4, # gpu 8개 기준 batch_size=128로 맞추기 위함
      weight_decay=0.1, 
      max_grad_norm = 1.0, # default 1.0 for gradient clipping, eval plan 설정값 1
      # num_train_epochs=10, # eval plan 설정값 iter 40,000
      max_steps=1000, # gradient_acc_steps=4이기 때문에 /4
    ## eval plan 기준 Pretrain args(eval_plan 설정값에 따름) END

      gradient_checkpointing=True,

      warmup_steps=0, # default 0
      learning_rate=0.0001, # default 0.00005
      # lr_scheduler_type="cosine",

      logging_steps=100,
      report_to="none", # disable wandb
      save_strategy="steps", # save_steps마다 저장이 수행됩니다.
      save_steps=0.5, # save_strategy가 "steps"인 경우, 두 개의 체크포인트 저장 사이의 업데이트 단계 수
      # save_total_limit=1, # 가장 최근의 n개의 모델이 저장된다.
      output_dir=output_dir,
      overwrite_output_dir=True,
      push_to_hub=True, # 모델을 저장할 때마다 모델을 Hub에 푸시할지 여부
      hub_model_id = 'leeseeun/llama2-7b-en-pretrain', #로컬 output_dir과 동기화할 리포지토리의 이름
)
model.train()

# print_gpu_memory()
trainer = Trainer(
    model=model, 
    tokenizer=tokenizer,
    args=args,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    train_dataset=dataset
)

trainer.train()
print("Complete Train")
trainer.save_model()
print("Save Model")
trainer.push_to_hub()