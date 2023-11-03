from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from model import load_model, get_max_length
from transformers import AutoTokenizer, LlamaForCausalLM
import torch
from tqdm import tqdm 

device = torch.device("cuda:0")

tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-2-7b-hf", 
    add_bos_token=True) # default don't add eos
tokenizer.pad_token = tokenizer.eos_token # 토크나이저의 패딩토큰을 종료토큰으로 설정
tokenizer.padding_side = "left"


ckpt_path = "leeseeun/model_temp"
model = LlamaForCausalLM.from_pretrained(
        ckpt_path,
        torch_dtype=torch.float16,
)

eval_dataset = ["I wanna go",]
results = []
        
model.to(device).eval()
    
for idx, d in tqdm(enumerate(eval_dataset), desc="Generating", total=len(eval_dataset)):
    prompt = d
    model_input = tokenizer(prompt, return_token_type_ids=False,return_tensors="pt").to(device)
    with torch.no_grad():
        output_text = tokenizer.decode(model.generate(**model_input, max_new_tokens=100)[0], )
    results.append(output_text)
for r in results: 
    print(r)