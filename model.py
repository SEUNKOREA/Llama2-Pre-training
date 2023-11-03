import torch
from transformers import AutoConfig, LlamaForCausalLM, AutoTokenizer
import os
import deepspeed
from check_gpu_memory import print_gpu_memory

### 토크나이저 & 모델을 로드하는 함수 작성
def load_model(model_name):
    print("Setting Configuration...")
    with deepspeed.zero.Init():
        config, unused_kwargs = AutoConfig.from_pretrained(
        model_name,
        return_unused_kwargs=True,
        max_position_embeddings=4096,
        use_cache=False,
        token="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"
        # config = LlamaConfig(
        #     max_position_embeddings=4096,
        #     rms_norm_eps=1e-05,
        #     torch_dtype="float16",
        #     #device_map="auto", # 모델을 사용 가능한 자원(GPU)에 효율적으로 배치하기 위한 자동 장치 매핑 설정
        #     #max_memory= {i: max_memory for i in range(n_gpus)} # 각 GPU에 할당할 최대 메모리를 설정
        # )
        )
        print("unused kwargs: ", unused_kwargs)
        print("Llama2 Configuration Complete !\n")

        print("Load Model based on config...")
        model = LlamaForCausalLM(config)
        print("Loaded Initialized Model based on config!\n")
        

    os.environ["TOKENIZERS_PARALLELISM"] = "false" # avoid warning: avoid using tokenizer before the fork
    print("Load Pretrained Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        add_bos_token=True,
        add_eos_token=True,
        token="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"
    )
    # Needed for Llama Tokenizer
    tokenizer.pad_token = tokenizer.eos_token # 토크나이저의 패딩토큰을 종료토큰으로 설정
    print("Loaded Pretrained Llama2 Tokenizer!\n")
    return model, tokenizer

### 모델 설정을 기반으로 최대 시퀀스 길이를 계산해서 반환하는 함수(기본값 2048)
def get_max_length(model):
    max_length = None

    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            print(f"Found max lenth: {max_length}")
            break

    if not max_length:
        max_length = 2048
        print(f"Using default max length: {max_length}")

    return max_length