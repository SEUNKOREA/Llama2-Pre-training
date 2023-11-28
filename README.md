# Pre-training Llama2-7b from Scratch 

## Requirements
아래의 패키지를 `pip install`를 이용해서 설치한다.

    deepspeed
    transformers
    accelerate
    datsets
    nvidia-ml-py3
    jsonlines

`mpi4py`는 `sudo apt install` 커맨드를 사용한다.

    sudo apt install python3-mpi4py


<br>


## 🌈 How to Start Pre-train Llama2-7b
Llama2 7b 모델을 영어뉴스 데이터로 ZeRO-3를 이용해서 multi-gpu로 사전학습시키는 코드입니다.

    deepspeed --num_gpus={사용할 GPU 개수} pretrain.py

<br>

- `--num_gpus`를 지정하지 않으면 사용가능한 모든 gpu를 사용하게 됩니다.
    ``````
    deepspeed pretrain.py
- 해당 코드를 실행하기 전 [결과가 저장될 로컬경로](https://github.com/SEUNKOREA/Llama2_PT/blob/85bca4868bfbf81864c9ea9df0854281f63ac794/pretrain.py#L55)(`output_dir`), [Deepspeed config 파일 경로](https://github.com/SEUNKOREA/Llama2_PT/blob/85bca4868bfbf81864c9ea9df0854281f63ac794/pretrain.py#L58C1-L58C22)(`deepspeed_config_path`), [허깅페이스 리포지토리](https://github.com/SEUNKOREA/Llama2_PT/blob/85bca4868bfbf81864c9ea9df0854281f63ac794/pretrain.py#L89)(`hub_model_id`) 등을 본인의 환경에 맞게 알맞게 설정하세요.
- 훈련과 관련된 하이퍼파라미터 설정은 [여기서](https://github.com/SEUNKOREA/Llama2_PT/blob/85bca4868bfbf81864c9ea9df0854281f63ac794/pretrain.py#L59) 수정할 수 있습니다.
    - 각 파라미터에 대한 자세한 설명은 [여기서](https://huggingface.co/docs/transformers/v4.35.2/en/main_classes/trainer#transformers.TrainingArguments) 확인할 수 있습니다.
- ZeRO-3와 관련된 DeepSpeed 설정은 [여기서](https://github.com/SEUNKOREA/Llama2_PT/blob/main/ds_config.json) 수정할 수 있습니다.
    - 각 설정에 대한 자세한 설명은 [여기서](https://www.deepspeed.ai/docs/config-json/) 확인할 수 있습니다.
- 해당 [코드](https://github.com/SEUNKOREA/Llama2_PT/blob/main/pretrain.py)에서는 [미리 토크나이징된 청크사이즈를 4096으로 토크나이징한 영어뉴스 데이터](https://huggingface.co/datasets/leeseeun/tokenized_news_2gb_4096)를 사용합니다. 
    - 메모리 문제 등 다른 이유로 청크사이즈를 다르게 하여 데이터를 토크나이징 하고 싶은 경우, [이 부분](https://github.com/SEUNKOREA/Llama2_PT/blob/85bca4868bfbf81864c9ea9df0854281f63ac794/pretrain.py#L32C1-L32C1)의 주석을 해제하고 `max_length`를 원하는 청크사이즈로 바꿔서 실행하세요.
    - `max_length`는 데이터를 모델에 어떤 청크사이즈로 잘러서 넣을지를 의미하며 Meta의 Llama2 모델에서는 4,096을 사용하였습니다.
    - 원본 데이터의 사이즈가 큰 경우, 토크나이징 과정에 많은 시간이 소요될 수 있습니다. 이런 경우 미리 특정 청크사이즈의 크기로 토크나이징을 진행한 후에 허브에 해당 데이터셋을 업로드 후 불러와서 사용하는 방법을 추천합니다. 해당 방법은 아래의 ["Tokenize Dataset"](https://github.com/SEUNKOREA/Llama2_PT/tree/main#-tokenize-dataset) 가이드에서 확인할 수 있습니다.

<br>
<br>

## 🌍 Generate Sentences
사전학습된 Llama2-7b 모델을 바탕으로 "I wanna go"의 뒷 문장을 생성하는 코드입니다.

    python3 generate.py

<br>

- 해당 코드를 실행하기 전 [사전학습된 모델의 체크포인트가 저장된 경로](https://github.com/SEUNKOREA/Llama2_PT/blob/85bca4868bfbf81864c9ea9df0854281f63ac794/generate.py#L16)가 올바른지 확인하세요.
- "I wanna go"뿐만 아니라 다른 문장으로도 generation을 하고 싶은 경우 [해당 부분](https://github.com/SEUNKOREA/Llama2_PT/blob/85bca4868bfbf81864c9ea9df0854281f63ac794/generate.py#L22C1-L22C1)에 다른 문장을 추가하면 됩니다.

<br>

## 🔥 Tokenize Dataset

    python3 preprocess.py

<br>

- 데이터를 직접 토크나이징하고 싶은 경우, 해당 가이드를 참고하세요.
- 해당 코드는 [(주)딥로딩](https://www.deeploading.com/)으로부터 제공받은 영어뉴스 데이터(news_data_en.jsonl)를 이용해서 데이터를 전처리하고 청크사이즈를 지정해서 해당 청크사이즈를 기준으로 토크나이징을 진행하는 코드입니다. 
    - 해당 데이터는 아래와 같은 형식입니다.
    ``````
    {"text": "(CNN)Share, and your gift will be multiplied. That may sound like an esoteric adage, but when Zully Broussard selflessly decided to give one of her kidneys to a stranger, her generosity paired up with big data. It resulted in six patients receiving transplants. That surprised and wowed her. \"I thought I was going to help this one person who I don't know, but the fact that so many people can have a life extension, that's pretty big,\" Broussard told CNN affiliate KGO. She may feel guided in her generosity by a higher power. \"Thanks for all the support and prayers,\" a comment on a Facebook page in her name read. ...생략... "}

- 다른 데이터를 사용하고 싶은 경우에도 해당 코드를 사용할 수 있지만 코드에 일부 수정이 필요할 수 있습니다.
- 해당 코드를 실행하기 전 [원본 데이터셋 경로](https://github.com/SEUNKOREA/Llama2_PT/blob/85bca4868bfbf81864c9ea9df0854281f63ac794/preprocess.py#L55)와 [청크 사이즈](https://github.com/SEUNKOREA/Llama2_PT/blob/85bca4868bfbf81864c9ea9df0854281f63ac794/preprocess.py#L62), [허깅페이스 레포지토리](https://github.com/SEUNKOREA/Llama2_PT/blob/85bca4868bfbf81864c9ea9df0854281f63ac794/preprocess.py#L66)를 올바르게 설정했는지 확인하세요.

<br>

## 🧩 CPU offload
- `CUDA Out Of Memory` 문제가 발생한 경우, CPU offload를 사용하여 해결할 수 있습니다.
- [해당 파일](https://github.com/SEUNKOREA/Llama2_PT/blob/main/ds_config.json)에 다음과 같이 *“offload_optimizer”, “offload_param”* 을 추가하면 됩니다.
    ``````
    {
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu"
        },
        "offload_param":{
            "device": "cpu"
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
    ...생략...
- 단, 이 작업을 수행할 경우 사전학습에 오랜 시간이 소요될 수 있습니다. 

<br>

## Acknowlegemnets
해당 코드를 테스트하고 실행함에 있어서 [(주)딥로딩](https://www.deeploading.com/)으로부터 서버를 지원받았습니다.
<br>
유용한 지원을 해주신 [(주)딥로딩](https://www.deeploading.com/)에 감사의 인사를 전합니다.
