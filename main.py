import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig, TextStreamer


# Requirements:
# * Linux/MacOs (The bitsandbytes library doesn't work on Windows. I used WSL.)
# * 24 GB of GPU memory
# * ~100 gb of diskspace

# Notes:
# This code, as is, will only run with a gpu and linux. I have a 4090 with 24 gb of memory.
# If you run this with a cpu, you'll have to tweak the code. It will take 90 gb of cpu
# memory and be very slow. I didn't bother making it configurable since it was unusable.

# Full steps:
# mkdir -p ~/miniconda3
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
# bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
# rm -rf ~/miniconda3/miniconda.sh
# ~/miniconda3/bin/conda init bash
# ~/miniconda3/bin/conda init zsh
# conda create --name py311 python=3.11
# conda activate py311
# conda install cuda -c nvidia
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# pip install accelerate
# pip install bitsandbytes
# pip install flash-attn --no-build-isolation
# pip install scipy


class TextStreamerWithoutSeparator(TextStreamer):
    def on_finalized_text(self, text: str, stream_end: bool = False):
        clean_text = text.replace("</s>", "")
        print(clean_text, flush=True, end="" if not stream_end else None)


def main():
    model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                 quantization_config=bnb_config,
                                                 attn_implementation="flash_attention_2")

    generation_config = GenerationConfig.from_model_config(model.config)
    generation_config.max_new_tokens = 2000
    generation_config.min_length = 1
    generation_config.do_sample = True
    generation_config.temperature = 1.0
    generation_config.top_p = 0.6
    generation_config.eos_token_id = tokenizer.eos_token_id
    generation_config.pad_token_id = generation_config.eos_token_id
    streamer = TextStreamerWithoutSeparator(tokenizer, skip_prompt=True)
    while True:
        next_input = input("> ")
        if next_input == 'exit':
            break
        print()
        text = f"<s> [INST] {next_input} [/INST] "
        inputs = tokenizer(text, return_tensors="pt").to("cuda")
        model.generate(**inputs, streamer=streamer, generation_config=generation_config)


if __name__ == '__main__':
    main()
