import torch
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Convert model from float32 to int4")
    parser.add_argument('--wikitext_validation', action='store_true', help="Use wikitext-2-raw-v1:validation to validate")
    parser.add_argument('--wikitext_test', action='store_true', help="Use wikitext-2-raw-v1:test to validate")
    parser.add_argument('--c4', action='store_true', help="Use allenai/c4:en:validation to validate")
    return parser.parse_args()

def main():
    args = parse_args()

    # model_name_set
    ### llama系列都没啥问题
    # model_name = "/home/lzg/Llama2/Llama-2-7b-hf"
    # quantized_model_dir = "/home/lzg/quantization_model/Llama2/quantized_llama2_7b_int4"
    model_name = "/home/lzg/Llama3/Meta-Llama-3-8B"
    quantized_model_dir = "/home/lzg/quantization_model/Llama3/quantized_llama3_8b_int4"
    # model_name = "/home/lzg/Llama3/Meta-Llama-3.1-8B"
    # quantized_model_dir = "/home/lzg/quantization_model/Llama3/quantized_llama3.1_8b_int4"

    ### 非llama得再改改
    # model_name = "/home/lzg/minicpm/MiniCPM3-4B"
    # quantized_model_dir = "/home/lzg/quantization_model/minicpm/quantized_minicpm3_4b_int4"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    # 设置量化配置（int4）
    quantize_config = BaseQuantizeConfig(bits=4, group_size=128, desc_act=False)

    model = AutoGPTQForCausalLM.from_pretrained(
        model_name,
        quantize_config=quantize_config,
        trust_remote_code=True
    ).to(device)

    # 选校准集
    ### 用wikitest:test
    if args.wikitext_validation or args.wikitext_test:
        if args.wikitext_validation:
            data = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
        else:
            data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        tokenized_data = tokenizer("\n\n".join(data["text"]), return_tensors="pt")

    # 用c4
    if args.c4:
        data_files = {"validation": "en/c4-validation.*.json.gz"}
        data = load_dataset("allenai/c4", data_files=data_files, split="validation")
        # data = load_dataset('allenai/c4', data_files='en/c4-train.0000*-of-01024.json.gz')
        # 这玩意太大了，只选用部分看看效果
        tokenized_data = tokenizer("\n\n".join(data['text'][:100]), return_tensors='pt')

    seq_len = tokenized_data.input_ids.shape[1]
    stride = 2048
    examples_ids = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + stride, seq_len)
        trg_len = end_loc - prev_end_loc
        if trg_len < stride:
            break
        input_ids = tokenized_data.input_ids[:, begin_loc:end_loc].to(device)
        attention_mask = torch.ones_like(input_ids)
        examples_ids.append({'input_ids': input_ids, 'attention_mask': attention_mask})

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    # quantize
    model.quantize(
        examples_ids,
        batch_size=1,
        use_triton=True,
    )

    model.save_quantized(quantized_model_dir, use_safetensors=True)

    print(f"量化后的模型已保存到: {quantized_model_dir}")