import argparse
from datasets import load_dataset
import torch
import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch import nn
from auto_gptq import AutoGPTQForCausalLM

def parse_args():
    parser = argparse.ArgumentParser(description="Model Inference and measure perplexity")
    parser.add_argument('--float16', action='store_true', help="Use float16 precision")
    parser.add_argument('--offical_quantized_model', action='store_true', help="Testing offical quantized model")
    parser.add_argument('--self_quantized_model', action='store_true', help="Testing self quantized model")
    return parser.parse_args()

def main():
    args = parse_args()
    
    device_ = "cuda"

    # usual model name

    # model_name = "/home/lzg/Llama2/Llama-2-7b-hf"
    # model_name = "/home/lzg/minicpm/MiniCPM3-4B"
    model_name = "/home/lzg/Llama3/Meta-Llama-3-8B"
    # model_name = "/home/lzg/Llama3/Meta-Llama-3.1-8B"
    # model_name = "/home/lzg/Qwen/Qwen-7B"

    # offical quantized model name
    offical_quantized_model_name = "/home/lzg/Llama3/Meta-Llama-3-8B-Instruct-GPTQ-Int4"

    # self quantized model name set
    original_model_name = "/home/lzg/Llama3/Meta-Llama-3-8B"
    quantized_model_name = "/home/lzg/quantization_model/Llama3/quantized_llama3_8b_int4"

    # 测试不同类型的模型
    if args.self_quantized_model:
        print("start testing self quantized model...")
        enc = AutoTokenizer.from_pretrained(
            original_model_name, 
            trust_remote_code=True
        )
        model = AutoGPTQForCausalLM.from_quantized(
            quantized_model_name,
            device="cuda"
        )
    elif args.offical_quantized_model:
        print("start testing offical quantized model...")
        enc = AutoTokenizer.from_pretrained(
            offical_quantized_model_name, 
            trust_remote_code=True
        )
        # 底下两种加载方式都行，但是第一个device必须等于cuda0，不然就有bug
        # model = AutoGPTQForCausalLM.from_quantized(
        #     offical_quantized_model_name,
        #     device="cuda:0"
        # )
        model = AutoModelForCausalLM.from_pretrained(
            offical_quantized_model_name,
            device_map="auto"
        )   
    else:
        print("start ppl testing for usual model...")
        enc = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto"
        )

    if args.float16:
        print("Using float16 precision...")
        model.half()

    testenc = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    testenc = enc("\n\n".join(testenc["text"]), return_tensors="pt")

    seqlen = 2048
    testenc = testenc.input_ids.to(device_)


    nsamples = testenc.numel() // seqlen
    model = model.eval()
    print("current precision is <", model.dtype, ">")
    nlls = []
    for i in tqdm.tqdm(range(nsamples), desc="evaluating..."):
        batch = testenc[:, (i * seqlen) : ((i + 1) * seqlen)].to(device_)
        with torch.no_grad():
            lm_logits = model(batch).logits
        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = testenc[
            :, (i * seqlen) : ((i + 1) * seqlen)
        ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        neg_log_likelihood = loss.float() * seqlen
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))

    results = {"ppl": ppl.item()}
    print(results)

if __name__ == "__main__":
    main()