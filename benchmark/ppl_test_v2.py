from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 加载模型和tokenizer
# model_name = "/datasets/models/llama-160m"
model_name = "/home/lzg/Llama2/Llama-2-7b-hf"
# model_name = '/home/lzg/opt/opt-1.3b'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model.eval()

# 加载测试数据集
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
encodings = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")

stride = 2048  # 每次前向传播的序列长度
seq_len = encodings.input_ids.size(1)  # 数据的总长度
nlls = []
prev_end_loc = 0

# 逐步计算NLL并累积计算Perplexity
for begin_loc in tqdm(range(0, seq_len, stride)):
    end_loc = min(begin_loc + stride, seq_len)
    trg_len = end_loc - prev_end_loc
    input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
    
    # 目标序列，用于计算损失
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100  # mask掉不需要的部分

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
        neg_log_likelihood = outputs.loss  # 直接从模型的输出获取NLL (CrossEntropy Loss)

    nlls.append(neg_log_likelihood)

    prev_end_loc = end_loc
    if end_loc == seq_len:
        break

# 计算PPL
ppl = torch.exp(torch.stack(nlls).mean())

print(f"Perplexity: {ppl.item()}")