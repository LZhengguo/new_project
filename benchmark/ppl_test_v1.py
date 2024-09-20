from datasets import load_dataset
import torch
import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch import nn

device = 'cuda'
model_name = "/home/lzg/Llama2/Llama-2-7b-hf"
# model_name = "/home/lzg/minicpm/MiniCPM3-4B"
# model_name = "/home/lzg/Llama3/Meta-Llama-3-8B"
# model_name = "/home/lzg/Llama3/Meta-Llama-3.1-8B"
# model_name = "/home/lzg/Qwen/Qwen-7B"
enc = AutoTokenizer.from_pretrained(
    model_name, trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
testenc = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
testenc = enc("\n\n".join(testenc["text"]), return_tensors="pt")

seqlen = 2048
testenc = testenc.input_ids.to(device)


nsamples = testenc.numel() // seqlen
model = model.eval()
print(model.dtype)
nlls = []
for i in tqdm.tqdm(range(nsamples), desc="evaluating..."):
    batch = testenc[:, (i * seqlen) : ((i + 1) * seqlen)].to(device)
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