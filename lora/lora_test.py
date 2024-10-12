import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
import torch.nn.functional as F


# 加载 LLaMA 模型和 tokenizer
model_name = "/home/lzg/Llama2/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# # 冻结 LLaMA 主干模型的所有参数
# for param in model.parameters():
#     param.requires_grad = False

# # 设置 LoRA 的配置
# lora_config = LoraConfig(
#     task_type=TaskType.CAUSAL_LM,  # 任务类型是自回归语言模型
#     r=8,  # LoRA 的秩（rank）
#     lora_alpha=32,  # LoRA 的缩放系数
#     lora_dropout=0.1,  # Dropout
#     target_modules=["q_proj", "v_proj"]  # 选择应用 LoRA 的模块
# )

# # 将 LoRA 加入模型
# model = get_peft_model(model, lora_config)

# # 检查哪些参数是可训练的
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(f"Trainable parameter: {name}")

# 将模型移到 GPU
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 训练 LoRA 层（这里只展示如何进行推理或训练部分）
# 以下是一个简单的推理代码，你可以根据需要扩展成训练流程
text = "The capital of France is"
inputs = tokenizer(text, return_tensors="pt").to(device)

# 前向传播
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

print(logits.shape)
# 获取预测的下一个 token
probs = F.softmax(logits[:, -1, :], dim=-1)
# predicted_token_id = torch.argmax(probs, dim=-1)
# predicted_token = tokenizer.decode(predicted_token_id)
predicted_token_id = torch.multinomial(probs, num_samples=1)
for item in predicted_token_id:
    predicted_token = tokenizer.decode(predicted_token_id)
    print(f"Predicted next token: {predicted_token}")