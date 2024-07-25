import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset


hf_token = "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"
model_name = "mistralai/Mistral-7B-Instruct-v0.3"

# Function to compute probability distribution
def get_prob_dist(logits):
    return F.softmax(logits, dim=-1).mean(dim=1)

tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_8bit=True, use_auth_token=hf_token)

if model.config.pad_token_id is None:
    model.config.pad_token_id = model.config.eos_token_id

dataset = load_dataset("karmat314/writingprompts-story", split="train").select(range(1))
input_text = dataset[0]['prompt']
print(input_text)
input_text = '[INST] Instruction: Generate a very short story using the given prompt ... Prompt: ' + input_text + ' Story: [/INST]'

encoded_input = tokenizer(input_text, return_tensors="pt", truncation=True)
input_ids = encoded_input.input_ids.to("cuda")
attention_mask = encoded_input.attention_mask.to("cuda")

# Generate base sample
torch.manual_seed(np.random.randint(0, 10000))
with torch.no_grad():
    base_sample = model.generate(input_ids, attention_mask=attention_mask, max_length=512, temperature=1.5, top_p=0.9, do_sample=True)
    base_logits = model(base_sample, attention_mask=torch.ones(base_sample.shape, device=base_sample.device)).logits
    base_prob_dist = get_prob_dist(base_logits)

sample_cnt = 0
# kl_div_thres = 0.25

for i in range(100):
    # Generate auxiliary sample
    torch.manual_seed(np.random.randint(0, 10000))
    with torch.no_grad():
        aux_sample = model.generate(input_ids, attention_mask=attention_mask, max_length=512, temperature=1.5, top_p=0.9, do_sample=True)
        aux_logits = model(aux_sample, attention_mask=torch.ones(aux_sample.shape, device=aux_sample.device)).logits
        aux_prob_dist = get_prob_dist(aux_logits)

    # Ensure base_prob_dist and aux_prob_dist have the same shape
    min_len = min(base_prob_dist.shape[-1], aux_prob_dist.shape[-1])
    base_prob_dist_trimmed = base_prob_dist[:, :min_len]
    aux_prob_dist_trimmed = aux_prob_dist[:, :min_len]
    
    # Calculate KL divergence
    kl_div_ = F.kl_div(base_prob_dist_trimmed.log(), aux_prob_dist_trimmed, reduction='batchmean')
    print(f"KL Divergence: {kl_div_.item()}")

#     sample_cnt += 1
#     if kl_div_ < kl_div_thres:
#         break

# base_generated_text = tokenizer.decode(base_sample[0], skip_special_tokens=True)
# final_aux_generated_text = tokenizer.decode(aux_sample[0], skip_special_tokens=True)
# print(f"Number of samples: {sample_cnt}", f"Base sample: {base_generated_text.split('Story:')[-1]}", f"Auxiliary sample: {final_aux_generated_text.split('Story:')[-1]}")