import argparse
import json
from turtle import filling
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from predict import load_data, construct_prompt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import random
import os
from matplotlib import pyplot as plt
random.seed(42)

# Add model path mapping
MODEL_PATH_MAP = {
    'qwen3-8b': '../../LLMs/QWEN/qwen3-8b',
    'llama3.1-8b': '../../LLMs/LLAMA/llama3.1-8b',
    'mistral-7b': '../../LLMs/mistral/mistral-7b',
    'deepseek-8b':'../../LLMs/deepseek/DeepSeek-R1-0528-Qwen3-8B'
}

class QwenModifier:
    def __init__(self,tok, model, human_token_id, machine_token_id, alpha,
                 human_mean,human_std, machine_mean,machine_std,human_basis,  machine_basis,threshold=0.4):
        self.model = model
        self.tok=tok
        self.human_token_id = human_token_id
        self.machine_token_id = machine_token_id
        self.alpha = alpha
        self.human_mean = human_mean  # [hidden_dim]
        self.human_std = human_std
        self.machine_mean = machine_mean
        self.machine_std = machine_std
        self.threshold = threshold
        self.human_basis = human_basis  # [k, H]
        self.machine_basis = machine_basis

        self.human_w = self.model.lm_head.weight[human_token_id]  # [hidden_dim]
        self.machine_w = self.model.lm_head.weight[machine_token_id]  # [hidden_dim]
        self.hook = None

    def register_hook(self):
        self.first_token_modified = False
        
        def hook_fn(module, input, output):
            if self.first_token_modified:
                return output

            if isinstance(output, tuple):
                hidden = output[0]  # [B, L, H] 只对最后一个 token 做处理
            else:
                hidden = output

            last_hidden = hidden[:, -1, :]  # [B, H]
            self.human_mean=self.human_mean.to(last_hidden.device)
            self.human_std=self.human_std.to(last_hidden.device)
            self.machine_mean=self.machine_mean.to(last_hidden.device)
            self.machine_std=self.machine_std.to(last_hidden.device)
            self.human_basis=self.human_basis.to(last_hidden.device)
            self.machine_basis=self.machine_basis.to(last_hidden.device)
            print(last_hidden.device,self.human_mean.device,self.human_std.device)
            norm_human = (last_hidden - self.human_mean) / self.human_std  # [B, L, H]
            dist_human = torch.norm(norm_human, dim=-1)  # [B, L]
            sim_human = F.cosine_similarity(last_hidden, self.human_mean.unsqueeze(0), dim=-1)  # [B]
            sim_machine = F.cosine_similarity(last_hidden, self.machine_mean.unsqueeze(0), dim=-1)  # [B]

            def project_energy(vec: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
                # vec: [B, H], basis: [k, H]
                # Returns: [B] - projection energy onto the subspace
                projection = torch.matmul(vec, basis.T)  # [B, k]
                energy = torch.norm(projection, dim=-1)  # L2 norm of projection coefficients
                return energy
            energy_human = project_energy(last_hidden, self.human_basis)  # [B]
            energy_machine = project_energy(last_hidden, self.machine_basis)  # [B]


            direction_human = self.human_w / self.human_w.norm()
            direction_machine = self.machine_w / self.machine_w.norm()

            direction_human = direction_human.to(hidden.device)
            direction_machine = direction_machine.to(hidden.device)

            perturbed = hidden.clone()
            # is_perturbe=Ture if sim_human[i] > self.threshold else False
            # print(hidden.shape)
            plt.figure(figsize=(10, 8))
            plt.subplot(1,2,1)
            print("before:")
            with torch.no_grad():
                logits_before = self.model.lm_head(last_hidden)  # [B, vocab_size]
                topk_before = torch.topk(logits_before, k=10, dim=-1)

                i = 0
                top_ids_before = topk_before.indices[i].tolist()
                top_scores_before = topk_before.values[i].tolist()
                top_tokens_before = self.tok.convert_ids_to_tokens(top_ids_before)
    
                print(f"\nSample {i} Top-10 token predictions (Before):")
                for tok, score in zip(top_tokens_before, top_scores_before):
                    print(f"{tok:10s} | logit: {score:.4f}")

                plt.bar(top_tokens_before, top_scores_before,color='#c4f5c6')
                plt.title("Before Adjustment")
                plt.xticks(rotation=45,fontsize=16)
                plt.ylabel("Logit Score")
                plt.ylim(50,None)
            
            # for i in range(last_hidden.size(0)):
                # if sim_human[i] > self.threshold:
                #     if sim_human[i]>sim_machine[i]:
                #         last_hidden[i] += self.alpha * direction_human
                #     else:
                # # elif sim_machine[i] > self.threshold:
                #         last_hidden[i] += self.alpha * direction_machine
            for i in range(last_hidden.size(0)):
                if energy_human[i] > energy_machine[i]:
                    last_hidden[i] += self.alpha * direction_human
                else:
                    last_hidden[i] += self.alpha * direction_machine
                plt.subplot(1,2,2)
                print("after:")
                with torch.no_grad():
                    logits_after = self.model.lm_head(last_hidden)  # [B, vocab_size]
                    topk_after = torch.topk(logits_after, k=10, dim=-1)   # 取 top 10
                    i = 0
                    top_ids_after = topk_after.indices[i].tolist()
                    top_scores_after = topk_after.values[i].tolist()
                    top_tokens_after = self.tok.convert_ids_to_tokens(top_ids_after)
                    print(f"\nSample {i} Top-10 token predictions (After):")
                    for tok, score in zip(top_tokens_after, top_scores_after):
                        print(f"{tok:10s} | logit: {score:.4f}")
                    plt.bar(top_tokens_after, top_scores_after,color='#ffcacb')
                    plt.title("After Adjustment")
                    plt.xticks(rotation=45,fontsize=16)
                    plt.ylabel("Logit Score")
                    plt.ylim(50,None)
        
            plt.savefig('adjustment_results.png')
            plt.show()
            #     else:
            #         continue
            # perturbed = hidden + self.alpha * (
            #     direction_human * mask_human.unsqueeze(-1).float() +
            #     direction_machine * mask_machine.unsqueeze(-1).float()
            # )

            self.first_token_modified = True
            perturbed = hidden.clone()
            perturbed[:, -1, :] = last_hidden
            if isinstance(output, tuple):
                return (perturbed,) + output[1:]
            else:
                return perturbed

        final_block = self.model.model.layers[-1]
        self.hook = final_block.register_forward_hook(hook_fn)

    def remove_hook(self):
        if self.hook is not None:
            self.hook.remove()
            self.hook = None

def load_model_and_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True,device_map='auto')
    model.eval()
    return model, tokenizer

@torch.no_grad()
def run_generation(model_name, model, tokenizer, prompt, device, max_new_tokens=50):

    if hasattr(model, "modifier") and hasattr(model.modifier, "first_token_modified"):
        model.modifier.first_token_modified = False

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output_ids = model.generate(
    **inputs, max_new_tokens=max_new_tokens, do_sample=False
    )

    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    output_text = tokenizer.decode(generated, skip_special_tokens=True)
    # print("res:",output_text)
    return output_text.strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="qwen3-8b", choices=MODEL_PATH_MAP.keys(),
                       help="Short name of the model to use")
    parser.add_argument('--text_type1', type=str, default='human', 
                       choices=['human', 'chatgpt', 'qwen_answers', 'llama_answers', 'deepseek_answers', 'gpt4', 'llama-chat', 'mistral'],
                       help='First text type to compare')
    parser.add_argument('--text_type2', type=str, default='chatgpt',
                       choices=['human', 'chatgpt', 'qwen_answers', 'llama_answers', 'deepseek_answers', 'gpt4', 'llama-chat', 'mistral'],
                       help='Second text type to compare')
    parser.add_argument('--hidden_stats_dir', type=str, default="hidden_stats/qwen3-8b")
    parser.add_argument('--alpha', type=float, default=100)
    parser.add_argument('--max_new_tokens', type=int, default=20)
    parser.add_argument('--num_samples', type=int, default=400)
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    experiment_type = 'human' if args.text_type1 == 'human' or args.text_type2 == 'human' else 'machine'

    # Get model path from mapping
    model_path = MODEL_PATH_MAP.get(args.model_name)
    if not model_path:
        raise ValueError(f"Model name '{args.model_name}' not found. Please add it to MODEL_PATH_MAP.")
    
    print("get model and tokenizer")
    model, tokenizer = load_model_and_tokenizer(model_path)

    # Automatically find token_id
    human_token_id = tokenizer.convert_tokens_to_ids("(b")
    machine_token_id = tokenizer.convert_tokens_to_ids("(a")

    # Load statistical features (mean and std)
    model_name=model_path.split('/')[-1]
    # path=os.path.join(args.hidden_stats_dir, model_name, '')
    path = args.hidden_stats_dir
    print(path)
    human_mean = np.load(f"{path}/human_mean.npy")
    human_std = np.load(f"{path}/human_std.npy")
    machine_mean = np.load(f"{path}/machine_mean.npy")
    machine_std = np.load(f"{path}/machine_std.npy")
    human_basis = np.load(f"{path}/human_svd_basis.npy")  # [k, H]
    machine_basis = np.load(f"{path}/machine_svd_basis.npy")  # [k, H]

    human_basis = torch.tensor(human_basis).to(device)
    machine_basis = torch.tensor(machine_basis).to(device)
    human_mean = torch.tensor(human_mean).to(device)
    human_std = torch.tensor(human_std).to(device)
    machine_mean = torch.tensor(machine_mean).to(device)
    machine_std = torch.tensor(machine_std).to(device)
    # print(human_mean)


    modifier = QwenModifier(tokenizer,model, human_token_id, machine_token_id, args.alpha,
                            human_mean,human_std, machine_mean,machine_std, human_basis,machine_basis)
    model.modifier = modifier  
    modifier.register_hook()

    # Load dataset
    data = load_data(args.text_type1, args.text_type2, args.model_name, args.num_samples)
    
    if not data:
        print("No data loaded! Please check your file paths and text types.")
        return

    random.shuffle(data)
    y_pred, y_true = [], []
    with tqdm(data) as loop:
        for item in loop:
            label = item['label']
            text = item['text']
            prompt = construct_prompt(text, experiment_type)
            print("prompt:",prompt)
            print("label:",label)
            decoded = run_generation(args.model_name, model, tokenizer, prompt, device, args.max_new_tokens)

            decoded_lower = decoded.lower()
            
            pred = -1  # Default to invalid prediction

            if '(a' in decoded_lower or 'a)' in decoded_lower or 'yes' in decoded_lower:
                pred = 1
            elif '(b' in decoded_lower or 'b)' in decoded_lower or 'no' in decoded_lower:
                pred = 0
            print("decoded:",decoded_lower)
            if pred == -1:
                continue # Skip samples with unparsable output
            if experiment_type == 'machine':
                pred = 1 - pred
            # print("label:",label)
            # print("pred:",pred)
            y_true.append(label)
            y_pred.append(pred)
            # print(y_true,y_pred)
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='macro')
            loop.set_postfix(accuracy=f"{acc:.4f}", f1=f"{f1:.4f}")

    overall_acc = accuracy_score(y_true, y_pred)
    overall_f1 = f1_score(y_true, y_pred, average='macro')

    human_idx = [i for i, y in enumerate(y_true) if y == 0]
    machine_idx = [i for i, y in enumerate(y_true) if y == 1]

    human_acc = accuracy_score([y_true[i] for i in human_idx], [y_pred[i] for i in human_idx])
    machine_acc = accuracy_score([y_true[i] for i in machine_idx], [y_pred[i] for i in machine_idx])
    human_f1 = f1_score([y_true[i] for i in human_idx], [y_pred[i] for i in human_idx], pos_label=0)
    machine_f1 = f1_score([y_true[i] for i in machine_idx], [y_pred[i] for i in machine_idx], pos_label=1)

    print("\n===== Evaluation Results =====")
    print(f"Overall Accuracy: {overall_acc:.4f}")
    print(f"Overall F1 Score: {overall_f1:.4f}")
    print(f"{args.text_type1.capitalize()} Accuracy: {human_acc:.4f}, {args.text_type1.capitalize()} F1: {human_f1:.4f}")
    print(f"{args.text_type2.capitalize()} Accuracy: {machine_acc:.4f}, {args.text_type2.capitalize()} F1: {machine_f1:.4f}")

    modifier.remove_hook()

if __name__ == '__main__':
    main()
