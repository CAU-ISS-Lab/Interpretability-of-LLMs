import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
import numpy as np
import os
import csv

MODEL_PATH_MAP = {
    'qwen3-8b': '../../LLMs/QWEN/qwen3-8b',
    'llama3.1-8b': '../../LLMs/LLAMA/llama3.1-8b',
    'mistral-7b': '../../LLMs/mistral/mistral-7b',
    'deepseek':'../../LLMs/deepseek/DeepSeek-R1-0528-Qwen3-8B'
}

def last_token_pool(hidden_states, attention_mask):
    sequence_lengths = attention_mask.sum(dim=1) - 1
    sequence_lengths = sequence_lengths.to(hidden_states.device)
    return hidden_states[torch.arange(hidden_states.size(0)), sequence_lengths]

def build_prompt(text: str,mode) -> str:
    if mode=='human':
        return f"{text}\nDoes this text written by machine?\n(a) Yes.\n(b) No.\n"
    else:
        return f"{text}\nWas this text written by yourself?\n(a) Yes.\n(b) No.\n"

@torch.no_grad()
def get_all_probs_and_features(index,model, tokenizer, texts, max_length=512, batch_size=8):
    mode='human'
    if index>=4:
        mode='machine'
    if index%2!=0:
        prompted_texts = [build_prompt(t,mode) for t in texts]
    else:
        prompted_texts = texts
    
    inputs_ref = tokenizer("hello", return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    inputs_ref = {k: v.to(model.device) for k, v in inputs_ref.items()}
    outputs_ref = model(**inputs_ref, output_hidden_states=True)
    num_layers = len(outputs_ref.hidden_states) - 1

    full_output_probs = [[] for _ in range(num_layers)]
    attn_output_probs = [[] for _ in range(num_layers)]
    mlp_output_probs = [[] for _ in range(num_layers)]

    attn_outputs = [None] * num_layers
    mlp_outputs = [None] * num_layers

    # 用于保存原始隐藏状态（last_token_pool前的）
    full_hidden_states_all = [[] for _ in range(num_layers)]
    attn_hidden_states_all = [[] for _ in range(num_layers)]
    mlp_hidden_states_all = [[] for _ in range(num_layers)]

    hook_handles = []

    def save_attn_output(layer_idx):
        def hook(module, input, output):
            attn_outputs[layer_idx] = output[0]
        return hook

    def save_mlp_output(layer_idx):
        def hook(module, input, output):
            mlp_outputs[layer_idx] = output
        return hook

    for i in range(num_layers):
        try:
            attn_layer = model.model.layers[i].self_attn
            mlp_layer = model.model.layers[i].mlp
        except AttributeError:
            attn_layer = model.transformer.h[i].attn
            mlp_layer = model.transformer.h[i].mlp

        hook_handles.append(attn_layer.register_forward_hook(save_attn_output(i)))
        hook_handles.append(mlp_layer.register_forward_hook(save_mlp_output(i)))

    for i in tqdm(range(0, len(prompted_texts), batch_size), desc="Batching"):
        batch_texts = prompted_texts[i:i+batch_size]
        tokenized = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
        tokenized = {k: v.to(model.device) for k, v in tokenized.items()}
        outputs = model(**tokenized, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        for layer_idx in range(1, num_layers + 1):
            mask = tokenized['attention_mask']

            # full output
            full_hs = hidden_states[layer_idx]  # [B, seq_len, hidden_dim]
            pooled_full = last_token_pool(full_hs, mask)  # [B, hidden_dim]
            logits = model.lm_head(pooled_full)
            full_output_probs[layer_idx - 1].append(torch.softmax(logits, dim=-1).cpu())
            full_hidden_states_all[layer_idx - 1].append(pooled_full.cpu().numpy())

            # attn output
            attn = attn_outputs[layer_idx - 1]
            if attn is not None:
                pooled_attn = last_token_pool(attn, mask)
                logits = model.lm_head(pooled_attn)
                attn_output_probs[layer_idx - 1].append(torch.softmax(logits, dim=-1).cpu())
                attn_hidden_states_all[layer_idx - 1].append(pooled_attn.cpu().numpy())

            # mlp output
            mlp = mlp_outputs[layer_idx - 1]
            if mlp is not None:
                pooled_mlp = last_token_pool(mlp, mask)
                logits = model.lm_head(pooled_mlp)
                mlp_output_probs[layer_idx - 1].append(torch.softmax(logits, dim=-1).cpu())
                mlp_hidden_states_all[layer_idx - 1].append(pooled_mlp.cpu().numpy())

    for h in hook_handles:
        h.remove()

    # print(full_output_probs.shape)
    full_output_probs = [torch.cat(p, dim=0) for p in full_output_probs]
    attn_output_probs = [torch.cat(p, dim=0) for p in attn_output_probs]
    mlp_output_probs = [torch.cat(p, dim=0) for p in mlp_output_probs]
    # print(full_output_probs.shape)

    full_hidden_states_all = [np.concatenate(p, axis=0) for p in full_hidden_states_all]
    attn_hidden_states_all = [np.concatenate(p, axis=0) for p in attn_hidden_states_all]
    mlp_hidden_states_all = [np.concatenate(p, axis=0) for p in mlp_hidden_states_all]

    return full_output_probs, attn_output_probs, mlp_output_probs, full_hidden_states_all, attn_hidden_states_all, mlp_hidden_states_all


def save_combined_features(save_dir, layer_start, layer_end, human_feats_list, machine_feats_list, subfolder):
    folder = os.path.join(save_dir, subfolder)
    os.makedirs(folder, exist_ok=True)

    for layer_idx in range(layer_start, layer_end + 1):
        # human_feats = np.concatenate(human_feats_list[layer_idx - 1], axis=0)  # [samples_num, hidden_dim]
        # machine_feats = np.concatenate(machine_feats_list[layer_idx - 1], axis=0)  # [samples_num, hidden_dim]

        combined = np.stack([human_feats_list[layer_idx - 1],machine_feats_list[layer_idx - 1]], axis=0)  # [2, samples_num, hidden_dim]
        print(combined.shape)

        save_path = os.path.join(folder, f"layer{layer_idx}.npy")
        np.save(save_path, combined)
        print(f"Saved combined features of layer {layer_idx} to {save_path}")


def compute_kl_divergence(p_probs, q_probs):
    kl_values = []
    for p, q in zip(p_probs, q_probs):
        p_mean = p.mean(dim=0) + 1e-8
        p_mean = p_mean / p_mean.sum()
        q_mean = q.mean(dim=0) + 1e-8
        q_mean = q_mean / q_mean.sum()
        kl = F.kl_div(p_mean.log(), q_mean, reduction='sum').item()
        kl_values.append(kl)
    return kl_values

def compute_js_divergence(p_probs, q_probs):
    js_values = []
    for p, q in zip(p_probs, q_probs):
        # print(p.shape)
        p_mean = p.mean(dim=0) + 1e-8
        p_mean = p_mean / p_mean.sum()
        q_mean = q.mean(dim=0) + 1e-8
        q_mean = q_mean / q_mean.sum()
        # print(q_mean.sum())
        m = 0.5 * (p_mean + q_mean)
        kl_pm = F.kl_div(m.log(), p_mean, reduction='sum')  # input=log(M), target=P
        kl_qm = F.kl_div(m.log(), q_mean, reduction='sum')
        js = 0.5 * (kl_pm + kl_qm)
        js_values.append(js.item())
    return js_values

def load_jsonl_pairs(file_path, max_samples=100, text_type1='human', text_type2='chatgpt'):
    """
    Load text pairs from jsonl or csv file for comparison.
    For jsonl: supports human_answers, chatgpt_answers, self_answers
    For csv: uses existing logic with target_model parameter
    """
    texts1, texts2 = [], []
    
    if '.jsonl' in file_path:
        # Map text_type to corresponding field names
        field_mapping = {
            'human': 'human_answers',
            'chatgpt': 'chatgpt_answers',
            'qwen_answers': 'qwen_answers',
            'llama_answers': 'llama_answers',
            'deepseek_answers': 'deepseek_answers'
        }
        
        field1 = field_mapping.get(text_type1)
        field2 = field_mapping.get(text_type2)
        
        if not field1 or not field2:
            raise ValueError(f"Unsupported text types: {text_type1}, {text_type2}. Must be from: {list(field_mapping.keys())}")
            
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                
                # Extract texts from the specified fields
                if field1 in item and field2 in item:
                    # Handle list format (take first element if it's a list)
                    text1 = item[field1][0] if isinstance(item[field1], list) else item[field1]
                    text2 = item[field2][0] if isinstance(item[field2], list) else item[field2]
                    
                    if text1 and text2:  # Ensure both texts are non-empty
                        texts1.append(text1)
                        texts2.append(text2)
                        
                if len(texts1) >= max_samples and max_samples != -1:
                    break
                    
    elif '.csv' in file_path:
        # Keep existing CSV logic for backward compatibility
        target_model = text_type2  # Use text_type2 as target_model for CSV
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row.get("generation", "").strip()
                model = row.get("model", "").strip().lower()
                if not text:
                    continue
                if model != "human":
                    if target_model and model != target_model:
                        continue
                    texts2.append(text)
                else:
                    texts1.append(text)
    else:
        raise ValueError(f"Unsupported file format. Must be .jsonl or .csv")
    
    if max_samples == -1:
        return texts1, texts2
    else:
        return texts1[:max_samples], texts2[:max_samples]

def get_model_layers(model):
    # for name, module in model.named_modules():
    #     print(name) 
    total_layers = len(model.model.layers)
    print(f"Total layers (top-level): {total_layers}")
    return total_layers

def calculate_snr(human_feats, machine_feats):
    if torch.is_tensor(human_feats):
        human_feats = human_feats.cpu().numpy()
    if torch.is_tensor(machine_feats):
        machine_feats = machine_feats.cpu().numpy()
    
    mean_diff = np.mean(human_feats, axis=0) - np.mean(machine_feats, axis=0)
    signal = np.sum(mean_diff ** 2)  

    noise_h = np.trace(np.cov(human_feats.T)) 
    noise_m = np.trace(np.cov(machine_feats.T))
    noise = noise_h + noise_m
    
    return signal / (noise + 1e-8)


def main(args):
    model_path = MODEL_PATH_MAP.get(args.model_name)
    if not model_path:
        raise ValueError(f"Model name '{args.model_name}' not found. Please add it to MODEL_PATH_MAP.")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map='auto')
    model.eval()

    total_layers = get_model_layers(model)
    last_layer = total_layers - 1
    start_layer = total_layers - 5

    pairs = [
        ("self", "human"),
        ("self", "human"),
        ("self", "chatgpt"),
        ("self", "chatgpt"),
        ("chatgpt", "human"),
        ("chatgpt", "human"),
    ]

    all_js_results = {}
    SNR_res={}
    index=0
    for text_type1, text_type2 in pairs:
        comparison_name = f"{text_type1}_vs_{text_type2}"
        if index%2!=0:
            comparison_name=comparison_name+'_IPP'
        print(f"\n== Processing {comparison_name} ==")

        texts1, texts2 = load_jsonl_pairs(
            args.data_path, max_samples=args.num_samples,
            text_type1=text_type1, text_type2=text_type2
        )

      
        print(f"Extracting {text_type1} probabilities...")
        h_full_probs, h_attn_probs, h_mlp_probs, h_full_states, h_attn_states, h_mlp_states = get_all_probs_and_features(
           index,model, tokenizer, texts1, args.max_length, args.batch_size
        )

        print(f"Extracting {text_type2} probabilities...")
        m_full_probs, m_attn_probs, m_mlp_probs, m_full_states, m_attn_states, m_mlp_states = get_all_probs_and_features(index,
            model, tokenizer, texts2, args.max_length, args.batch_size
        )

        feature_save_dir=os.path.join(args.save_dir,args.model_name,comparison_name)
        save_combined_features(feature_save_dir,start_layer,last_layer,h_full_states,m_full_states,'final_feats')
        save_combined_features(feature_save_dir,start_layer,last_layer,h_attn_states,m_attn_states,'attn_feats')
        save_combined_features(feature_save_dir,start_layer,last_layer,h_mlp_states,m_mlp_states,'mlp_feats')

        save_combined_features(feature_save_dir,start_layer,last_layer,h_full_probs,m_full_probs,'final_probs')
        save_combined_features(feature_save_dir,start_layer,last_layer,h_attn_probs,m_attn_probs,'attn_probs')
        save_combined_features(feature_save_dir,start_layer,last_layer,h_mlp_probs,m_mlp_probs,'mlp_probs')

        js_full = compute_js_divergence(h_full_probs, m_full_probs)
        js_attn = compute_js_divergence(h_attn_probs, m_attn_probs)
        js_mlp = compute_js_divergence(h_mlp_probs, m_mlp_probs)

        # SNR_prob=calculate_snr(h_full_probs[-1],m_full_probs[-1])
        # SNR_feature=calculate_snr(h_full_states[-1],m_full_states[-1])

        all_js_results[comparison_name] = {
            "js_full": js_full,
            "js_attn": js_attn,
            "js_mlp": js_mlp,
        }
        # SNR_res[comparison_name]={
        #     'prob':SNR_prob,
        #     'feature':SNR_feature
        # }
        index+=1

    print(SNR_res)
    # Plot all in 1x3 subplots
    plt.figure(figsize=(18, 9))
    for idx, (comparison_name, js_data) in enumerate(all_js_results.items()):
        plt.subplot(3, 2, idx+1)
        x = list(range(len(js_data["js_full"])))
        plt.plot(x, js_data["js_full"], label='JS @ Full', marker='o')
        plt.plot(x, js_data["js_attn"], label='JS @ Attention', marker='x')
        plt.plot(x, js_data["js_mlp"], label='JS @ MLP', marker='s')
        label = chr(ord('a') + idx)  # 通过索引生成序号字符，如 'a', 'b', 'c'...
        plt.text(0.5, -0.42, f'({label})', transform=plt.gca().transAxes, fontsize=19)
        plt.title(comparison_name,fontsize=17)
        plt.ylim(0, 0.7)
        plt.xlabel("Layer Index",fontsize=14)
        plt.ylabel("JS Divergence",fontsize=14)
        plt.grid(True)
        plt.legend(fontsize=13,loc='upper left')

    plt.tight_layout()
    figure_path = os.path.join('figures', args.model_name + '_' + args.output_path)
    os.makedirs('figures', exist_ok=True)
    plt.savefig(figure_path)
    plt.show()
    print(f"\n✅ JS divergence plot saved to: {figure_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='deepseek', choices=['qwen3-8b', 'llama3.1-8b', 'mistral-7b','deepseek'],
                       help='Short name of the model to use.')
    parser.add_argument('--data_path', type=str, default='data/train_all.jsonl')
    parser.add_argument('--target_model', type=str, default='mistral', help='For CSV files: target model name')
    parser.add_argument('--text_type1', type=str, default='human', choices=['human', 'chatgpt', 'self'], 
                       help='For JSONL files: first text type to compare')
    parser.add_argument('--text_type2', type=str, default='chatgpt', choices=['human', 'chatgpt', 'self'],
                       help='For JSONL files: second text type to compare')
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--num_samples', type=int, default=400)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--output_path', type=str, default='js_qa.png')
    parser.add_argument('--save_dir', type=str, default='./feature')
    args = parser.parse_args()
    main(args)
