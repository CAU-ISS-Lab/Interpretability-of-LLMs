# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# import os

# def load_feature(path):
#     data = np.load(path)
#     return data

# def plot_tsne(ax, human_feats, machine_feats, title):
#     all_feats = np.concatenate([human_feats, machine_feats], axis=0)  # [2*samples_num, hidden_dim]
#     tsne = TSNE(n_components=2, random_state=42)
#     all_embeds = tsne.fit_transform(all_feats)  # [2*samples_num, 2]

#     n = human_feats.shape[0]
#     ax.scatter(all_embeds[:n, 0], all_embeds[:n, 1], label='Human', alpha=0.6, s=10)
#     ax.scatter(all_embeds[n:, 0], all_embeds[n:, 1], label='Machine', alpha=0.6, s=10)
#     ax.set_title(title)
#     ax.set_xticks([])
#     ax.set_yticks([])

# def main(features_dir):
#     layers = [31, 32,33, 34, 35]
#     feature_types = ['attn', 'mlp', 'final']

#     fig, axes = plt.subplots(len(layers), len(feature_types), figsize=(15, 20))
#     plt.subplots_adjust(hspace=0.4, wspace=0.3)

#     for row_idx, layer in enumerate(layers):
#         for col_idx, ftype in enumerate(feature_types):
#             file_path = os.path.join(features_dir, ftype, f"layer{layer}.npy")
#             if not os.path.exists(file_path):
#                 print(f"Missing file: {file_path}")
#                 continue
#             data = load_feature(file_path)  # shape: [2, samples_num, hidden_dim]
#             human_feats = data[0]
#             machine_feats = data[1]

#             ax = axes[row_idx, col_idx]
#             plot_tsne(ax, human_feats, machine_feats, title=f"Layer {layer} - {ftype}")

#             if row_idx == 0 and col_idx == 0:
#                 ax.legend(loc='upper right')

#     model_name=features_dir.split('/')[-1]
#     path=os.path.join('figure',model_name+'_t-sne.png')
#     plt.suptitle(f"t-SNE Visualization of Human vs Machine Features Using {model_name}")
#     plt.tight_layout(rect=[0, 0, 1, 0.96])
#     plt.savefig(path)
#     plt.show()

# if __name__ == "__main__":
#     features_dir = "./feature/qwen3-8b" 
#     main(features_dir)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
import os
import argparse
from get_KL_visual import load_jsonl_pairs,build_prompt,MODEL_PATH_MAP
from tqdm import tqdm


def load_feature(path):
    data = np.load(path)
    return data

def reduce_dim(feats, method='tsne'):
    """
    feats: np.array of shape [N, D]
    method: 'tsne', 'pca', or 'svd'
    return: [N, 2] 2D embedding
    """
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    elif method == 'pca':
        reducer = PCA(n_components=2)
    elif method == 'svd':
        reducer = TruncatedSVD(n_components=2)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return reducer.fit_transform(feats)

def plot_embedding(ax, human_feats, machine_feats, title, label_human, label_machine, method='tsne'):
    all_feats = np.concatenate([human_feats, machine_feats], axis=0)  # [2N, D]
    all_embeds = reduce_dim(all_feats, method=method)

    n = human_feats.shape[0]
    # print(all_embeds)
    ax.scatter(all_embeds[:n, 0], all_embeds[:n, 1], label=label_human, alpha=0.6, s=10)
    ax.scatter(all_embeds[n:, 0], all_embeds[n:, 1], label=label_machine, alpha=0.6, s=10)
    ax.set_title(title,fontsize=20)
    ax.set_xticks([])
    ax.set_yticks([])

def get_text_embeddings(texts, tokenizer, model, device='cuda:1'):
    import torch
    from transformers import AutoTokenizer, AutoModel
    embeddings=[]
    with torch.no_grad():
        for text in tqdm(texts):
            prompt=build_prompt(text)
            tokens = tokenizer(text, return_tensors='pt', padding=True, truncation=True).to(device)
            outputs = model(**tokens)
            embedding = outputs.last_hidden_state[:,-1,:]  # [CLS] token or first token
            embeddings.append(embedding.squeeze(0).cpu().numpy())
            # print(embedding.size(0))
    return embeddings

def plot_text_projection_tsne(json_path, human_svd_path, machine_svd_path, embedding_model_name, method='tsne', max_samples=100):
    import torch
    from transformers import AutoTokenizer, AutoModel

    print("Loading text pairs...")
    human_texts, machine_texts = load_jsonl_pairs(json_path, max_samples=max_samples,target_model='gpt4')

    print("Loading SVD bases...")
    human_basis = np.load(human_svd_path)  # shape: [k, d]
    machine_basis = np.load(machine_svd_path)

    print("Loading embedding model...")
    tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
    model = AutoModel.from_pretrained(embedding_model_name,device_map='auto').eval()

    print("Embedding texts...")
    human_emb = get_text_embeddings(human_texts, tokenizer, model)
    machine_emb = get_text_embeddings(machine_texts, tokenizer, model)

    print("Projecting onto SVD bases...")
    proj_human = human_emb @ human_basis.T  # [N, k]
    proj_machine = machine_emb @ machine_basis.T

    print("Running t-SNE...")
    all_proj = np.concatenate([proj_human, proj_machine], axis=0)
    all_embeds = reduce_dim(all_proj, method=method)
    n = proj_human.shape[0]

    print("Plotting...")
    plt.figure(figsize=(8, 6))
    plt.scatter(all_embeds[:n, 0], all_embeds[:n, 1], label='Human', alpha=0.6, s=20)
    plt.scatter(all_embeds[n:, 0], all_embeds[n:, 1], label='Machine', alpha=0.6, s=20)
    plt.legend()
    plt.title(f"{method.upper()} of Text Embeddings Projected to SVD Bases")
    plt.xticks([])
    plt.yticks([])
    os.makedirs('figure', exist_ok=True)
    plt.savefig(f'figure/qwen3-8b_text_projected_{method}_gpt4.png')
    plt.show()

def main(features_base_dir, method='tsne'):
    types = ['self_vs_human', 'self_vs_chatgpt', 'chatgpt_vs_human']
    modes = ['', '_IPP']  # sample in first row, qa in second row
    # modes=['_probs']

    fig, axes = plt.subplots(len(modes), len(types), figsize=(18, 10))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    for col_idx, data_type in enumerate(types):
        for row_idx, mode in enumerate(modes):
            features_dir = os.path.join(features_base_dir, f"{data_type}{mode}", 'final_feats')
            # features_dir = os.path.join(features_base_dir, f"{data_type}", f'final{mode}')
# 
            if not os.path.exists(features_dir):
                print(f"Directory does not exist: {features_dir}")
                continue

            all_files = sorted([f for f in os.listdir(features_dir) if f.endswith('.npy')])
            if not all_files:
                print(f"No .npy files in {features_dir}")
                continue
            last_layer_file = all_files[-1]
            file_path = os.path.join(features_dir, last_layer_file)

            if not os.path.exists(file_path):
                print(f"Missing file: {file_path}")
                continue

            data = load_feature(file_path)  # shape: [2, N, D]
            # print(data)
            feats_1 = data[0]
            feats_2 = data[1]

            ax = axes[row_idx,col_idx]
            title = f"{data_type}{mode}"
            parts = data_type.split('_')
            label_1 = parts[0]
            label_2 = parts[2]

            plot_embedding(ax, feats_1, feats_2, title=title,
                label_human=label_1, label_machine=label_2, method=method)
            label = chr(ord('a') + row_idx*len(types)+col_idx)  
            ax.legend(loc='upper right', fontsize=18) 
            # 添加子图标注：(a), (b), ...
            ax.text(0.5, -0.05, f"({label})", fontsize=20, ha='center', va='center', transform=ax.transAxes)

    model_name = os.path.basename(features_base_dir.rstrip('/'))
    out_path = os.path.join('figures', f"{model_name}_compare_{method}.png")
    os.makedirs('figures', exist_ok=True)
    # plt.suptitle(f"{method.upper()} Visualization of Final Layer Features")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_dir', type=str, default="./feature/deepseek")
    parser.add_argument('--method', type=str, choices=['tsne', 'pca', 'svd'], default='tsne')
    parser.add_argument('--json_path', type=str,  default="data/train.csv")  # NEW
    parser.add_argument('--human_svd', type=str, default=None)
    parser.add_argument('--machine_svd', type=str, default=None)  
    # parser.add_argument('--human_svd', type=str, default='./hidden_stats/qwen3-8b/chatgpt/human_svd_basis.npy')  # NEW
    # parser.add_argument('--machine_svd', type=str, default='./hidden_stats/qwen3-8b/chatgpt/machine_svd_basis.npy')  # NEW
    parser.add_argument('--embedding_model', type=str, default='../../LLMs/QWEN/qwen3-8b')  # NEW
    parser.add_argument('--max_samples', type=int, default=200)  # NEW
    args = parser.parse_args()

    if args.json_path and args.human_svd and args.machine_svd:
        plot_text_projection_tsne(
            json_path=args.json_path,
            human_svd_path=args.human_svd,
            machine_svd_path=args.machine_svd,
            embedding_model_name=args.embedding_model,
            method=args.method,
            max_samples=args.max_samples,
        )
    else:
        main(args.features_dir, method=args.method)
