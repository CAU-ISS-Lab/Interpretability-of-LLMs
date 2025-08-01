import argparse
import json
import os
from typing import List, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import rbf_kernel
import torch.nn.functional as F
from get_KL_visual import load_jsonl_pairs, build_prompt
from sklearn.decomposition import PCA

# Add model path mapping like in get_KL_visual.py
MODEL_PATH_MAP = {
    'qwen3-8b': '../../LLMs/QWEN/qwen3-8b',
    'llama3.1-8b': '../../LLMs/LLAMA/llama3.1-8b',
    'mistral-7b': '../../LLMs/mistral/mistral-7b',
    'deepseek-8b':'../../LLMs/deepseek/DeepSeek-R1-0528-Qwen3-8B'
}

# Add text type configuration
TEXT_TYPE_CONFIG = {
    'human': {'label': 0, 'prefix': 'human'},
    'chatgpt': {'label': 1, 'prefix': 'chatgpt'},
    'qwen_answers': {'label': 1, 'prefix': 'qwen_answers'},
    'llama_answers': {'label': 1, 'prefix': 'llama_answers'},
    'deepseek_answers': {'label': 1, 'prefix': 'deepseek_answers'},
    'gpt4': {'label': 1, 'prefix': 'gpt4'},
    'llama-chat': {'label': 1, 'prefix': 'llama-chat'},
    'mistral': {'label': 1, 'prefix': 'mistral'}
}

def get_file_paths(text_type1, text_type2, model_name):
    """Get correct data file paths, now supports new train_all.jsonl format"""
    experiment_type = 'human' if text_type1 == 'human' or text_type2 == 'human' else 'machine'
    
    # Check if either text_type uses the new _answers format
    new_answer_types = {'qwen_answers', 'llama_answers', 'deepseek_answers'}
    if text_type1 in new_answer_types or text_type2 in new_answer_types:
        # Use the new combined all files for machine experiments
        return "data/train_all.jsonl"
    
    # Human experiments use specific combination data files
    if experiment_type == 'human':
        if (text_type1 == 'human' and text_type2 == 'chatgpt') or (text_type1 == 'chatgpt' and text_type2 == 'human'):
            return "data/train_all.jsonl"
        else:
            # Other human experiment combinations use CSV
            return "data/train.csv"
    else:
        # Machine experiments use train_all.jsonl
        return "data/train_all.jsonl"


def compute_grassmann_distance(U: np.ndarray, V: np.ndarray) -> float:
    """
    Compute Grassmannian distance between two subspaces.
    U, V: shape (d, k)
    """
    M = U.T @ V
    _, s, _ = np.linalg.svd(M)
    principal_angles = np.arccos(np.clip(s, -1.0, 1.0))
    return np.linalg.norm(principal_angles)


def compute_projection_frobenius_distance(U: np.ndarray, V: np.ndarray) -> float:
    """
    Compute Frobenius norm between two projection matrices.
    U, V: shape (d, k)
    """
    PU = U @ U.T
    PV = V @ V.T
    return np.linalg.norm(PU - PV, ord='fro')

def compute_mmd(x: torch.Tensor, y: torch.Tensor, gamma: float = 1.0) -> float:
    x_np, y_np = x.numpy(), y.numpy()
    xx = rbf_kernel(x_np, x_np, gamma=gamma)
    yy = rbf_kernel(y_np, y_np, gamma=gamma)
    xy = rbf_kernel(x_np, y_np, gamma=gamma)
    return xx.mean() + yy.mean() - 2 * xy.mean()

def compute_linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)

    x_xt = torch.matmul(x, x.T)
    y_yt = torch.matmul(y, y.T)

    dot_product_similarity = (x_xt * y_yt).sum()
    normalization_x = (x_xt * x_xt).sum()
    normalization_y = (y_yt * y_yt).sum()

    return dot_product_similarity / (normalization_x.sqrt() * normalization_y.sqrt())

def compute_mean_cosine_similarity(x: torch.Tensor, y: torch.Tensor) -> float:
    x_mean = x.mean(dim=0)
    y_mean = y.mean(dim=0)
    return F.cosine_similarity(x_mean, y_mean, dim=0).item()



# def load_jsonl_data(filepath: str) -> Tuple[List[str], List[str]]:
#     human_answers, chatgpt_answers = [], []
#     with open(filepath, 'r', encoding='utf-8') as f:
#         for line in f:
#             data = json.loads(line)
#             human_answers.extend(data.get("human_answers", []))
#             chatgpt_answers.extend(data.get("chatgpt_answers", []))
#     return human_answers, chatgpt_answers


def get_last_hidden_states(model, tokenizer, texts: List[str], experiment_type: str = 'human') -> torch.Tensor:
    model.eval()
    all_embeddings = []
    device=model.device

    with torch.no_grad():
        for text in tqdm(texts, desc="Encoding texts"):
            prompt = build_prompt(text, experiment_type)
            # prompt = text
            inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=1024).to(device)
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
            last_token_embedding = hidden_states[:, -1, :]
            all_embeddings.append(last_token_embedding.squeeze(0).cpu())

    return torch.stack(all_embeddings)

def remove_projection(embeddings: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """
    Remove the projection of embeddings onto the given direction vector.
    embeddings: (n, d)
    direction: (1, d) or (d,)
    Returns: (n, d)
    """
    direction = F.normalize(direction, p=2, dim=-1)  # 单位化 u
    proj = torch.matmul(embeddings, direction.T)  # shape: (n, 1)
    return embeddings - proj * direction  # (n, d) - (n, 1) * (1, d) = (n, d)


def compute_statistics(embeddings: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    emb_np = embeddings.numpy()
    mean = np.mean(emb_np, axis=0)
    std = np.std(emb_np, axis=0)
    return mean, std


def compute_svd_basis(embeddings: torch.Tensor, k: int) -> np.ndarray:
    svd = TruncatedSVD(n_components=k, random_state=42)
    svd.fit(embeddings.numpy())
    return svd.components_  # shape: [k, hidden_dim]

def compute_pca_basis(embeddings: torch.Tensor, k: int) -> np.ndarray:
    pca = PCA(n_components=k, random_state=42)
    pca.fit(embeddings.numpy())
    return pca.components_  # shape: [k, hidden_dim]


def save_statistics(mean: np.ndarray, std: np.ndarray, save_dir: str, prefix: str):
    np.save(os.path.join(save_dir, f"{prefix}_mean.npy"), mean)
    np.save(os.path.join(save_dir, f"{prefix}_std.npy"), std)


def save_svd_basis(basis: np.ndarray, save_dir: str, prefix: str):
    np.save(os.path.join(save_dir, f"{prefix}_svd_basis.npy"), basis)


def main(args):
    # Get model path from mapping
    model_path = MODEL_PATH_MAP.get(args.model_name)
    if not model_path:
        raise ValueError(f"Model name '{args.model_name}' not found. Please add it to MODEL_PATH_MAP.")
    
    # Determine experiment type and data path
    experiment_type = 'human' if args.text_type1 == 'human' or args.text_type2 == 'human' else 'machine'
    data_path = get_file_paths(args.text_type1, args.text_type2, args.model_name)
    
    print(f"Experiment type: {experiment_type}")
    print(f"Data path: {data_path}")
    
    # Load data with new format support
    if '.jsonl' in data_path:
        texts1, texts2 = load_jsonl_pairs(data_path, max_samples=200, 
                                     text_type1=args.text_type1, text_type2=args.text_type2)
        comparison_name = f"{args.text_type1}_vs_{args.text_type2}"
    else:
        # Use unified logic for CSV files
        texts1, texts2 = load_jsonl_pairs(data_path, max_samples=200, 
                                     text_type1=args.text_type1, text_type2=args.text_type2)
        comparison_name = f"{args.text_type1}_vs_{args.text_type2}"

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map='auto')

    # Get embeddings with experiment-specific prompts
    print(f"Processing {args.text_type1} texts...")
    human_embeddings = get_last_hidden_states(model, tokenizer, texts1, experiment_type)
    print(f"Processing {args.text_type2} texts...")
    machine_embeddings = get_last_hidden_states(model, tokenizer, texts2, experiment_type)

    # For machine experiments, the roles are swapped to align with prompt logic
    # 'self' (text_type1) -> 'Yes' answer -> 'machine' subspace
    # 'other' (text_type2) -> 'No' answer -> 'human' subspace
    if experiment_type == 'machine':
        print("\n[INFO] Machine experiment type detected. Swapping roles for subspace construction.")
        print(f"[INFO]   - 'machine' subspace (for 'Yes' answer) will be from: {args.text_type1}")
        print(f"[INFO]   - 'human' subspace (for 'No' answer) will be from: {args.text_type2}")
        human_embeddings, machine_embeddings = machine_embeddings, human_embeddings

    # Compute stats
    human_mean, human_std = compute_statistics(human_embeddings)
    machine_mean, machine_std = compute_statistics(machine_embeddings)

    # Save results with updated directory structure
    save_dir = os.path.join(args.save_dir, args.model_name, comparison_name, f"svd_rank_{args.svd_rank}")
    os.makedirs(save_dir, exist_ok=True)
    save_statistics(human_mean, human_std, save_dir, "human")
    save_statistics(machine_mean, machine_std, save_dir, "machine")

    if args.use_pca:
        print("[INFO] Using PCA to compute subspaces")
        human_svd_basis = compute_pca_basis(human_embeddings, args.svd_rank)
        machine_svd_basis = compute_pca_basis(machine_embeddings, args.svd_rank)
    else:
        print("[INFO] Using SVD to compute subspaces")
        human_svd_basis = compute_svd_basis(human_embeddings, args.svd_rank)
        machine_svd_basis = compute_svd_basis(machine_embeddings, args.svd_rank)

    save_svd_basis(human_svd_basis, save_dir, "human")
    save_svd_basis(machine_svd_basis, save_dir, "machine")

    # print(f"All statistics and SVD bases saved to: {save_dir}")
    human_basis = human_svd_basis.T
    machine_basis = machine_svd_basis.T

    grassmann_dist = compute_grassmann_distance(human_basis, machine_basis)
    frob_dist = compute_projection_frobenius_distance(human_basis, machine_basis)

    max_proj_frobenius = np.sqrt(2 * args.svd_rank)
    max_grassmann = np.sqrt(args.svd_rank) * (np.pi / 2)  # Maximum when all principal angles are π/2

    print(f"Grassmannian distance: {grassmann_dist:.4f} (max: {max_grassmann:.4f})")
    print(f"Projection Frobenius distance: {frob_dist:.4f} (max: {max_proj_frobenius:.4f})")

    print(f"Normalized Grassmannian distance: {grassmann_dist / max_grassmann:.4f}")
    print(f"Normalized Frobenius distance: {frob_dist / max_proj_frobenius:.4f}")

    print("\n==== Computing distributional similarity metrics ====")

    for sigma in [0.5, 1, 2, 5, 10]:
        gamma = 1.0 / (2 * sigma ** 2)
        score = compute_mmd(human_embeddings, machine_embeddings, gamma=gamma)
        print(f"MMD with sigma={sigma}: {score:.4f}")
    mmd_value = compute_mmd(human_embeddings, machine_embeddings, gamma=1.0)
    cka_value = compute_linear_cka(human_embeddings, machine_embeddings)
    cosine_value = compute_mean_cosine_similarity(human_embeddings, machine_embeddings)

    # print(f"MMD (RBF kernel, γ=1.0): {mmd_value:.4f}")
    print(f"Linear CKA: {cka_value:.4f}")
    print(f"Mean Embedding Cosine Similarity: {cosine_value:.4f}")

    if args.compare_dir_1 and args.compare_dir_2:
        print("\n==== Comparing machine SVD bases from two folders ====")
        basis_path_1 = os.path.join(args.compare_dir_1, "machine_svd_basis.npy")
        basis_path_2 = os.path.join(args.compare_dir_2, "machine_svd_basis.npy")

        if not os.path.exists(basis_path_1) or not os.path.exists(basis_path_2):
            print("Error: One of the machine_svd_basis.npy files not found.")
            return

        basis1 = np.load(basis_path_1).T  # shape: (d, k)
        basis2 = np.load(basis_path_2).T  # shape: (d, k)

        gd = compute_grassmann_distance(basis1, basis2)
        fd = compute_projection_frobenius_distance(basis1, basis2)

        max_fd = np.sqrt(2 * basis1.shape[1])
        max_gd = np.sqrt(basis1.shape[1]) * (np.pi / 2)

        print(f"Cross-model Grassmannian distance: {gd:.4f} (max: {max_gd:.4f})")
        print(f"Cross-model Projection Frobenius distance: {fd:.4f} (max: {max_fd:.4f})")

        print(f"Normalized GD: {gd / max_gd:.4f}")
        print(f"Normalized FD: {fd / max_fd:.4f}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract hidden state statistics and SVD subspaces")
    parser.add_argument("--model_name", type=str, default="qwen3-8b", choices=MODEL_PATH_MAP.keys(),
                       help="Short name of the model to use")
    parser.add_argument("--save_dir", type=str, default="./hidden_stats")
    parser.add_argument("--text_type1", type=str, default='human', 
                       choices=['human', 'chatgpt', 'qwen_answers', 'llama_answers', 'deepseek_answers', 'gpt4', 'llama-chat', 'mistral'], 
                       help='First text type to compare')
    parser.add_argument("--text_type2", type=str, default='chatgpt', 
                       choices=['human', 'chatgpt', 'qwen_answers', 'llama_answers', 'deepseek_answers', 'gpt4', 'llama-chat', 'mistral'],
                       help='Second text type to compare')
    parser.add_argument("--svd_rank", type=int, default=64, help="Number of SVD basis vectors to keep")
    parser.add_argument("--compare_dir_1", type=str, default='./hidden_stats/qwen3-8b/chatgpt', help="First directory to compare machine_svd_basis.npy")
    parser.add_argument("--compare_dir_2", type=str, default='./hidden_stats/qwen3-8b/mistral', help="Second directory to compare machine_svd_basis.npy")
    parser.add_argument("--use_pca", action='store_true', help="Use PCA instead of SVD to compute subspace")
    args = parser.parse_args()
    main(args)
