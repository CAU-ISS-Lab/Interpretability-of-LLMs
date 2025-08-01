import argparse
import json
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import accuracy_score, f1_score
import csv
import os 

# Add model path mapping
MODEL_PATH_MAP = {
    'qwen3-8b': './model/Qwen3-8B',
    'llama3.1-8b': './model/Llama-3.1-8B',
    'deepseek-8b': './model/DeepSeek-R1-0528-Qwen3-8B',
}

# Define text type to file mapping
TEXT_TYPE_CONFIG = {
    'human': {
        'sources': ['jsonl', 'csv'],
        'jsonl_field': 'human_answers',
        'csv_model': 'human'
    },
    'chatgpt': {
        'sources': ['jsonl'],
        'jsonl_field': 'chatgpt_answers',
        'csv_model': None
    },
    'qwen_answers': {
        'sources': ['jsonl'],
        'jsonl_field': 'qwen_answers',
        'csv_model': None
    },
    'llama_answers': {
        'sources': ['jsonl'],
        'jsonl_field': 'llama_answers',
        'csv_model': None
    },
    'deepseek_answers': {
        'sources': ['jsonl'],
        'jsonl_field': 'deepseek_answers',
        'csv_model': None
    },
    'gpt4': {
        'sources': ['csv'],
        'jsonl_field': None,
        'csv_model': 'gpt4'
    },
    'llama-chat': {
        'sources': ['csv'],
        'jsonl_field': None,
        'csv_model': 'llama-chat'
    },
    'mistral': {
        'sources': ['csv'],
        'jsonl_field': None,
        'csv_model': 'mistral'
    }
}

def get_file_paths(text_type1, text_type2, model_name):
    """
    Determine which files need to be loaded based on text types.
    Machine experiments use test_all.jsonl, human experiments use test.csv.
    """
    experiment_type = 'human' if text_type1 == 'human' or text_type2 == 'human' else 'machine'
    
    # Check if either text_type uses the new _answers format
    new_answer_types = {'qwen_answers', 'llama_answers', 'deepseek_answers'}
    if text_type1 in new_answer_types or text_type2 in new_answer_types:
        # Use the new combined all files for machine experiments
        return {text_type1: "data/test_all.jsonl", text_type2: "data/test_all.jsonl"}

    if experiment_type == 'human':
        # Human experiments use simple csv file
        if {text_type1, text_type2} in [{'human', 'chatgpt'}]:
            # Use jsonl file for human vs chatgpt
            primary_file = "data/test_all.jsonl"
        else:
            # Use csv file for other human experiments
            primary_file = "data/test.csv"
        
        return {text_type1: primary_file, text_type2: primary_file}
    
    else:
        # Machine experiments use test_all.jsonl
        return {text_type1: "data/test_all.jsonl", text_type2: "data/test_all.jsonl"}

def load_texts_from_jsonl(file_path, text_type, num_samples):
    """Load texts from a JSONL file for a specific text type."""
    texts = []
    config = TEXT_TYPE_CONFIG[text_type]
    field_name = config['jsonl_field']
    
    if not os.path.exists(file_path):
        return texts
        
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if len(texts) >= num_samples:
                break
            item = json.loads(line.strip())
            
            if field_name in item:
                text = item[field_name]
                # Handle list format (take first element if it's a list)
                if isinstance(text, list):
                    text = text[0] if text else ""
                
                if text and text.strip():  # Ensure text is non-empty
                    texts.append(text.strip())
    
    return texts

def load_texts_from_csv(file_path, text_type, num_samples):
    """Load texts from a CSV file for a specific text type."""
    texts = []
    config = TEXT_TYPE_CONFIG[text_type]
    target_model = config['csv_model']
    
    if not os.path.exists(file_path):
        return texts
        
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if len(texts) >= num_samples:
                break
            
            text = row.get("generation", "").strip()
            model = row.get("model", "").strip().lower()
            
            if not text:
                continue
            
            if model == target_model:
                texts.append(text)
    
    return texts

def load_data(text_type1, text_type2, model_name, num_samples=100):
    """
    Load texts for two text types and return list of dicts.
    Each dict contains 'text' and 'label' (0 for text_type1, 1 for text_type2).
    """
    # Get file paths for each text type
    file_paths = get_file_paths(text_type1, text_type2, model_name)
    
    # Load texts for each type
    texts1 = []
    texts2 = []
    
    # Load text_type1
    if text_type1 in file_paths:
        file_path = file_paths[text_type1]
        config = TEXT_TYPE_CONFIG[text_type1]
        
        if 'jsonl' in config['sources'] and file_path.endswith('.jsonl'):
            texts1 = load_texts_from_jsonl(file_path, text_type1, num_samples)
        elif 'csv' in config['sources'] and file_path.endswith('.csv'):
            texts1 = load_texts_from_csv(file_path, text_type1, num_samples)
    
    # Load text_type2
    if text_type2 in file_paths:
        file_path = file_paths[text_type2]
        config = TEXT_TYPE_CONFIG[text_type2]
        
        if 'jsonl' in config['sources'] and file_path.endswith('.jsonl'):
            texts2 = load_texts_from_jsonl(file_path, text_type2, num_samples)
        elif 'csv' in config['sources'] and file_path.endswith('.csv'):
            texts2 = load_texts_from_csv(file_path, text_type2, num_samples)
    
    # Combine into dataset format
    data = []
    for text in texts1:
        data.append({'text': text, 'label': 0})
    for text in texts2:
        data.append({'text': text, 'label': 1})
    
    return data

def construct_prompt(text, experiment_type='human'):
    """
    Create prompt from input text.
    Different prompts for human vs machine experiments.
    """
    clear_text = text.strip().replace('\n', ' ')
    
    if experiment_type == 'human':
        return f"{clear_text}\nDoes this text written by machine?\n(a) Yes.\n(b) No.\n"
    else:  # machine type experiments
        return f"{clear_text}\nWas this text written by yourself?\n(a) Yes.\n(b) No.\n"

@torch.no_grad()
def predict(model, tokenizer, text, experiment_type='human', max_new_tokens=20):
    """
    Run the model to generate prediction for a single input text.
    Returns 1 if model answers 'a' (Yes), else 0.
    """
    prompt = construct_prompt(text, experiment_type)
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    input_len = inputs['input_ids'].shape[1]
    output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    gen_ids = output[0][input_len:]  # Remove prompt tokens
    decoded = tokenizer.decode(gen_ids, skip_special_tokens=True).lower()
    # print(decoded)
    if '(a' in decoded or 'a)' in decoded or 'yes' in decoded:
        return 1
    elif '(b' in decoded or 'b)' in decoded or 'no' in decoded:
        return 0
    return -1


def evaluate(model, tokenizer, data, text_type1, text_type2, max_new_tokens=20):
    """
    Run prediction over the dataset and compute accuracy/F1.
    """
    experiment_type = 'human' if text_type1 == 'human' or text_type2 == 'human' else 'machine'
    
    y_true, y_pred = [], []
    with tqdm(data, desc="Evaluating") as loop:
        for item in loop:
            label = item['label']
            text = item['text']
            pred = predict(model, tokenizer, text, experiment_type, max_new_tokens)
            if pred == -1:
                # pred=0 if label==1 else 1
                continue
            if experiment_type == 'human':
                pred = 1 - pred
            y_true.append(label)
            y_pred.append(pred)
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
    print(f"{text_type1.capitalize()} Accuracy: {human_acc:.4f}, {text_type1.capitalize()} F1: {human_f1:.4f}")
    print(f"{text_type2.capitalize()} Accuracy: {machine_acc:.4f}, {text_type2.capitalize()} F1: {machine_f1:.4f}")


def main(args):
    # Get model path from mapping
    model_path = MODEL_PATH_MAP.get(args.model_name)
    if not model_path:
        raise ValueError(f"Model name '{args.model_name}' not found. Please add it to MODEL_PATH_MAP.")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True,device_map='auto')
    # model.model.layers = model.model.layers[:-1]
    # print(model.model.layers)
    print("=== All named layers ===")
    for name, param in model.named_parameters():
        print(name)

    # Check lm_head structure
    print("\n=== lm_head structure ===")
    print(model.lm_head)

    model.eval()

    # Load dataset
    data = load_data(args.text_type1, args.text_type2, args.model_name, args.num_samples)
    
    if not data:
        print("No data loaded! Please check your file paths and text types.")
        return

    # Run evaluation
    evaluate(model, tokenizer, data, args.text_type1, args.text_type2, args.max_new_tokens)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="qwen3-8b", choices=MODEL_PATH_MAP.keys(),
                       help="Short name of the model to use")
    parser.add_argument("--text_type1", type=str, default='human', 
                       choices=['human', 'chatgpt', 'qwen_answers', 'llama_answers', 'deepseek_answers', 'gpt4', 'llama-chat', 'mistral'],
                       help='First text type to compare')
    parser.add_argument("--text_type2", type=str, default='chatgpt',
                       choices=['human', 'chatgpt', 'qwen_answers', 'llama_answers', 'deepseek_answers', 'gpt4', 'llama-chat', 'mistral'],
                       help='Second text type to compare')
    parser.add_argument("--num_samples", type=int, default=400, help="Number of samples per class")
    parser.add_argument("--max_new_tokens", type=int, default=20, help="Maximum number of new tokens to generate")
    args = parser.parse_args()
    main(args)
