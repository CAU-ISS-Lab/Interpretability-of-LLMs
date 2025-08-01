import os
import json
import subprocess
import argparse
from datetime import datetime
import re
import random

def get_exp_type_0_configs(args):
    """Generate human experiment configurations (exp_type=0)."""
    base_config = {
        "model_name": args.model_name,
        "num_samples": args.num_samples,
        "max_new_tokens": args.max_new_tokens,
    }
    
    # Add alpha parameter for edit_predict mode
    if args.mode == "edit_predict":
        base_config["alpha"] = args.alpha
    
    # Select corresponding self experiment based on model
    model_to_self = {
        'qwen3-8b': 'qwen_answers',
        'llama3.1-8b': 'llama_answers', 
        'deepseek-8b': 'deepseek_answers'
    }
    
    self_type = model_to_self[args.model_name]
    
    scenarios = [
        # Human vs machine experiments (including model-specific self)
        {"text_type1": "human", "text_type2": "chatgpt", "description": "Human vs ChatGPT"},
        {"text_type1": "human", "text_type2": "gpt4", "description": "Human vs GPT-4"},
        {"text_type1": "human", "text_type2": "llama-chat", "description": "Human vs Llama-chat"},
        {"text_type1": "human", "text_type2": "mistral", "description": "Human vs Mistral"},
        {"text_type1": "human", "text_type2": self_type, "description": f"Human vs {self_type}"},
    ]
    
    configs = []
    for scenario in scenarios:
        config = {
            "name": f"{scenario['text_type1']}_vs_{scenario['text_type2']}",
            "description": scenario["description"],
            "exp_type": 0,
            "config": {**base_config, **scenario}
        }
        # For exp_type=0 edit_predict mode, add subspace path if provided
        if args.mode == "edit_predict" and args.subspace_path:
            config["config"]["hidden_stats_dir"] = args.subspace_path
        
        configs.append(config)
    
    return configs

def get_exp_type_1_configs(args):
    """Generate self-subspace validation configurations (exp_type=1)."""
    base_config = {
        "model_name": args.model_name,
        "num_samples": args.num_samples,
        "max_new_tokens": args.max_new_tokens,
    }
    
    # Add alpha parameter for edit_predict mode
    if args.mode == "edit_predict":
        base_config["alpha"] = args.alpha
    
    # Define experiment combinations for different models
    model_to_self = {
        'qwen3-8b': 'qwen_answers',
        'llama3.1-8b': 'llama_answers', 
        'deepseek-8b': 'deepseek_answers'
    }
    
    self_type = model_to_self[args.model_name]
    other_types = ['chatgpt', 'llama_answers', 'deepseek_answers']
    if self_type in other_types:
        other_types.remove(self_type)
    if self_type == 'qwen_answers':
        other_types = ['chatgpt', 'llama_answers', 'deepseek_answers']
    elif self_type == 'llama_answers':
        other_types = ['chatgpt', 'qwen_answers', 'deepseek_answers']  
    elif self_type == 'deepseek_answers':
        other_types = ['chatgpt', 'qwen_answers', 'llama_answers']
    
    scenarios = [
        # Human vs self
        {"text_type1": "human", "text_type2": self_type, "description": f"Human vs {self_type}"},
    ]
    
    # Self vs others
    for other_type in other_types:
        scenarios.append({
            "text_type1": self_type, 
            "text_type2": other_type, 
            "description": f"{self_type} vs {other_type}"
        })
    
    configs = []
    for scenario in scenarios:
        config = {
            "name": f"{scenario['text_type1']}_vs_{scenario['text_type2']}",
            "description": scenario["description"],
            "exp_type": 1,
            "config": {**base_config, **scenario},
        }
        # For edit_predict mode with exp_type=1, mark that subspace needs to be built
        if args.mode == "edit_predict":
            config["needs_subspace"] = True
        
        configs.append(config)
    
    return configs

def get_exp_type_2_configs(args):
    """Generate generalization validation configurations (exp_type=2)."""
    base_config = {
        "model_name": args.model_name,
        "num_samples": args.num_samples,
        "max_new_tokens": args.max_new_tokens,
    }
    
    # Add alpha parameter for edit_predict mode
    if args.mode == "edit_predict":
        base_config["alpha"] = args.alpha
    
    # Use specified subspace path for edit_predict mode
    if args.mode == "edit_predict" and not args.subspace_path:
        print("Warning: exp_type=2 with edit_predict mode requires --subspace_path parameter")
        return []
    
    # Define experiment combinations for different models
    model_to_self = {
        'qwen3-8b': 'qwen_answers',
        'llama3.1-8b': 'llama_answers',
        'deepseek-8b': 'deepseek_answers'
    }
    
    self_type = model_to_self[args.model_name]
    other_types = ['llama_answers', 'deepseek_answers']
    if self_type in other_types:
        other_types.remove(self_type)
    if self_type == 'qwen_answers':
        other_types = ['llama_answers', 'deepseek_answers']
    elif self_type == 'llama_answers':
        other_types = ['qwen_answers', 'deepseek_answers']
    elif self_type == 'deepseek_answers':
        other_types = ['qwen_answers', 'llama_answers']
    
    scenarios = []
    for other_type in other_types:
        scenarios.append({
            "text_type1": self_type,
            "text_type2": other_type,
            "description": f"{self_type} vs {other_type} (using specified subspace)"
        })
    
    configs = []
    for scenario in scenarios:
        config = {
            "name": f"{scenario['text_type1']}_vs_{scenario['text_type2']}_fixed",
            "description": scenario["description"],
            "exp_type": 2,
            "config": {**base_config, **scenario}
        }
        # Add subspace path for edit_predict mode
        if args.mode == "edit_predict":
            config["config"]["hidden_stats_dir"] = args.subspace_path
        
        configs.append(config)
    
    return configs

def get_exp_type_3_configs(args):
    """Generate human subspace generalization configurations (exp_type=3)."""
    base_config = {
        "model_name": args.model_name,
        "num_samples": args.num_samples,
        "max_new_tokens": args.max_new_tokens,
    }
    
    # Add alpha parameter for edit_predict mode
    if args.mode == "edit_predict":
        base_config["alpha"] = args.alpha
    
    # Use specified subspace path for edit_predict mode
    if args.mode == "edit_predict" and not args.subspace_path:
        print("Warning: exp_type=3 with edit_predict mode requires --subspace_path parameter")
        return []
    
    scenarios = [
        {"text_type1": "human", "text_type2": "chatgpt", "description": "Human vs chatgpt (using specified subspace)"},
        {"text_type1": "human", "text_type2": "qwen_answers", "description": "Human vs Qwen (using specified subspace)"},
        {"text_type1": "human", "text_type2": "deepseek_answers", "description": "Human vs DeepSeek (using specified subspace)"},
        {"text_type1": "human", "text_type2": "llama_answers", "description": "Human vs Llama (using specified subspace)"},
    ]
    
    configs = []
    for scenario in scenarios:
        config = {
            "name": f"{scenario['text_type1']}_vs_{scenario['text_type2']}_specified_subspace",
            "description": scenario["description"],
            "exp_type": 3,
            "config": {**base_config, **scenario}
        }
        # Add subspace path for edit_predict mode
        if args.mode == "edit_predict":
            config["config"]["hidden_stats_dir"] = args.subspace_path
        
        configs.append(config)
    
    return configs

def get_scenario_configs(args):
    """Generate configurations based on exp_type."""
    if args.exp_type == 0:
        return get_exp_type_0_configs(args)
    elif args.exp_type == 1:
        return get_exp_type_1_configs(args)
    elif args.exp_type == 2:
        return get_exp_type_2_configs(args)
    elif args.exp_type == 3:
        return get_exp_type_3_configs(args)
    else:
        raise ValueError(f"Unsupported exp_type: {args.exp_type}")

def print_scenario_summary(scenarios, exp_type):
    """Print a summary of all scenarios and their configurations."""
    print("\n" + "="*80)
    print(f"BATCH PREDICTION CONFIGURATION SUMMARY (EXP_TYPE={exp_type})")
    print("="*80)
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n[{i}] {scenario['name'].upper()}")
        print(f"Type: exp_type={scenario['exp_type']}")
        print(f"Description: {scenario['description']}")
        print("Parameters:")
        for key, value in scenario['config'].items():
            print(f"  --{key}: {value}")
        if scenario.get('needs_subspace'):
            print("  Note: Will auto-build subspace before edit prediction")
    
    print("\n" + "="*80)