import os
import json
import subprocess
import argparse
from datetime import datetime
import re
import random


def parse_prediction_output(output, config):
    """Parse the output from predict.py or edit_pred.py to extract metrics."""
    # Get the text types from config
    text_type1 = config.get('text_type1', 'unknown')
    text_type2 = config.get('text_type2', 'unknown')
    
    label0_line_start = f"{text_type1.capitalize()} Accuracy:"
    label1_line_start = f"{text_type2.capitalize()} Accuracy:"
    
    lines = output.split('\n')
    results = {}
    
    for line in lines:
        if "Overall Accuracy:" in line:
            match = re.search(r'(\d+\.\d+)', line)
            if match:
                results['overall_accuracy'] = float(match.group(1))
        elif "Overall F1 Score:" in line:
            match = re.search(r'(\d+\.\d+)', line)
            if match:
                results['overall_f1'] = float(match.group(1))
        elif line.strip().startswith(label0_line_start):
            # The first group is always stored as 'human' for consistency in the JSON output
            parts = line.split(',')
            acc_match = re.search(r'(\d+\.\d+)', parts[0])
            f1_match = re.search(r'(\d+\.\d+)', parts[1])
            if acc_match:
                results['human_accuracy'] = float(acc_match.group(1)) # Keep key consistent
            if f1_match:
                results['human_f1'] = float(f1_match.group(1)) # Keep key consistent
        elif line.strip().startswith(label1_line_start):
            # The second group is always stored as 'machine' for consistency
            parts = line.split(',')
            acc_match = re.search(r'(\d+\.\d+)', parts[0])
            f1_match = re.search(r'(\d+\.\d+)', parts[1])
            if acc_match:
                results['machine_accuracy'] = float(acc_match.group(1)) # Keep key consistent
            if f1_match:
                results['machine_f1'] = float(f1_match.group(1)) # Keep key consistent
    
    return results

def run_command(cmd, verbose=True):
    """Run a command and return the result."""
    if verbose:
        print(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running command:")
        print(f"Command: {' '.join(cmd)}")
        print(f"Stderr: {e.stderr}")
        return ""

def run_codebook_construction(model_name, text_type1, text_type2, svd_rank,use_pca=False,verbose=True):
    """Run codebook_construction.py to build subspace."""
    cmd = [
        "python", "codebook_construction.py",
        "--model_name", model_name,
        "--text_type1", text_type1, 
        "--text_type2", text_type2,
        "--svd_rank", str(svd_rank)
    ]
    if use_pca:
        cmd.append('--use_pca')
    
    print(f"Building subspace for {text_type1} vs {text_type2}...")
    output = run_command(cmd, verbose)
    
    # Return the expected path where hidden stats will be saved
    comparison_name = f"{text_type1}_vs_{text_type2}"
    hidden_stats_path = f"./hidden_stats/{model_name}/{comparison_name}/svd_rank_{svd_rank}"
    return hidden_stats_path

def run_prediction(script_name, config, verbose=True):
    """
    Run a prediction script with a validated configuration and return parsed results.
    This function tailors the command-line arguments based on the specific script being run.
    """
    cmd = ["python", script_name]

    # Define parameters required by each script
    script_params = {
        "predict.py": [
            "model_name", "text_type1", "text_type2", 
            "num_samples", "max_new_tokens"
        ],
        "edit_pred.py": [
            "model_name", "text_type1", "text_type2",
            "num_samples", "max_new_tokens", 
            "alpha", "hidden_stats_dir"
        ]
    }

    # Get the list of allowed parameters for the target script
    allowed_params = script_params.get(script_name)
    if not allowed_params:
        raise ValueError(f"Script '{script_name}' is not configured in run_prediction.")

    # Add only the allowed parameters from the config to the command
    for key in allowed_params:
        value = config.get(key)
        if value is not None:
            cmd.extend([f"--{key}", str(value)])
        else:
            # Raise an error if a required parameter is missing, except for hidden_stats_dir which can be optional in some contexts
            # (though batch_predict logic should always provide it when needed).
            if key != 'hidden_stats_dir':
                 print(f"Warning: Required parameter '--{key}' is missing for {script_name}. Skipping command execution.")
                 return {}


    output = run_command(cmd, verbose)
    if output:
        return parse_prediction_output(output, config)
    return {}