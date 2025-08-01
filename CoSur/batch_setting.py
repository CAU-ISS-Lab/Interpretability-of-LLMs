import os
import json
import subprocess
import argparse
from datetime import datetime
import re
import random
from exp_setting import get_exp_type_0_configs, get_exp_type_1_configs, get_exp_type_2_configs, get_exp_type_3_configs, get_scenario_configs, print_scenario_summary
from run_cmd import run_command, run_codebook_construction, run_prediction


def create_result_dir():
    """Create predict_result directory     all_results = {
        "execution_info": {
            "model_name": args.model_name,
            "exp_type": args.exp_type,
            "prediction_mode": "all",
            "execution_timestamp": timestamp,
            "total_scenarios": len(scenarios),
            "valid_scenarios": len(valid_scenarios),
            "invalid_scenarios": len(invalid_scenarios)
        },
        "parameters": {
            "num_samples": args.num_samples,
            "max_new_tokens": args.max_new_tokens,
            "svd_rank": args.svd_rank if args.exp_type == 1 else None,
            "alpha": args.alpha,
            "subspace_path": args.subspace_path,
        },
        "global_subspace_info": build_info if needs_global_subspace_build else None,
        "results": {}
    }."""
    result_dir = "predict_result"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    return result_dir

def check_data_availability(text_type1, text_type2, model_name):
    """Check if the required data files exist based on the new loading logic."""
    required_files = []
    
    # Check if either text_type uses the new _answers format
    new_answer_types = {'qwen_answers', 'llama_answers', 'deepseek_answers'}
    if text_type1 in new_answer_types or text_type2 in new_answer_types:
        # Use the new combined all files for machine experiments
        required_files.append("data/test_all.jsonl")
        return True, []  # Assume the file exists for now since this is new functionality
    
    # Determine experiment type using same logic as other files
    experiment_type = 'human' if text_type1 == 'human' or text_type2 == 'human' else 'machine'
    
    if experiment_type == 'human':
        # Human experiments use specific combination data files
        if (text_type1 == 'human' and text_type2 == 'chatgpt') or (text_type1 == 'chatgpt' and text_type2 == 'human'):
            required_files.append("data/test_all.jsonl")
        else:
            # Other human experiment combinations use CSV
            required_files.append("data/test.csv")
    else:
        # Machine experiments use test_all.jsonl
        required_files.append("data/test_all.jsonl")
    
    # Check for existence of files
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    return len(missing_files) == 0, missing_files

def run_test_experiment(args):
    """Run a complete test experiment with both predict and edit_predict modes."""
    # Auto-build subspace if needed for test mode
    if args.exp_type in [0, 2, 3] and not args.subspace_path:
        print(f"\nSubspace path not provided for exp_type={args.exp_type} in test mode. Building automatically...")
        build_text_type1, build_text_type2 = None, 'chatgpt'
        
        if args.exp_type in [0, 3]:
            build_text_type1 = 'human'
            print(f"Building subspace for: {build_text_type1} vs {build_text_type2}")
        elif args.exp_type == 2:
            model_to_self = {
                'qwen3-8b': 'qwen_answers',
                'llama3.1-8b': 'llama_answers',
                'deepseek-8b': 'deepseek_answers'
            }
            build_text_type1 = model_to_self[args.model_name]
            print(f"Building subspace for: {build_text_type1} (self) vs {build_text_type2}")
        
        auto_built_subspace_path = run_codebook_construction(
            args.model_name, build_text_type1, build_text_type2, args.svd_rank, args.verbose
        )
        args.subspace_path = auto_built_subspace_path
        print(f"Subspace built and saved to: {args.subspace_path}")

    # Generate all scenarios for the exp_type
    # Use a temporary args object with predict mode to generate base scenarios
    temp_args = argparse.Namespace(**vars(args))
    temp_args.mode = "predict"
    scenarios = get_scenario_configs(temp_args)
    
    if not scenarios:
        print("No scenarios generated for test. Please check your parameters.")
        return
    
    # Randomly select one scenario
    selected_scenario = random.choice(scenarios)
    print(f"\n{'='*60}")
    print("TEST MODE: Running complete experiment for randomly selected scenario")
    print(f"{'='*60}")
    print(f"Selected scenario: {selected_scenario['name']}")
    print(f"Description: {selected_scenario['description']}")
    print(f"Exp_type: {selected_scenario['exp_type']}")
    
    # Create result directory
    result_dir = create_result_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    test_results = {
        "test_info": {
            "model_name": args.model_name,
            "exp_type": args.exp_type,
            "selected_scenario": selected_scenario['name'],
            "description": selected_scenario['description'],
            "execution_timestamp": timestamp
        },
        "parameters": {
            "num_samples": args.num_samples,
            "max_new_tokens": args.max_new_tokens,
            "svd_rank": args.svd_rank if args.exp_type == 1 else None,
            "alpha": args.alpha,
            "subspace_path": args.subspace_path if args.exp_type in [2, 3] else None,
        },
        "subspace_info": None,  # Will be filled if subspace is built/used
        "results": {}
    }
    
    # Check data availability
    text_type1 = selected_scenario['config']['text_type1']
    text_type2 = selected_scenario['config']['text_type2']
    data_available, missing_files = check_data_availability(text_type1, text_type2, args.model_name)
    
    if not data_available:
        print(f"Error: Missing data files: {missing_files}")
        return
    
    # Run predict.py (baseline)
    print("\n--- Running predict.py (baseline) ---")
    predict_config = selected_scenario['config'].copy()
    predict_results = run_prediction("predict.py", predict_config, args.verbose)
    
    if predict_results:
        test_results['results']['predict_baseline'] = predict_results
        print(f"✓ Baseline prediction completed")
        print(f"  Overall Accuracy: {predict_results.get('overall_accuracy', 'N/A'):.4f}")
        print(f"  Overall F1: {predict_results.get('overall_f1', 'N/A'):.4f}")
    else:
        print("✗ Baseline prediction failed")
        return
    
    # Prepare edit_predict configuration
    print("\n--- Preparing edit_pred.py (with model editing) ---")
    edit_config = selected_scenario['config'].copy()
    edit_config["alpha"] = args.alpha
    
    # Handle subspace for different exp_types
    subspace_available = False
    used_subspace_path = None
    
    if args.exp_type == 1:
        # Build subspace automatically for exp_type=1 only
        print("Building subspace for test experiment...")
        hidden_stats_path = run_codebook_construction(
            args.model_name, text_type1, text_type2, args.svd_rank, args.verbose
        )
        edit_config['hidden_stats_dir'] = hidden_stats_path
        used_subspace_path = hidden_stats_path
        test_results['subspace_info'] = {
            "path": hidden_stats_path,
            "text_type1": text_type1,
            "text_type2": text_type2,
            "svd_rank": args.svd_rank,
            "auto_built": True
        }
        print(f"Subspace saved to: {hidden_stats_path}")
        subspace_available = True
    
    elif args.exp_type in [0, 2, 3] and args.subspace_path:
        # Use provided subspace path for other exp_types
        edit_config['hidden_stats_dir'] = args.subspace_path
        used_subspace_path = args.subspace_path
        test_results['subspace_info'] = {
            "path": args.subspace_path,
            "auto_built": False
        }
        subspace_available = True
        print(f"Using provided subspace: {args.subspace_path}")
    
    # Run edit_pred.py if subspace is available
    if subspace_available:
        print("\n--- Running edit_pred.py (with model editing) ---")
        edit_results = run_prediction("edit_pred.py", edit_config, args.verbose)
        
        if edit_results:
            test_results['results']['edit_predict'] = edit_results
            print(f"✓ Model editing prediction completed")
            print(f"  Overall Accuracy: {edit_results.get('overall_accuracy', 'N/A'):.4f}")
            print(f"  Overall F1: {edit_results.get('overall_f1', 'N/A'):.4f}")
            if used_subspace_path:
                print(f"  Used subspace: {used_subspace_path}")
            
            # Compare results
            baseline_acc = predict_results.get('overall_accuracy', 0)
            edit_acc = edit_results.get('overall_accuracy', 0)
            improvement = edit_acc - baseline_acc
            print(f"\n--- Performance Comparison ---")
            print(f"Baseline accuracy: {baseline_acc:.4f}")
            print(f"Edited accuracy: {edit_acc:.4f}")
            print(f"Improvement: {improvement:+.4f}")
            
            test_results['results']['comparison'] = {
                "baseline_accuracy": baseline_acc,
                "edited_accuracy": edit_acc,
                "accuracy_improvement": improvement,
                "subspace_path": used_subspace_path
            }
        else:
            print("✗ Model editing prediction failed")
    else:
        print("\n--- Skipping edit_pred.py (no subspace available) ---")
        if args.exp_type in [0, 2, 3]:
            print(f"For exp_type={args.exp_type}, please provide --subspace_path parameter")
    
    # Save test results
    output_filename = f"{args.model_name}_exp_type_{args.exp_type}_test_results_{timestamp}.json"
    output_path = os.path.join(result_dir, output_filename)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, indent=4, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print("TEST EXPERIMENT COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {output_path}")

def run_all_experiments(args, skip_confirmation=False):
    """Run complete experiments with both predict and edit_predict modes for all scenarios."""
    # Generate all scenarios for the exp_type
    temp_args = argparse.Namespace(**vars(args))
    temp_args.mode = "predict"
    scenarios = get_scenario_configs(temp_args)
    
    if not scenarios:
        print("No scenarios generated for all mode. Please check your parameters.")
        return
    
    print(f"\n{'='*80}")
    print(f"ALL MODE: Running complete experiments for exp_type={args.exp_type}")
    print(f"{'='*80}")
    
    # Print basic parameters first
    print(f"\nBASIC PARAMETERS:")
    print(f"  Model Name: {args.model_name}")
    print(f"  Experiment Type: exp_type={args.exp_type}")
    print(f"  Number of Samples: {args.num_samples}")
    print(f"  Max New Tokens: {args.max_new_tokens}")
    print(f"  Alpha (for editing): {args.alpha}")
    if args.exp_type == 1:
        print(f"  SVD Rank (auto-build subspace): {args.svd_rank}")
    elif args.subspace_path:
        print(f"  Subspace Path: {args.subspace_path}")
    print(f"  Verbose Mode: {args.verbose}")
    
    # Determine subspace strategy before validation
    subspace_path_for_run = args.subspace_path
    needs_global_subspace_build = False
    build_info = {}

    if args.exp_type in [0, 2, 3] and not args.subspace_path:
        needs_global_subspace_build = True
        build_text_type2 = 'chatgpt'
        if args.exp_type in [0, 3]:
            build_text_type1 = 'human'
        elif args.exp_type == 2:
            model_to_self = {
                'qwen3-8b': 'qwen_answers',
                'llama3.1-8b': 'llama_answers',
                'deepseek-8b': 'deepseek_answers'
            }
            build_text_type1 = model_to_self[args.model_name]
        
        comparison_name = f"{build_text_type1}_vs_{build_text_type2}"
        subspace_path_for_run = f"./hidden_stats/{args.model_name}/{comparison_name}/svd_rank_{args.svd_rank}"
        build_info = {
            'text_type1': build_text_type1,
            'text_type2': build_text_type2,
            'path': subspace_path_for_run
        }
        print(f"\nINFO: --subspace_path not provided. Subspace will be auto-built at:")
        print(f"      {subspace_path_for_run}")

    print(f"\nEXECUTION PLAN:")
    print(f"  Total scenarios: {len(scenarios)}")
    print(f"  Each scenario will run:")
    print(f"    1. predict.py (baseline)")
    print(f"    2. edit_pred.py (model editing)")
    print(f"    3. Performance comparison")
    
    # Pre-validate all scenarios and prepare configurations
    valid_scenarios = []
    invalid_scenarios = []
    
    print(f"\nPRE-VALIDATION AND CONFIGURATION PREPARATION:")
    print("-" * 60)
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n[{i}/{len(scenarios)}] Validating: {scenario['name']}")
        print(f"  Description: {scenario['description']}")
        
        # Check data availability
        text_type1 = scenario['config']['text_type1']
        text_type2 = scenario['config']['text_type2']
        data_available, missing_files = check_data_availability(text_type1, text_type2, args.model_name)
        
        if not data_available:
            print(f"  Missing data files: {missing_files}")
            invalid_scenarios.append({
                'scenario': scenario,
                'reason': f"Missing data files: {missing_files}"
            })
            continue
        
        # Prepare predict configuration
        predict_config = scenario['config'].copy()
        print(f"  ✓ Predict config prepared")
        print(f"    - text_type1: {text_type1}")
        print(f"    - text_type2: {text_type2}")
        print(f"    - model_name: {predict_config['model_name']}")
        print(f"    - num_samples: {predict_config['num_samples']}")
        print(f"    - max_new_tokens: {predict_config['max_new_tokens']}")
        
        # Prepare edit_predict configuration
        edit_config = scenario['config'].copy()
        edit_config["alpha"] = args.alpha
        
        # Handle subspace for different exp_types
        subspace_available = False
        subspace_path = None
        
        if args.exp_type == 1:
            # For exp_type=1, subspace will be auto-built
            comparison_name = f"{text_type1}_vs_{text_type2}"
            subspace_path = f"./hidden_stats/{args.model_name}/{comparison_name}/svd_rank_{args.svd_rank}"
            edit_config['hidden_stats_dir'] = subspace_path
            subspace_available = True
            print(f"  ✓ Edit config prepared (auto-build subspace)")
            print(f"    - Will build subspace at: {subspace_path}")
        elif args.exp_type in [0, 2, 3]:
            # Use the pre-determined subspace path (either provided or to be auto-built)
            edit_config['hidden_stats_dir'] = subspace_path_for_run
            subspace_path = subspace_path_for_run
            subspace_available = True
            print(f"  ✓ Edit config prepared")
            if needs_global_subspace_build:
                print(f"    - Will use auto-built subspace from: {subspace_path_for_run}")
            else:
                print(f"    - Using provided subspace: {subspace_path_for_run}")
        else:
            print(f"  No subspace available for editing")
            invalid_scenarios.append({
                'scenario': scenario,
                'reason': f"No subspace path provided for exp_type={args.exp_type} and auto-build not configured."
            })
            continue
        
        print(f"    - alpha: {edit_config['alpha']}")
        print(f"    - hidden_stats_dir: {edit_config['hidden_stats_dir']}")
        
        # Store validated scenario with prepared configs
        valid_scenarios.append({
            'scenario': scenario,
            'predict_config': predict_config,
            'edit_config': edit_config,
            'text_type1': text_type1,
            'text_type2': text_type2,
            'subspace_path': subspace_path,
            'needs_subspace_build': args.exp_type == 1
        })
        print(f"  Scenario validated and configs prepared")
    
    # Print validation summary
    print(f"\nVALIDATION SUMMARY:")
    print(f"  Valid scenarios: {len(valid_scenarios)}")
    print(f"  Invalid scenarios: {len(invalid_scenarios)}")
    
    if invalid_scenarios:
        print(f"\nINVALID SCENARIOS:")
        for item in invalid_scenarios:
            print(f"  - {item['scenario']['name']}: {item['reason']}")
    
    if not valid_scenarios:
        print("\nNo valid scenarios found. Exiting.")
        return
    
    print(f"\nVALID SCENARIOS TO RUN:")
    for i, item in enumerate(valid_scenarios, 1):
        scenario = item['scenario']
        print(f"  [{i}] {scenario['name']}")
        print(f"      - {scenario['description']}")
        if item['needs_subspace_build']:
            print(f"      - Will auto-build subspace: {item['subspace_path']}")
        else:
            print(f"      - Using subspace: {item['subspace_path']}")
    
    # Ask for confirmation
    if not skip_confirmation:
        print(f"\n{'='*80}")
        print("READY TO START ALL EXPERIMENTS")
        print(f"{'='*80}")
        response = input(f"\nProceed with running all {len(valid_scenarios)} scenarios? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("All experiments cancelled.")
            return
    
    # Build global subspace if needed
    if needs_global_subspace_build:
        print("\nBuilding required global subspace...")
        auto_built_subspace_path = run_codebook_construction(
            args.model_name, build_info['text_type1'], build_info['text_type2'], args.svd_rank, args.verbose
        )
        print(f"Subspace built and saved to: {auto_built_subspace_path}")
        # Sanity check that the built path matches the predicted path
        assert auto_built_subspace_path == build_info['path']

    # Create result directory
    result_dir = create_result_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    all_results = {
        "execution_info": {
            "model_name": args.model_name,
            "exp_type": args.exp_type,
            "prediction_mode": "all",
            "execution_timestamp": timestamp,
            "total_scenarios": len(scenarios),
            "valid_scenarios": len(valid_scenarios),
            "invalid_scenarios": len(invalid_scenarios)
        },
        "parameters": {
            "num_samples": args.num_samples,
            "max_new_tokens": args.max_new_tokens,
            "svd_rank": args.svd_rank if args.exp_type == 1 else None,
            "alpha": args.alpha,
            "subspace_path": args.subspace_path if args.exp_type in [0, 2, 3] else None,
        },
        "results": {}
    }
    
    successful_scenarios = 0
    
    print(f"\nSTARTING CONTINUOUS EXECUTION...")
    print(f"{'='*80}")
    
    # Run all valid scenarios continuously
    for i, item in enumerate(valid_scenarios, 1):
        scenario = item['scenario']
        predict_config = item['predict_config']
        edit_config = item['edit_config']
        text_type1 = item['text_type1']
        text_type2 = item['text_type2']
        
        print(f"\n[{i}/{len(valid_scenarios)}] Processing: {scenario['name']}")
        print(f"Type: exp_type={scenario['exp_type']}")
        print(f"Description: {scenario['description']}")
        
        scenario_results = {
            "description": scenario['description'],
            "exp_type": scenario['exp_type'],
            "text_type1": text_type1,
            "text_type2": text_type2,
            "predict_baseline": {},
            "edit_predict": {},
            "comparison": {}
        }
        
        # Step 1: Run predict.py (baseline)
        print(f"\nStep 1: Running predict.py (baseline)")
        predict_results = run_prediction("predict.py", predict_config, args.verbose)
        
        if predict_results:
            scenario_results['predict_baseline'] = predict_results
            print(f"Baseline prediction completed")
            print(f"   Overall Accuracy: {predict_results.get('overall_accuracy', 'N/A'):.4f}")
            print(f"   Overall F1: {predict_results.get('overall_f1', 'N/A'):.4f}")
        else:
            print("Baseline prediction failed. Skipping edit_predict for this scenario.")
            all_results['results'][scenario['name']] = scenario_results
            continue
        
        # Step 2: Handle subspace construction if needed
        used_subspace_path = None
        if item['needs_subspace_build']:
            print(f"\nStep 2: Building subspace for experiment")
            hidden_stats_path = run_codebook_construction(
                args.model_name, text_type1, text_type2, args.svd_rank, args.verbose
            )
            edit_config['hidden_stats_dir'] = hidden_stats_path
            used_subspace_path = hidden_stats_path
            print(f"Subspace built and saved to: {hidden_stats_path}")
        else:
            used_subspace_path = edit_config['hidden_stats_dir']
            print(f"\nStep 2: Using provided subspace: {edit_config['hidden_stats_dir']}")
        
        # Add subspace info to scenario results
        scenario_results['subspace_info'] = {
            "path": used_subspace_path,
            "auto_built": item['needs_subspace_build'],
            "svd_rank": args.svd_rank if item['needs_subspace_build'] else None
        }
        
        # Step 3: Run edit_pred.py
        print(f"\nStep 3: Running edit_pred.py (model editing)")
        edit_results = run_prediction("edit_pred.py", edit_config, args.verbose)
        
        if edit_results:
            scenario_results['edit_predict'] = edit_results
            print(f"Model editing prediction completed")
            print(f"   Overall Accuracy: {edit_results.get('overall_accuracy', 'N/A'):.4f}")
            print(f"   Overall F1: {edit_results.get('overall_f1', 'N/A'):.4f}")
            
            # Step 4: Compare results
            baseline_acc = predict_results.get('overall_accuracy', 0)
            edit_acc = edit_results.get('overall_accuracy', 0)
            improvement = edit_acc - baseline_acc
            print(f"\nStep 4: Performance Comparison")
            print(f"   Baseline accuracy: {baseline_acc:.4f}")
            print(f"   Edited accuracy: {edit_acc:.4f}")
            print(f"   Improvement: {improvement:+.4f}")
            
            scenario_results['comparison'] = {
                "baseline_accuracy": baseline_acc,
                "edited_accuracy": edit_acc,
                "accuracy_improvement": improvement,
                "subspace_path": used_subspace_path
            }
            
            successful_scenarios += 1
            print(f"Scenario {scenario['name']} completed successfully!")
        else:
            print("Model editing prediction failed")
        
        # Add scenario results to all_results
        all_results['results'][scenario['name']] = scenario_results
        
        # Print progress
        remaining = len(valid_scenarios) - i
        if remaining > 0:
            print(f"\n{remaining} scenarios remaining...")
    
    # Update execution info
    all_results['execution_info']['successful_scenarios'] = successful_scenarios
    
    # Save all results
    output_filename = f"{args.model_name}_exp_type_{args.exp_type}_all_results_{timestamp}.json"
    output_path = os.path.join(result_dir, output_filename)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print("ALL EXPERIMENTS COMPLETE")
    print(f"{'='*80}")
    print(f"Total scenarios processed: {successful_scenarios}/{len(valid_scenarios)}")
    print(f"Results saved to: {output_path}")
    
    # Print summary of results
    if successful_scenarios > 0:
        print(f"\nRESULTS SUMMARY:")
        for scenario_name, result in all_results['results'].items():
            if result['predict_baseline'] and result['edit_predict']:
                baseline_acc = result['predict_baseline'].get('overall_accuracy', 0)
                edit_acc = result['edit_predict'].get('overall_accuracy', 0)
                improvement = edit_acc - baseline_acc
                subspace_path = result.get('subspace_info', {}).get('path', 'N/A')
                print(f"   {scenario_name}:")
                print(f"     Baseline: {baseline_acc:.4f} | Edited: {edit_acc:.4f} | Improvement: {improvement:+.4f}")
                print(f"     Subspace: {subspace_path}")
                if result.get('subspace_info', {}).get('svd_rank'):
                    print(f"     SVD Rank: {result['subspace_info']['svd_rank']}")

def run_all_experiment_types(args):
    """Run all experiment types (0, 1, 2, 3) in sequence for exp_type=4."""
    print(f"\n{'='*100}")
    print("EXP_TYPE=4: RUNNING ALL EXPERIMENT TYPES (0, 1, 2, 3)")
    print(f"{'='*100}")
    
    # Print global parameters
    print(f"\nGLOBAL PARAMETERS:")
    print(f"  Model Name: {args.model_name}")
    print(f"  Mode: {args.mode}")
    print(f"  Number of Samples: {args.num_samples}")
    print(f"  Max New Tokens: {args.max_new_tokens}")
    print(f"  Alpha (for editing): {args.alpha}")
    print(f"  SVD Rank: {args.svd_rank}")
    print(f"  Verbose Mode: {args.verbose}")
    if args.subspace_path:
        print(f"  Original Subspace Path: {args.subspace_path} (will be ignored for auto-subspace building)")
    
    # Define all experiment types and their descriptions
    experiment_types = [
        {
            'exp_type': 0,
            'name': 'Human Experiments',
            'description': 'Human vs machine models (ChatGPT, GPT-4, Llama-chat, Mistral, model-specific)',
            'auto_subspace': 'human_vs_chatgpt'
        },
        {
            'exp_type': 1,
            'name': 'Self-Subspace Validation',
            'description': 'Model validation in own subspace (auto-build per scenario)',
            'auto_subspace': 'per_scenario'
        },
        {
            'exp_type': 2,
            'name': 'Generalization Validation',
            'description': 'Cross-model generalization using self_vs_chatgpt subspace',
            'auto_subspace': 'self_vs_chatgpt'
        },
        {
            'exp_type': 3,
            'name': 'Human Subspace Generalization',
            'description': 'Human vs models using human_vs_chatgpt subspace',
            'auto_subspace': 'human_vs_chatgpt'
        }
    ]
    
    # Count total scenarios for all experiment types
    total_scenarios = 0
    exp_summaries = []
    
    print(f"\nPREVIEW OF ALL EXPERIMENT TYPES:")
    print("-" * 80)
    
    for exp_info in experiment_types:
        exp_type = exp_info['exp_type']
        temp_args = argparse.Namespace(**vars(args))
        temp_args.exp_type = exp_type
        temp_args.mode = "predict"  # Use predict mode to get base scenarios
        temp_args.subspace_path = None  # Clear subspace path for auto-building
        
        scenarios = get_scenario_configs(temp_args)
        scenario_count = len(scenarios)
        total_scenarios += scenario_count
        
        exp_summaries.append({
            'exp_type': exp_type,
            'name': exp_info['name'],
            'description': exp_info['description'],
            'auto_subspace': exp_info['auto_subspace'],
            'scenario_count': scenario_count,
            'scenarios': scenarios
        })
        
        print(f"\n[EXP_TYPE={exp_type}] {exp_info['name']}")
        print(f"  Description: {exp_info['description']}")
        print(f"  Auto-subspace strategy: {exp_info['auto_subspace']}")
        print(f"  Number of scenarios: {scenario_count}")
        
        if scenario_count > 0:
            print(f"  Scenarios:")
            for i, scenario in enumerate(scenarios, 1):
                print(f"    {i}. {scenario['name']} - {scenario['description']}")
        else:
            print(f"    No scenarios available")
    
    print(f"\nTOTAL EXECUTION PLAN:")
    print(f"  Total experiment types: {len(experiment_types)}")
    print(f"  Total scenarios across all types: {total_scenarios}")
    if args.mode in ['all', 'test']:
        print(f"  Each scenario will run both predict.py and edit_pred.py")
    else:
        script_name = "edit_pred.py" if args.mode == "edit_predict" else "predict.py"
        print(f"  Each scenario will run {script_name}")
    
    # Ask for confirmation
    print(f"\n{'='*80}")
    print("CONFIRMATION REQUIRED")
    print(f"{'='*80}")
    response = input(f"\nProceed with running ALL {len(experiment_types)} experiment types "
                    f"({total_scenarios} total scenarios)? (y/N): ")
    if response.lower() not in ['y', 'yes']:
        print("All experiments cancelled.")
        return
    
    # Execute all experiment types in sequence
    successful_exp_types = 0
    failed_exp_types = []
    
    print(f"\nSTARTING SEQUENTIAL EXECUTION OF ALL EXPERIMENT TYPES...")
    print(f"{'='*100}")
    
    for i, exp_summary in enumerate(exp_summaries, 1):
        exp_type = exp_summary['exp_type']
        exp_name = exp_summary['name']
        scenario_count = exp_summary['scenario_count']
        
        if scenario_count == 0:
            print(f"\n[{i}/{len(exp_summaries)}] SKIPPING EXP_TYPE={exp_type} ({exp_name}) - No scenarios")
            continue
        
        print(f"\n[{i}/{len(exp_summaries)}] STARTING EXP_TYPE={exp_type} ({exp_name})")
        print(f"Scenarios to process: {scenario_count}")
        print("-" * 60)
        
        # Create a new args object for this experiment type
        exp_args = argparse.Namespace(**vars(args))
        exp_args.exp_type = exp_type
        exp_args.subspace_path = None  # Let auto-subspace building handle this
        
        try:
            if args.mode == "test":
                run_test_experiment(exp_args)
            elif args.mode == "all":
                run_all_experiments(exp_args, skip_confirmation=True)
            else:
                # Handle predict or edit_predict mode by calling main batch logic
                # We need to create a mini main function logic here
                scenarios = get_scenario_configs(exp_args)
                
                # Auto-build subspace if needed for edit_predict mode
                if exp_args.mode == "edit_predict" and exp_type in [0, 2, 3]:
                    print(f"\nBuilding auto-subspace for exp_type={exp_type}...")
                    build_text_type1, build_text_type2 = None, 'chatgpt'
                    
                    if exp_type in [0, 3]:
                        build_text_type1 = 'human'
                    elif exp_type == 2:
                        model_to_self = {
                            'qwen3-8b': 'qwen_answers',
                            'llama3.1-8b': 'llama_answers',
                            'deepseek-8b': 'deepseek_answers'
                        }
                        build_text_type1 = model_to_self[exp_args.model_name]
                    
                    auto_built_subspace_path = run_codebook_construction(
                        exp_args.model_name, build_text_type1, build_text_type2, exp_args.svd_rank, exp_args.verbose
                    )
                    exp_args.subspace_path = auto_built_subspace_path
                    print(f"Auto-built subspace saved to: {exp_args.subspace_path}")
                
                # Create mini batch execution
                result_dir = create_result_dir()
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                script_name = "edit_pred.py" if exp_args.mode == "edit_predict" else "predict.py"
                
                all_results = {}
                for scenario in scenarios:
                    # Check data availability
                    text_type1 = scenario['config']['text_type1']
                    text_type2 = scenario['config']['text_type2']
                    data_available, missing_files = check_data_availability(text_type1, text_type2, exp_args.model_name)
                    
                    if not data_available:
                        print(f"Warning: Missing data files for {scenario['name']}: {missing_files}. Skipping...")
                        continue
                    
                    # Handle subspace for edit_predict mode
                    if exp_args.mode == "edit_predict":
                        if exp_type == 1:
                            # Build subspace for each scenario in exp_type=1
                            hidden_stats_path = run_codebook_construction(
                                exp_args.model_name, text_type1, text_type2, exp_args.svd_rank, exp_args.verbose
                            )
                            scenario['config']['hidden_stats_dir'] = hidden_stats_path
                        elif exp_type in [0, 2, 3] and exp_args.subspace_path:
                            scenario['config']['hidden_stats_dir'] = exp_args.subspace_path
                    
                    # Run prediction
                    print(f"Processing {scenario['name']}...")
                    scenario_results = run_prediction(script_name, scenario['config'], exp_args.verbose)
                    
                    if scenario_results:
                        all_results[scenario['name']] = {
                            "description": scenario['description'],
                            "exp_type": scenario['exp_type'],
                            "metrics": scenario_results
                        }
                        print(f"✓ {scenario['name']} completed")
                
                # Save results
                output_filename = f"{exp_args.model_name}_exp_type_{exp_type}_{exp_args.mode}_batch_results_{timestamp}.json"
                output_path = os.path.join(result_dir, output_filename)
                
                final_output = {
                    "execution_info": {
                        "model_name": exp_args.model_name,
                        "exp_type": exp_type,
                        "prediction_mode": exp_args.mode,
                        "execution_timestamp": timestamp,
                        "total_scenarios": len(scenarios),
                        "successful_scenarios": len(all_results)
                    },
                    "parameters": {
                        "num_samples": exp_args.num_samples,
                        "max_new_tokens": exp_args.max_new_tokens,
                        "svd_rank": exp_args.svd_rank if exp_type == 1 else None,
                        "alpha": exp_args.alpha if exp_args.mode == "edit_predict" else None,
                        "subspace_path": exp_args.subspace_path,
                    },
                    "results": all_results
                }
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(final_output, f, indent=4, ensure_ascii=False)
                
                print(f"Results saved to: {output_path}")
            
            successful_exp_types += 1
            print(f"✓ EXP_TYPE={exp_type} ({exp_name}) completed successfully!")
            
        except Exception as e:
            print(f"✗ EXP_TYPE={exp_type} ({exp_name}) failed with error: {str(e)}")
            failed_exp_types.append((exp_type, exp_name, str(e)))
            continue
        
        # Print progress
        remaining = len(exp_summaries) - i
        if remaining > 0:
            print(f"\n{remaining} experiment type(s) remaining...")
    
    # Final summary
    print(f"\n{'='*100}")
    print("ALL EXPERIMENT TYPES EXECUTION COMPLETE")
    print(f"{'='*100}")
    print(f"Total experiment types processed: {successful_exp_types}/{len(experiment_types)}")
    
    if successful_exp_types > 0:
        print(f"\nSUCCESSFUL EXPERIMENT TYPES:")
        for exp_summary in exp_summaries:
            exp_type = exp_summary['exp_type']
            exp_name = exp_summary['name']
            if exp_type not in [ft[0] for ft in failed_exp_types]:
                print(f"  ✓ EXP_TYPE={exp_type}: {exp_name}")
    
    if failed_exp_types:
        print(f"\nFAILED EXPERIMENT TYPES:")
        for exp_type, exp_name, error in failed_exp_types:
            print(f"  ✗ EXP_TYPE={exp_type}: {exp_name} - {error}")
    
    print(f"\nAll experiment type results have been saved to separate files in the predict_result/ directory.")
