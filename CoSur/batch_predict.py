#!/usr/bin/env python3
"""
Batch prediction script that runs predict.py or edit_pred.py for multiple comparison scenarios.
Supports five experiment types:
- exp_type=0: Human experiments (human vs chatgpt/gpt4/mistral/llama-chat/qwen_answers/llama_answers/deepseek_answers)
- exp_type=1: Model validation in own subspace (auto-build subspace)
- exp_type=2: Generalization validation (use specified subspace to test other comparisons)
- exp_type=3: Human subspace generalization (use specified subspace)
- exp_type=4: Run all experiment types (0, 1, 2, 3) sequentially

Mode options:
- predict: Run only predict.py (baseline)
- edit_predict: Run only edit_pred.py (model editing)
- test: Run both predict.py and edit_pred.py for comparison on a random scenario
- all: Run both predict.py and edit_pred.py for all scenarios in the exp_type
"""

import os
import json
import subprocess
import argparse
from datetime import datetime
import re
import random
from exp_setting import get_exp_type_0_configs, get_exp_type_1_configs, get_exp_type_2_configs, get_exp_type_3_configs, get_scenario_configs, print_scenario_summary
from run_cmd import run_command, run_codebook_construction, run_prediction
from batch_setting import run_test_experiment, run_all_experiments, run_all_experiment_types, create_result_dir, check_data_availability


def main():
    parser = argparse.ArgumentParser(
        description="Batch prediction runner for multiple comparison scenarios.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Experiment Type Description:
  exp_type=0: Human experiments (human vs chatgpt/gpt4/mistral/llama-chat/model_self)
  exp_type=1: Model validation in own subspace (automatically build subspace)
  exp_type=2: Generalization validation (use specified subspace to test other comparisons)
  exp_type=3: Human subspace generalization (use specified subspace)
  exp_type=4: Run all experiment types (0, 1, 2, 3) sequentially

Mode Description:
  predict: Run only predict.py (baseline)
  edit_predict: Run only edit_pred.py (model editing)
  test: Run both predict.py and edit_pred.py for comparison on a random scenario
  all: Run both predict.py and edit_pred.py for all scenarios in the exp_type

Examples:
  # Run human experiments (baseline)
  python batch_predict.py --model_name qwen3-8b --exp_type 0 --mode predict
  
  # Run human experiments (model editing) 
  python batch_predict.py --model_name qwen3-8b --exp_type 0 --mode edit_predict --alpha 100 --subspace_path hidden_stats/qwen3-8b/human_vs_chatgpt/svd_rank_64
  
  # Run own subspace validation (automatically build subspace)
  python batch_predict.py --model_name qwen3-8b --exp_type 1 --mode edit_predict --svd_rank 64 --alpha 100
  
  # Run all experiments for exp_type=1 (both predict and edit_predict)
  python batch_predict.py --model_name qwen3-8b --exp_type 1 --mode all --svd_rank 64 --alpha 100
  
  # Run all experiments for exp_type=0 (both predict and edit_predict)
  python batch_predict.py --model_name qwen3-8b --exp_type 0 --mode all --alpha 100 --subspace_path hidden_stats/qwen3-8b/human_vs_chatgpt/svd_rank_64
  
  # Run generalization validation
  python batch_predict.py --model_name qwen3-8b --exp_type 2 --mode edit_predict --subspace_path hidden_stats/qwen3-8b/human_vs_qwen_answers/svd_rank_64 --alpha 100
  
  # Run human subspace generalization
  python batch_predict.py --model_name qwen3-8b --exp_type 3 --mode edit_predict --subspace_path hidden_stats/qwen3-8b/human_vs_chatgpt/svd_rank_64 --alpha 100
  
  # Test mode (randomly select one small experiment, run both predict and edit_predict)
  python batch_predict.py --model_name qwen3-8b --exp_type 1 --mode test --svd_rank 64 --alpha 100
  
  # Run ALL experiment types sequentially (exp_type=4)
  python batch_predict.py --model_name qwen3-8b --exp_type 4 --mode all --svd_rank 64 --alpha 100
        """
    )
    
    # Required parameters
    parser.add_argument("--model_name", type=str, default='qwen3-8b',
                       choices=['qwen3-8b', 'llama3.1-8b', 'deepseek-8b'],
                       help="Name of the model to use for predictions")
    
    parser.add_argument("--exp_type", type=int, choices=[0, 1, 2, 3, 4], default=0,
                       help="Experiment type: 0=human, 1=self-subspace, 2=generalization, 3=human-subspace-generalization, 4=all experiment types")
    
    parser.add_argument("--mode", type=str, choices=['predict', 'edit_predict', 'test', 'all'], required=True,
                       help="Execution mode: 'predict' (baseline only), 'edit_predict' (model editing only), 'test' (both for comparison on random scenario), 'all' (both for all scenarios)")
    
    # General parameters
    parser.add_argument("--num_samples", type=int, default=600,
                       help="Maximum number of samples to process")
    
    parser.add_argument("--max_new_tokens", type=int, default=20,
                       help="Maximum number of new tokens to generate")
    
    # Edit prediction specific parameters
    parser.add_argument("--svd_rank", type=int, default=64,
                       help="SVD rank for subspace construction (for exp_type=1 and test/all mode)")
    
    parser.add_argument("--alpha", type=float, default=100.0,
                       help="Alpha parameter for model editing (required for edit_predict, test and all modes)")
    
    # Subspace path parameter
    parser.add_argument("--subspace_path", type=str, default=None,
                       help="Path to subspace directory (required for exp_type=0,2,3 with edit_predict/test/all mode; auto-built for exp_type=1)")
    
    # Control parameters
    parser.add_argument("--verbose", action="store_true",
                       help="Print detailed command information")
    
    parser.add_argument("--dry_run", action="store_true",
                       help="Show configurations without running predictions")
    
    parser.add_argument("--use_pca", action='store_true', help="Use PCA instead of SVD to compute subspace")
    
    args = parser.parse_args()
    
    # Handle exp_type=4: Run all experiment types
    if args.exp_type == 4:
        run_all_experiment_types(args)
        return
    
    # Handle different modes
    if args.mode == "test":
        run_test_experiment(args)
        return
    
    if args.mode == "all":
        run_all_experiments(args)
        return
    
    # Generate scenario configurations
    scenarios = get_scenario_configs(args)
    
    if not scenarios:
        print("No scenarios generated. Please check your parameters.")
        return
    
    # Print configuration summary
    print_scenario_summary(scenarios, args.exp_type)
    
    # Ask for confirmation unless dry run
    if not args.dry_run:
        response = input(f"\nProceed with exp_type={args.exp_type} batch prediction? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("Batch prediction cancelled.")
            return
    
    # Auto-build subspace if not provided for edit_predict mode with exp_type 0, 2, 3
    auto_built_subspace_info = None
    if args.mode == "edit_predict" and args.exp_type in [0, 2, 3] and not args.subspace_path:
        print(f"\nSubspace path not provided for exp_type={args.exp_type}. Building automatically...")
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
            args.model_name, build_text_type1, build_text_type2, args.svd_rank, args.use_pca,args.verbose
        )
        args.subspace_path = auto_built_subspace_path
        auto_built_subspace_info = {
            "path": auto_built_subspace_path,
            "text_type1": build_text_type1,
            "text_type2": build_text_type2,
            "svd_rank": args.svd_rank
        }
        print(f"Subspace built and saved to: {args.subspace_path}")
    
    if args.dry_run:
        print("\nDRY RUN MODE - No predictions will be executed.")
        return
    
    # Create result directory
    result_dir = create_result_dir()
    
    # Determine script to run
    script_name = "edit_pred.py" if args.mode == "edit_predict" else "predict.py"
    
    # Execute predictions
    all_results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\nStarting exp_type={args.exp_type} batch prediction using '{script_name}'...")
    print("="*60)
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n[{i}/{len(scenarios)}] Processing: {scenario['name']}")
        print(f"Type: exp_type={scenario['exp_type']}")
        print(f"Description: {scenario['description']}")
        
        # Check if required data files exist
        text_type1 = scenario['config']['text_type1']
        text_type2 = scenario['config']['text_type2']
        data_available, missing_files = check_data_availability(text_type1, text_type2, args.model_name)
        
        if not data_available:
            print(f"Warning: Missing data files: {missing_files}. Skipping...")
            continue
        
        # Handle subspace construction and tracking
        used_subspace_path = None
        
        # Handle subspace construction for exp_type=1 edit_predict mode
        if scenario.get('needs_subspace') and args.mode == "edit_predict":
            print("Building subspace automatically...")
            hidden_stats_path = run_codebook_construction(
                args.model_name, text_type1, text_type2, args.svd_rank, args.use_pca,args.verbose
            )
            scenario['config']['hidden_stats_dir'] = hidden_stats_path
            used_subspace_path = hidden_stats_path
            print(f"Subspace saved to: {hidden_stats_path}")
        
        # Handle subspace path for exp_type=0,2,3 edit_predict mode
        elif args.mode == "edit_predict" and args.exp_type in [0, 2, 3]:
            if args.subspace_path:
                scenario['config']['hidden_stats_dir'] = args.subspace_path
                used_subspace_path = args.subspace_path
                print(f"Using subspace: {args.subspace_path}")
            else:
                print(f"Error: Subspace path is required for exp_type={args.exp_type} but was not provided or built. Exiting.")
                return

        # Run prediction
        print("Executing prediction...")
        scenario_results = run_prediction(script_name, scenario['config'], args.verbose)
        
        if scenario_results:
            all_results[scenario['name']] = {
                "description": scenario['description'],
                "exp_type": scenario['exp_type'],
                "subspace_path": used_subspace_path,  # Add subspace path tracking
                "svd_rank": args.svd_rank if used_subspace_path else None,  # Track SVD rank used
                "metrics": scenario_results
            }
            print(f"✓ Successfully completed {scenario['name']}")
            print(f"  Overall Accuracy: {scenario_results.get('overall_accuracy', 'N/A'):.4f}")
            print(f"  Overall F1: {scenario_results.get('overall_f1', 'N/A'):.4f}")
            if used_subspace_path:
                print(f"  Used subspace: {used_subspace_path}")
        else:
            print(f"✗ Failed to get results for {scenario['name']}")
    
    # Save results to JSON
    output_filename = f"{args.model_name}_exp_type_{args.exp_type}_{args.mode}_batch_results_{timestamp}.json"
    output_path = os.path.join(result_dir, output_filename)
    
    final_output = {
        "execution_info": {
            "model_name": args.model_name,
            "exp_type": args.exp_type,
            "prediction_mode": args.mode,
            "execution_timestamp": timestamp,
            "total_scenarios": len(scenarios),
            "successful_scenarios": len(all_results)
        },
        "parameters": {
            "num_samples": args.num_samples,
            "max_new_tokens": args.max_new_tokens,
            "svd_rank": args.svd_rank if args.exp_type == 1 or args.mode == "test" else None,
            "alpha": args.alpha if args.mode in ["edit_predict", "test"] else None,
            "subspace_path": args.subspace_path,
        },
        "subspace_info": auto_built_subspace_info,  # Add auto-built subspace information
        "results": all_results
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)
    
    print("\n" + "="*60)
    print("BATCH PREDICTION COMPLETE")
    print("="*60)
    print(f"Experiment Type: exp_type={args.exp_type}")
    print(f"Mode: {args.mode}")
    print(f"Total scenarios processed: {len(all_results)}/{len(scenarios)}")
    print(f"Results saved to: {output_path}")
    
    # Print summary of results
    if all_results:
        print("\nResults Summary:")
        for scenario_name, result in all_results.items():
            metrics = result['metrics']
            print(f"  {scenario_name} (exp_type={result['exp_type']}):")
            print(f"    Overall Acc: {metrics.get('overall_accuracy', 'N/A'):.4f}, F1: {metrics.get('overall_f1', 'N/A'):.4f}")
            print(f"    Type1 Acc: {metrics.get('human_accuracy', 'N/A'):.4f}, F1: {metrics.get('human_f1', 'N/A'):.4f}")
            print(f"    Type2 Acc: {metrics.get('machine_accuracy', 'N/A'):.4f}, F1: {metrics.get('machine_f1', 'N/A'):.4f}")
            if result.get('subspace_path'):
                print(f"    Subspace: {result['subspace_path']}")
                if result.get('svd_rank'):
                    print(f"    SVD Rank: {result['svd_rank']}")

if __name__ == "__main__":
    main() 