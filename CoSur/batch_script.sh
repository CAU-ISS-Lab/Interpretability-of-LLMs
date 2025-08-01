
#ablation study
#alpha
python batch_predict.py --model_name qwen3-8b --exp_type 1 --mode edit_predict --svd_rank 64 --alpha 20
python batch_predict.py --model_name qwen3-8b --exp_type 1 --mode edit_predict --svd_rank 64 --alpha 50
python batch_predict.py --model_name qwen3-8b --exp_type 1 --mode edit_predict --svd_rank 64 --alpha 150

#k
python batch_predict.py --model_name qwen3-8b --exp_type 1 --mode edit_predict --svd_rank 8 --alpha 100
python batch_predict.py --model_name qwen3-8b --exp_type 1 --mode edit_predict --svd_rank 16 --alpha 100
python batch_predict.py --model_name qwen3-8b --exp_type 1 --mode edit_predict --svd_rank 32 --alpha 100

# #pca
python batch_predict.py --model_name qwen3-8b --exp_type 1 --mode edit_predict --svd_rank 64 --alpha 100 --use_pca

#main_result
python batch_predict.py --model_name qwen3-8b --exp_type 1 --mode edit_predict --svd_rank 64 --alpha 100
python batch_predict.py --model_name llama3.1-8b --exp_type 1 --mode edit_predict --svd_rank 64 --alpha 100
python batch_predict.py --model_name deepseek-8b --exp_type 1 --mode edit_predict --svd_rank 64 --alpha 100

#generalization
python batch_predict.py --model_name qwen3-8b --exp_type 2 --mode edit_predict --subspace_path hidden_stats/qwen3-8b/qwen_answers_vs_chatgpt/svd_rank_64 --alpha 100
python batch_predict.py --model_name llama3.1-8b --exp_type 2 --mode edit_predict --subspace_path hidden_stats/llama3.1-8b/llama_answers_vs_chatgpt/svd_rank_64 --alpha 100
python batch_predict.py --model_name deepseek-8b --exp_type 2 --mode edit_predict --subspace_path hidden_stats/deepseek-8b/deepseek_answers_vs_chatgpt/svd_rank_64 --alpha 100

#aigt_detection
python batch_predict.py --model_name qwen3-8b --exp_type 3 --mode edit_predict --subspace_path hidden_stats/qwen3-8b/human_vs_chatgpt/svd_rank_64 --alpha 100