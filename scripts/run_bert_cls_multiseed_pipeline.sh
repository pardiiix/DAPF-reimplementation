#!/usr/bin/env bash
set -euo pipefail

SEEDS=(0 1 2 3 4)

for SEED in "${SEEDS[@]}"; do
  echo "=============================="
  echo "Running BERT-CLS seed ${SEED}"
  echo "=============================="

  python -m baselines.train_bert_cls \
    --project_root ./ \
    --off_line_model_dir ./model/bert-base-uncased \
    --data_dir ./data/ \
    --src_data ccc_train_all \
    --trg_data adress-train_all \
    --trg_test_data adress-test_all \
    --output_dir ./output_bert_cls_multiseed/seed_${SEED} \
    --seed ${SEED} \
    --batch_size 4 \
    --num_epochs 10 \
    --learning_rate 1e-5 \
    --max_length 512 \
    --ce_class_weights \
    --save_hidden_states

  python -m baselines.run_bert_cls_interpretability \
    --results_csv ./output_bert_cls_multiseed/seed_${SEED}/test_results.csv \
    --interpretability_pt ./output_bert_cls_multiseed/seed_${SEED}/test_results_interpretability.pt \
    --output_dir ./output_bert_cls_interpretability_multiseed/seed_${SEED}

  python -m baselines.run_bert_cls_attribution \
    --model_dir ./output_bert_cls_multiseed/seed_${SEED} \
    --results_csv ./output_bert_cls_multiseed/seed_${SEED}/test_results.csv \
    --output_dir ./output_bert_cls_interpretability_multiseed/seed_${SEED} \
    --n_steps 32

  python -m baselines.run_bert_cls_faithfulness \
    --model_dir ./output_bert_cls_multiseed/seed_${SEED} \
    --results_csv ./output_bert_cls_multiseed/seed_${SEED}/test_results.csv \
    --attributions_csv ./output_bert_cls_interpretability_multiseed/seed_${SEED}/integrated_gradients_token_attributions_clean.csv \
    --output_dir ./output_bert_cls_interpretability_multiseed/seed_${SEED}/faithfulness_positive \
    --ranking positive

  python -m baselines.run_bert_cls_faithfulness \
    --model_dir ./output_bert_cls_multiseed/seed_${SEED} \
    --results_csv ./output_bert_cls_multiseed/seed_${SEED}/test_results.csv \
    --attributions_csv ./output_bert_cls_interpretability_multiseed/seed_${SEED}/integrated_gradients_token_attributions_clean.csv \
    --output_dir ./output_bert_cls_interpretability_multiseed/seed_${SEED}/faithfulness_absolute \
    --ranking absolute

done

python -m baselines.summarize_bert_cls_multiseed \
  --train_root ./output_bert_cls_multiseed \
  --interp_root ./output_bert_cls_interpretability_multiseed \
  --output_dir ./output_bert_cls_interpretability_multiseed_summary \
  --seeds 0,1,2,3,4
