#!/usr/bin/env bash
set -euo pipefail

mkdir -p ./output_roberta_cls_prompt_multiseed

# This pipeline intentionally runs training only. The attribution and faithfulness
# scripts must be adapted to RoBERTa separately after model evaluation.
for SEED in 0 1 2 3 4; do
  echo "============================================================"
  echo "Training RoBERTa-CLS+Prompt seed ${SEED}"
  echo "Started: $(date)"
  echo "============================================================"

  python -u -m baselines.train_roberta_cls_prompt \
    --project_root ./ \
    --off_line_model_dir ./model/roberta-base \
    --data_dir ./data/ \
    --src_data ccc_train_all \
    --trg_data adress-train_all \
    --trg_test_data adress-test_all \
    --train_mode pooled \
    --prompt_style dapf_template7 \
    --output_dir "./output_roberta_cls_prompt_multiseed/seed_${SEED}" \
    --seed "${SEED}" \
    --batch_size 4 \
    --num_epochs 10 \
    --learning_rate 1e-5 \
    --max_length 512 \
    --ce_class_weights \
    --save_hidden_states

  echo "Completed seed ${SEED}: $(date)"
done

echo "ALL FIVE ROBERTA-CLS-PROMPT TRAINING RUNS COMPLETED SUCCESSFULLY"
