#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Auto-nohup wrapper
# ============================================================
# Run this script normally:
#
#   bash scripts/run_bert_cls_multiseed_full_pipeline.sh
#
# It will relaunch itself under nohup and return immediately.
# Logs will be saved to:
#
#   output_bert_cls_multiseed/run_bert_cls_multiseed_full_pipeline.log
#
# To run in the foreground for debugging:
#
#   BERT_CLS_NOHUP=1 bash scripts/run_bert_cls_multiseed_full_pipeline.sh
# ============================================================

mkdir -p output_bert_cls_multiseed
mkdir -p output_bert_cls_interpretability_multiseed
mkdir -p output_bert_cls_interpretability_multiseed_summary

LOG_FILE="${LOG_FILE:-output_bert_cls_multiseed/run_bert_cls_multiseed_full_pipeline.log}"

if [[ "${BERT_CLS_NOHUP:-0}" != "1" ]]; then
  echo "Launching BERT-CLS full multi-seed pipeline with nohup..."
  echo "Log file: ${LOG_FILE}"
  export BERT_CLS_NOHUP=1
  export PYTHONUNBUFFERED=1

  nohup bash "$0" "$@" > "${LOG_FILE}" 2>&1 &
  PID=$!

  echo "Started background process with PID: ${PID}"
  echo "Watch progress with:"
  echo "  tail -f ${LOG_FILE}"
  exit 0
fi

echo "============================================================"
echo "BERT-CLS full multi-seed pipeline started"
echo "Start time: $(date)"
echo "Log file: ${LOG_FILE}"
echo "============================================================"

SEEDS=(0 1 2 3 4)

for SEED in "${SEEDS[@]}"; do
  echo "============================================================"
  echo "Training BERT-CLS seed ${SEED}"
  echo "Time: $(date)"
  echo "============================================================"

  python -u -m baselines.train_bert_cls \
    --project_root ./ \
    --off_line_model_dir ./model/bert-base-uncased \
    --data_dir ./data/ \
    --src_data ccc_train_all \
    --trg_data adress-train_all \
    --trg_test_data adress-test_all \
    --train_mode pooled \
    --output_dir ./output_bert_cls_multiseed/seed_${SEED} \
    --seed ${SEED} \
    --batch_size 4 \
    --num_epochs 10 \
    --learning_rate 1e-5 \
    --max_length 512 \
    --ce_class_weights \
    --save_hidden_states

  echo "============================================================"
  echo "Running BERT-CLS post-hoc interpretability seed ${SEED}"
  echo "Time: $(date)"
  echo "============================================================"

  python -u -m baselines.run_bert_cls_interpretability \
    --results_csv ./output_bert_cls_multiseed/seed_${SEED}/test_results.csv \
    --interpretability_pt ./output_bert_cls_multiseed/seed_${SEED}/test_results_interpretability.pt \
    --output_dir ./output_bert_cls_interpretability_multiseed/seed_${SEED}

  echo "============================================================"
  echo "Running BERT-CLS Integrated Gradients seed ${SEED}"
  echo "Time: $(date)"
  echo "============================================================"

  python -u -m baselines.run_bert_cls_attribution \
    --model_dir ./output_bert_cls_multiseed/seed_${SEED} \
    --results_csv ./output_bert_cls_multiseed/seed_${SEED}/test_results.csv \
    --output_dir ./output_bert_cls_interpretability_multiseed/seed_${SEED} \
    --n_steps 32 \
    --device auto

  echo "============================================================"
  echo "Running BERT-CLS faithfulness positive seed ${SEED}"
  echo "Time: $(date)"
  echo "============================================================"

  python -u -m baselines.run_bert_cls_faithfulness \
    --model_dir ./output_bert_cls_multiseed/seed_${SEED} \
    --results_csv ./output_bert_cls_multiseed/seed_${SEED}/test_results.csv \
    --attributions_csv ./output_bert_cls_interpretability_multiseed/seed_${SEED}/integrated_gradients_token_attributions_clean.csv \
    --output_dir ./output_bert_cls_interpretability_multiseed/seed_${SEED}/faithfulness_positive \
    --ranking positive \
    --device auto

  echo "============================================================"
  echo "Running BERT-CLS faithfulness absolute seed ${SEED}"
  echo "Time: $(date)"
  echo "============================================================"

  python -u -m baselines.run_bert_cls_faithfulness \
    --model_dir ./output_bert_cls_multiseed/seed_${SEED} \
    --results_csv ./output_bert_cls_multiseed/seed_${SEED}/test_results.csv \
    --attributions_csv ./output_bert_cls_interpretability_multiseed/seed_${SEED}/integrated_gradients_token_attributions_clean.csv \
    --output_dir ./output_bert_cls_interpretability_multiseed/seed_${SEED}/faithfulness_absolute \
    --ranking absolute \
    --device auto

  echo "============================================================"
  echo "Completed seed ${SEED}"
  echo "Time: $(date)"
  echo "============================================================"
done

echo "============================================================"
echo "Summarizing BERT-CLS multi-seed results"
echo "Time: $(date)"
echo "============================================================"

python -u -m baselines.summarize_bert_cls_multiseed \
  --train_root ./output_bert_cls_multiseed \
  --interp_root ./output_bert_cls_interpretability_multiseed \
  --output_dir ./output_bert_cls_interpretability_multiseed_summary \
  --seeds 0,1,2,3,4

echo "============================================================"
echo "Aggregating BERT-CLS attributions across seeds"
echo "Time: $(date)"
echo "============================================================"

python -u -m interpretability.run_attribution_aggregated \
  --interp_root ./output_bert_cls_interpretability_multiseed \
  --output_dir ./output_bert_cls_interpretability_multiseed_summary/attribution_aggregated \
  --seeds 0,1,2,3,4 \
  --min_count 10 \
  --min_samples 5 \
  --loose_min_count 5 \
  --loose_min_samples 3

echo "============================================================"
echo "BERT-CLS full multi-seed pipeline complete"
echo "End time: $(date)"
echo "============================================================"