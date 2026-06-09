#!/bin/bash

set -e

SEEDS=(0 1 2 3 4)

TRAIN_ROOT="./output_multiseed"
INTERP_ROOT="./output_interpretability_multiseed"

TRAIN_MODEL_DIR="./model/bert-base-uncased"
POSTHOC_MODEL_DIR="./model"

RUN_ATTRIBUTION=${RUN_ATTRIBUTION:-1}
RUN_FAITHFULNESS=${RUN_FAITHFULNESS:-1}

mkdir -p "$TRAIN_ROOT"
mkdir -p "$INTERP_ROOT"

for SEED in "${SEEDS[@]}"; do
  echo "============================================================"
  echo "Running seed ${SEED}"
  echo "============================================================"

  SEED_TRAIN_ROOT="${TRAIN_ROOT}/seed_${SEED}"
  SEED_INTERP_ROOT="${INTERP_ROOT}/seed_${SEED}"

  if [ -d "$SEED_TRAIN_ROOT" ]; then
    mv "$SEED_TRAIN_ROOT" "${SEED_TRAIN_ROOT}_archive_$(date +%Y%m%d_%H%M%S)"
  fi

  mkdir -p "$SEED_TRAIN_ROOT"
  mkdir -p "$SEED_INTERP_ROOT"

  echo "Training seed ${SEED}..."

  python -u prompt_finetune.py \
    --project_root ./ \
    --logs_root "${SEED_TRAIN_ROOT}/" \
    --off_line_model_dir "$TRAIN_MODEL_DIR" \
    --data_dir ./data/ \
    --src_data ccc_train_all \
    --trg_data adress-train_all \
    --trg_test_data adress-test_all \
    --model bert \
    --model_name bert-base-uncased \
    --template_type manual \
    --verbalizer_type manual \
    --template_id 7 \
    --meta 2 \
    --seed "$SEED" \
    --gpu_num 0 \
    --batch_size 4 \
    --trg_batch_size 4 \
    --num_epochs 10 \
    --ce_class_weights \
    --no_tensorboard \
    --save_wrapped_inputs \
    --save_hidden_states \
    --save_attentions \
    > "${SEED_TRAIN_ROOT}/train_seed_${SEED}.log" 2>&1

  CHECKPOINT_DIR=$(find "$SEED_TRAIN_ROOT" -type d -path "*/checkpoints" | sort | tail -n 1)

  if [ -z "$CHECKPOINT_DIR" ]; then
    echo "ERROR: No checkpoints directory found for seed ${SEED}"
    exit 1
  fi

  CKPT="${CHECKPOINT_DIR}/best-checkpoint.ckpt"
  RESULTS_CSV="${CHECKPOINT_DIR}/epoch-1/test_results.csv"
  INTERP_PT="${CHECKPOINT_DIR}/epoch-1/test_results_interpretability.pt"

  echo "Checkpoint directory: $CHECKPOINT_DIR"

  if [ ! -f "$CKPT" ]; then
    echo "ERROR: Missing checkpoint: $CKPT"
    exit 1
  fi

  if [ ! -f "$RESULTS_CSV" ]; then
    echo "ERROR: Missing test results: $RESULTS_CSV"
    exit 1
  fi

  if [ ! -f "$INTERP_PT" ]; then
    echo "ERROR: Missing interpretability tensor: $INTERP_PT"
    exit 1
  fi

  echo "Running post-hoc metrics and representation probing for seed ${SEED}..."

  python -m interpretability.run_interpretability \
    --results_csv "$RESULTS_CSV" \
    --interpretability_pt "$INTERP_PT" \
    --output_dir "$SEED_INTERP_ROOT"

  if [ "$RUN_ATTRIBUTION" = "1" ]; then
    echo "Running Integrated Gradients attribution for seed ${SEED}..."

    python -m interpretability.run_attribution \
      --project_root ./ \
      --off_line_model_dir "$POSTHOC_MODEL_DIR" \
      --model bert \
      --model_name bert-base-uncased \
      --template_id 7 \
      --checkpoint "$CKPT" \
      --results_csv "$RESULTS_CSV" \
      --output_dir "$SEED_INTERP_ROOT" \
      --n_steps 32
  fi

  if [ "$RUN_FAITHFULNESS" = "1" ]; then
    ATTR_CSV="${SEED_INTERP_ROOT}/integrated_gradients_token_attributions_clean.csv"

    if [ ! -f "$ATTR_CSV" ]; then
      echo "ERROR: Missing attribution file for faithfulness: $ATTR_CSV"
      exit 1
    fi

    echo "Running faithfulness with positive ranking for seed ${SEED}..."

    python -m interpretability.run_faithfulness \
      --project_root ./ \
      --off_line_model_dir "$POSTHOC_MODEL_DIR" \
      --model bert \
      --model_name bert-base-uncased \
      --template_id 7 \
      --checkpoint "$CKPT" \
      --results_csv "$RESULTS_CSV" \
      --attributions_csv "$ATTR_CSV" \
      --output_dir "${SEED_INTERP_ROOT}/faithfulness_positive" \
      --ranking positive

    echo "Running faithfulness with absolute ranking for seed ${SEED}..."

    python -m interpretability.run_faithfulness \
      --project_root ./ \
      --off_line_model_dir "$POSTHOC_MODEL_DIR" \
      --model bert \
      --model_name bert-base-uncased \
      --template_id 7 \
      --checkpoint "$CKPT" \
      --results_csv "$RESULTS_CSV" \
      --attributions_csv "$ATTR_CSV" \
      --output_dir "${SEED_INTERP_ROOT}/faithfulness_absolute" \
      --ranking absolute
  fi

  echo "Finished seed ${SEED}"
done

echo "All seeds finished."
