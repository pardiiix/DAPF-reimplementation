# DAPF
Officical Implementation of: "Domain Adaptation via Prompt Learning for Alzheimer’s Detection", accepeted in EMNLP-Findings 2024.
<img width="926" alt="image" src="https://github.com/user-attachments/assets/9c0e01d5-2a28-4f44-b375-97b8e079ec05">

#Install


Run "pip install -r /path/to/requirements.txt" 


#Domain-Adaptive Prompt based Finetuning Command

After downloading the DAPF repository and running requirements.txt to install packages, you can run the following commands in the parent directory of DAPF directory. We have used OpenPrompt framework for prompt finetune the PLMs. I have downloaded the code of OpenPrompt code directly instead of installing the packages and made some changes in some files for experiment purpose. 

Before running the run_prompt_finetune.py or run_prompt_finetune_test.py in the following instruction, you'll have to define the project_root, logs_root, off_line_model_dir, data_dir configurations in your scripts. These configuration should be set to 1) the parent directory of your prompt_ad_code folder; 2) the directory to store your output (model or results); 3) the directory you store pre-trained model downloaded from huggingface; 4) the directory you store ADReSS data (csv file), respectively.
--project_root /parent/directory/DAPF \
--logs_root /directory/to/store/your/output \
--off_line_model_dir /model \      
--data_dir /directory/you/store/ADReSS/data \

After setting up the packages and downloading the PLMs in ./model/ directory, the training and test command is: you can run_prompt_finetune.py and run_switchprompt.py

#DATASETS

For the domain adaptation experiments, we adopted:

   - the task-based benchmark dataset ADReSS2020 (https://dementia.talkbank.org/ADReSS-2020/), with its train and test split
   -  the conversational dataset Carolina Conversation Collection (CCC) (https://carolinaconversations.musc.edu/ccc/help/access)
     
Datasets access need to be taken. the dataset format is in the ./data/ folder. In cross validation for ADReSS -> CCC experiments, we adopt 5 fold cross validtion (CV), with validation split stored in DAPF/latest_tmp_dir/five_fold.json

#Ensemble Output

To get specific PLM's ensemble output based on different prompt template, command:

python post_process_vote.py rand_test_merge ./output/ model_name

For the ensembling the output for the CV experiemnts result:

python post_process_vote_cv.py rand_cv_emg ./output/ model_name

model_name can be "bert-base-uncased" or "roberta-base"

To get the BERT+RoBRETa feature combination results, you can run the following command:
for CCC -> ADReSS experiment:

python post_process_vote.py rand_test_robbertmg ./output/path_to_save/ roberta-base

For  ADReSS -> CCC  experiment (CV):
python post_process_vote_cv.py rand_cv_robbertmg ./output/path_to_save/ bert-base-uncased


#Manual Template selection

The different template positioning is given in template/manual_tempalte.txt file. 
N.B. : If template 7 is added , template in line 8 is selected

#Citation:

If you find this work helpful, please cite the following in your publication, thanks!

@inproceedings
{farzana-parde-2024-domain,
    title = "Domain Adaptation via Prompt Learning for {A}lzheimer`s Detection",
    author = "Farzana, Shahla  and
      Parde, Natalie",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2024",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-emnlp.937/",
    doi = "10.18653/v1/2024.findings-emnlp.937",
    pages = "15963--15976",
    
}



Run:
```

source .venv/bin/activate

nohup python -u prompt_finetune.py \
  --project_root ./ \
  --logs_root ./output/ \
  --off_line_model_dir ./model/t5-large \
  --data_dir ./data/ \
  --src_data ccc_train_all \
  --trg_data adress-train_all \
  --trg_test_data adress-test_all \
  --model t5 \
  --model_name t5-large \
  --template_type manual \
  --verbalizer_type manual \
  --template_id 7 \
  --meta 2 \
  --seed 0 \
  --gpu_num 0 \
  --batch_size 4 \
  --trg_batch_size 4 \
  --num_epochs 10 \
  --ce_class_weights \
  --no_tensorboard \
  > output/train_1epoch.log 2>&1 &
```





---

# Interpretability Extension: DAPF for CCC → ADReSS

This repository branch includes an interpretability-oriented extension of the original DAPF pipeline. The goal is to preserve the original domain-adaptive prompt-learning setup while saving additional artifacts needed for post-hoc interpretability analysis.

The main experiment uses the CCC → ADReSS adaptation setting:

* Source training data: `ccc_train_all`
* Target training data: `adress-train_all`
* Target test data: `adress-test_all`
* Backbone: `bert-base-uncased`
* Template type: manual
* Template ID: 7
* Verbalizer type: manual
* Seed: 0
* Batch size: 4
* Epochs: 10

The interpretability extension saves:

* prompt-wrapped tokenized inputs;
* attention masks;
* prediction labels and probabilities;
* final-layer `[CLS]` hidden states;
* final-layer `[MASK]` hidden states;
* optional attention outputs;
* post-hoc prediction metrics;
* representation probing results;
* Integrated Gradients attribution outputs;
* deletion/insertion faithfulness diagnostics.

## Activate Environment

Before running any experiments:

```bash
source .venv/bin/activate
```

## Optional: Archive Previous Outputs Before Re-running

If an output folder already exists with the same configuration, archive it before re-running so that new results are not mixed with older files.

```bash
mkdir -p ./archived_outputs

if [ -d ./output/bert-base-uncased_tempmanual7_verbmanual_epoch10_optimadamw_stk16_100_domain2_bs4_prlr0.5_joint_cvFalse/version_0 ]; then
  mv ./output/bert-base-uncased_tempmanual7_verbmanual_epoch10_optimadamw_stk16_100_domain2_bs4_prlr0.5_joint_cvFalse/version_0 \
     ./archived_outputs/bert_seed0_template7_epoch10_bs4_hidden_attention_$(date +%Y%m%d_%H%M%S)
fi
```

This step is not required, but it is recommended when reproducing experiments.

## Main BERT Interpretability Training Run

The following command trains the BERT-based DAPF model on CCC + ADReSS train and evaluates on ADReSS test. It also saves wrapped inputs, hidden states, and attentions.

```bash
nohup python -u prompt_finetune.py \
  --project_root ./ \
  --logs_root ./output/ \
  --off_line_model_dir ./model/bert-base-uncased \
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
  --seed 0 \
  --gpu_num 0 \
  --batch_size 4 \
  --trg_batch_size 4 \
  --num_epochs 10 \
  --ce_class_weights \
  --no_tensorboard \
  --save_wrapped_inputs \
  --save_hidden_states \
  --save_attentions \
  > output/ccc_to_adress_bert_template7_seed0_bs4_hidden_attention.log 2>&1 &
```

Explanation of the interpretability flags:

* `--save_wrapped_inputs`: saves the prompt-wrapped `input_ids` and `attention_mask` into the test results CSV.
* `--save_hidden_states`: extracts and saves final-layer `[CLS]` and `[MASK]` hidden-state vectors.
* `--save_attentions`: stores final-layer attention summaries when available.
* `--ce_class_weights`: applies class weighting during training.
* `--no_tensorboard`: disables TensorBoard logging.

## Check Generated Outputs

After training finishes, check that the expected output files were created:

```bash
find ./output/bert-base-uncased_tempmanual7_verbmanual_epoch10_optimadamw_stk16_100_domain2_bs4_prlr0.5_joint_cvFalse/version_0/checkpoints \
  -name "best-checkpoint.ckpt" -o \
  -name "test_results.csv" -o \
  -name "*interpretability.pt"
```

Expected important files include:

```text
best-checkpoint.ckpt
epoch-1/test_results.csv
epoch-1/test_results_interpretability.pt
epoch0/test_results.csv
epoch0/test_results_interpretability.pt
...
epoch9/test_results.csv
epoch9/test_results_interpretability.pt
```

The `epoch-1` folder corresponds to the final evaluation using the best checkpoint.

## Verify Saved Interpretability Tensor Shapes

For the ADReSS test set, the expected hidden-state tensor shape is:

```text
torch.Size([48, 768])
```

This means 48 test transcripts and 768-dimensional BERT hidden-state vectors.

## Post-Hoc Metrics, Calibration, and Representation Probing

After training, run the post-hoc interpretability summary script:

```bash
python -m interpretability.run_interpretability \
  --results_csv ./output/bert-base-uncased_tempmanual7_verbmanual_epoch10_optimadamw_stk16_100_domain2_bs4_prlr0.5_joint_cvFalse/version_0/checkpoints/epoch-1/test_results.csv \
  --interpretability_pt ./output/bert-base-uncased_tempmanual7_verbmanual_epoch10_optimadamw_stk16_100_domain2_bs4_prlr0.5_joint_cvFalse/version_0/checkpoints/epoch-1/test_results_interpretability.pt \
  --output_dir ./output_interpretability/ccc_to_adress_bert_template7_seed0_bs4_best_attention
```

This script saves:

```text
prediction_metrics.csv
representation_probe_results.csv
```

The prediction metrics include:

* balanced accuracy;
* macro precision;
* macro recall;
* macro-F1;
* weighted precision;
* weighted recall;
* weighted-F1;
* AUROC;
* Expected Calibration Error.

Expected Calibration Error measures how closely predicted confidence matches empirical accuracy. Lower ECE indicates better calibration.

The representation probing file reports whether diagnosis labels can be recovered from frozen BERT hidden states using a simple logistic-regression probe. The probe is run using stratified five-fold cross-validation. The two saved representation types are:

* final-layer `[CLS]` hidden states;
* final-layer `[MASK]` hidden states.

The `[MASK]` representation is especially important for prompt-based classification because it is the prediction position used by the manual prompt.

## Inspect Representation Probing Results

```bash
cat ./output_interpretability/ccc_to_adress_bert_template7_seed0_bs4_best_attention/representation_probe_results.csv
```

The file has the following format:

```text
representation,target,score_mean,score_std,scores
```

Example interpretation:

* `score_mean`: mean cross-validated macro-F1;
* `score_std`: standard deviation across folds;
* `scores`: individual fold scores.

## Integrated Gradients Attribution

Token-level attribution was implemented using Layer Integrated Gradients.

Run attribution over the full ADReSS test set:

```bash
python -m interpretability.run_attribution \
  --project_root ./ \
  --off_line_model_dir ./model \
  --model bert \
  --model_name bert-base-uncased \
  --template_id 7 \
  --checkpoint ./output/bert-base-uncased_tempmanual7_verbmanual_epoch10_optimadamw_stk16_100_domain2_bs4_prlr0.5_joint_cvFalse/version_0/checkpoints/best-checkpoint.ckpt \
  --results_csv ./output/bert-base-uncased_tempmanual7_verbmanual_epoch10_optimadamw_stk16_100_domain2_bs4_prlr0.5_joint_cvFalse/version_0/checkpoints/epoch-1/test_results.csv \
  --output_dir ./output_interpretability/ccc_to_adress_bert_template7_seed0_bs4_best_attention \
  --n_steps 32
```

This creates:

```text
integrated_gradients_token_attributions.csv
integrated_gradients_token_attributions_clean.csv
top_positive_attributions_clean.csv
top_negative_attributions_clean.csv
```

The raw file contains all tokens, including prompt tokens and special tokens. The clean file removes prompt/special tokens and keeps reportable transcript tokens.

Attribution interpretation:

* positive attribution: token pushes the model toward the dementia class;
* negative attribution: token pushes the model away from the dementia class.

However, token attribution should be interpreted cautiously because token-level attribution may be sensitive to prompt structure, subword tokenization, transcript artifacts, and perturbation behavior.

## Inspect Attribution Outputs

```bash
python - <<'PY'
import pandas as pd

base = "./output_interpretability/ccc_to_adress_bert_template7_seed0_bs4_best_attention"

for name in [
    "integrated_gradients_token_attributions.csv",
    "integrated_gradients_token_attributions_clean.csv",
    "top_positive_attributions_clean.csv",
    "top_negative_attributions_clean.csv",
]:
    path = f"{base}/{name}"
    df = pd.read_csv(path)
    print("\nFILE:", name)
    print("shape:", df.shape)
    print("unique samples:", df["id"].nunique() if "id" in df.columns else "NA")
    print(df.head())
PY
```

## Faithfulness Analysis

Deletion and insertion tests were implemented to evaluate whether token-level attributions are faithful to the model’s prediction behavior.

In deletion, high-attribution tokens are masked. If attribution is faithful, masking dementia-supporting tokens should reduce `P(dementia)`.

In insertion, the transcript begins from a masked baseline, and high-attribution tokens are restored. If attribution is faithful, restoring important tokens should recover `P(dementia)`.

Two ranking strategies are used:

* `positive`: ranks tokens by positive attribution toward dementia;
* `absolute`: ranks tokens by absolute attribution magnitude.

### Faithfulness: Positive Attribution Ranking

```bash
python -m interpretability.run_faithfulness \
  --project_root ./ \
  --off_line_model_dir ./model \
  --model bert \
  --model_name bert-base-uncased \
  --template_id 7 \
  --checkpoint ./output/bert-base-uncased_tempmanual7_verbmanual_epoch10_optimadamw_stk16_100_domain2_bs4_prlr0.5_joint_cvFalse/version_0/checkpoints/best-checkpoint.ckpt \
  --results_csv ./output/bert-base-uncased_tempmanual7_verbmanual_epoch10_optimadamw_stk16_100_domain2_bs4_prlr0.5_joint_cvFalse/version_0/checkpoints/epoch-1/test_results.csv \
  --attributions_csv ./output_interpretability/ccc_to_adress_bert_template7_seed0_bs4_best_attention/integrated_gradients_token_attributions_clean.csv \
  --output_dir ./output_interpretability/ccc_to_adress_bert_template7_seed0_bs4_best_attention/faithfulness_positive \
  --ranking positive
```

### Faithfulness: Absolute Attribution Ranking

```bash
python -m interpretability.run_faithfulness \
  --project_root ./ \
  --off_line_model_dir ./model \
  --model bert \
  --model_name bert-base-uncased \
  --template_id 7 \
  --checkpoint ./output/bert-base-uncased_tempmanual7_verbmanual_epoch10_optimadamw_stk16_100_domain2_bs4_prlr0.5_joint_cvFalse/version_0/checkpoints/best-checkpoint.ckpt \
  --results_csv ./output/bert-base-uncased_tempmanual7_verbmanual_epoch10_optimadamw_stk16_100_domain2_bs4_prlr0.5_joint_cvFalse/version_0/checkpoints/epoch-1/test_results.csv \
  --attributions_csv ./output_interpretability/ccc_to_adress_bert_template7_seed0_bs4_best_attention/integrated_gradients_token_attributions_clean.csv \
  --output_dir ./output_interpretability/ccc_to_adress_bert_template7_seed0_bs4_best_attention/faithfulness_absolute \
  --ranking absolute
```

Each faithfulness run creates:

```text
deletion_curve.csv
insertion_curve.csv
faithfulness_summary.csv
```

## Summarize Faithfulness Results

```bash
python - <<'PY'
import pandas as pd

bases = {
    "positive": "./output_interpretability/ccc_to_adress_bert_template7_seed0_bs4_best_attention/faithfulness_positive",
    "absolute": "./output_interpretability/ccc_to_adress_bert_template7_seed0_bs4_best_attention/faithfulness_absolute",
}

for name, base in bases.items():
    print("\n" + "="*80)
    print("RANKING:", name)

    deletion = pd.read_csv(f"{base}/deletion_curve.csv")
    insertion = pd.read_csv(f"{base}/insertion_curve.csv")

    print("\nDeletion probability_drop by fraction:")
    print(deletion.groupby("fraction")["probability_drop"].agg(["mean", "std", "count"]))

    print("\nInsertion probability_recovered by fraction:")
    print(insertion.groupby("fraction")["probability_recovered"].agg(["mean", "std", "count"]))

    print("\nDeletion by label:")
    print(deletion.groupby(["label", "fraction"])["probability_drop"].mean())

    print("\nInsertion by label:")
    print(insertion.groupby(["label", "fraction"])["probability_recovered"].mean())
PY
```

Interpretation:

* Positive `probability_drop` means masking important tokens reduced dementia probability.
* Negative `probability_drop` means masking important tokens increased dementia probability.
* Positive `probability_recovered` means inserting important tokens recovered dementia probability.
* Negative or near-zero recovery suggests weak faithfulness.

In the current experiment, deletion and insertion results did not show stable global faithfulness for token-level attribution. Therefore, token-level Integrated Gradients should be treated as exploratory rather than as the main interpretability evidence.

## Current Main Findings

The strongest current interpretability evidence comes from representation probing rather than token-level attribution.

The main findings are:

1. The BERT-based DAPF model can be modified to save interpretability artifacts while preserving meaningful CCC → ADReSS classification performance.
2. Saved final-layer hidden states allow post-hoc probing of diagnostic information.
3. Diagnosis information is linearly recoverable from hidden states.
4. The `[MASK]` hidden state is more informative than the `[CLS]` hidden state for diagnosis probing.
5. Token-level Integrated Gradients can be computed, but deletion/insertion tests suggest that the token rankings are not globally faithful.
6. Therefore, representation-level analysis is currently more reliable than token-level attribution for this prompt-based AD detection setting.

## Notes on Reproducibility

The exact output directory depends on the model, template, epoch, batch size, seed, and whether previous versions already exist. The main output directory used in this experiment was:

```text
./output/bert-base-uncased_tempmanual7_verbmanual_epoch10_optimadamw_stk16_100_domain2_bs4_prlr0.5_joint_cvFalse/version_0/checkpoints/
```

The main interpretability output directory was:

```text
./output_interpretability/ccc_to_adress_bert_template7_seed0_bs4_best_attention/
```

If a new run creates `version_1`, `version_2`, or another version number, update the paths in the post-hoc commands accordingly.

## Recommended Next Steps

Future experiments should strengthen the interpretability analysis by running:

* multiple random seeds;
* multiple manual prompt templates;
* layer-wise `[CLS]` vs `[MASK]` probing;
* additional model backbones such as RoBERTa;
* clinical-feature alignment using lexical richness, disfluency, informativeness, and content-unit features;
* error analysis for false positives and false negatives.

These steps are important before treating the results as publication-ready.

---




To run multiseeds:
```
chmod +x scripts/run_multiseed_pipeline.sh
nohup bash scripts/run_multiseed_pipeline.sh 
```

After training is completed, need to run the separate summarization script:
```
python -m interpretability.summarize_multiseed \
  --train_root ./output_multiseed \
  --interp_root ./output_interpretability_multiseed \
  --output_dir ./output_interpretability_multiseed_summary_seed01 \
  --seeds 0,1,2,3,4
```

Aggregated attribution run:
```
python -m interpretability.run_attribution_aggregated \
  --interp_root ./output_interpretability_multiseed \
  --output_dir ./output_interpretability_multiseed_summary/attribution_aggregated \
  --seeds 0,1,2,3,4 \
  --min_count 10 \
  --min_samples 5 \
  --loose_min_count 5 \
  --loose_min_samples 3
```
Layer-wise Representation Probing:
```
for SEED in 0 1 2 3 4; do
  CHECKPOINT_DIR=$(find ./output_multiseed/seed_${SEED} -type d -path "*/checkpoints" | sort | tail -n 1)

  python -m interpretability.run_layerwise_probing \
    --project_root ./ \
    --off_line_model_dir ./model \
    --model bert \
    --model_name bert-base-uncased \
    --template_id 7 \
    --checkpoint "${CHECKPOINT_DIR}/best-checkpoint.ckpt" \
    --results_csv "${CHECKPOINT_DIR}/epoch-1/test_results.csv" \
    --output_dir "./output_interpretability_multiseed/seed_${SEED}/layerwise_probing" \
    --batch_size 8
done
```
Then summarize:
```
python -m interpretability.summarize_layerwise_probing \
  --interp_root ./output_interpretability_multiseed \
  --output_dir ./output_interpretability_multiseed_summary/layerwise_probing \
  --seeds 0,1,2,3,4
```

To run visualizations, find the seed closest to the average F1 score, and run for that seed:

```
python -m interpretability.visualize_representations \
  --results_csv ./output_multiseed/seed_0/bert-base-uncased_tempmanual7_verbmanual_epoch10_optimadamw_stk16_100_domain2_bs4_prlr0.5_joint_cvFalse/version_0/checkpoints/epoch-1/test_results.csv \
  --interpretability_pt ./output_multiseed/seed_0/bert-base-uncased_tempmanual7_verbmanual_epoch10_optimadamw_stk16_100_domain2_bs4_prlr0.5_joint_cvFalse/version_0/checkpoints/epoch-1/test_results_interpretability.pt \
  --output_dir ./output_interpretability_multiseed/seed_0/representation_visualizations \
  --seed 0
```
## Experimental branches

Additional baselines are maintained on separate branches:

- `bert-cls-baseline`: standard BERT sequence-classification baseline under the same CCC → ADReSS data regime.
- `bert-cls-prompt-baseline`: prompt-as-input BERT sequence-classification ablation.

Switch to the corresponding branch before running branch-specific scripts.

After the two branches and experiments are run and completed, run paired error analysis on the results in `paired-error-analysis` branch
using epoch 1 since that was the closest in my case to the mean results. Feel free to change the dapf pattern if different in your case
```
python -m baselines.run_paired_error_analysis_three_models \
  --dapf_root ./output_multiseed \
  --dapf_pattern 'seed_{seed}/bert-base-uncased_tempmanual7_verbmanual_epoch10_optimadamw_stk16_100_domain2_bs4_prlr0.5_joint_cvFalse/version_{seed}/checkpoints/epoch-1/test_results.csv' \
  --bert_cls_root ./output_bert_cls_multiseed \
  --bert_cls_prompt_root ./output_bert_cls_prompt_multiseed \
  --output_dir ./paired_error_analysis \
  --seeds 0,1,2,3,4
  ```

  Later added:

  To run the random label control test:
  ```
  (.venv) pardisranjbar@Pardiss-MBP DAPF-main % for SEED in 0 1 2 3 4; do                        
  CHECKPOINT_DIR=$(find ./output_multiseed/seed_${SEED} -type d -path "*/checkpoints" | sort | tail -n 1)

  python -m interpretability.run_interpretability \
    --results_csv "${CHECKPOINT_DIR}/epoch-1/test_results.csv" \
    --interpretability_pt "${CHECKPOINT_DIR}/epoch-1/test_results_interpretability.pt" \
    --output_dir "./output_interpretability_multiseed_randomlabel/seed_${SEED}"
done
```

To run shap:

```
source .venv/bin/activate

nohup python -u prompt_finetune.py \
  --project_root ./ \
  --logs_root ./output_multiseed/seed_0/ \
  --off_line_model_dir ./model/bert-base-uncased \
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
  --seed 0 \
  --gpu_num 0 \
  --batch_size 4 \
  --trg_batch_size 4 \
  --num_epochs 10 \
  --ce_class_weights \
  --no_tensorboard \
  --no_training \
  --run_partition_shap \
  --shap_max_evals 500 \
  --shap_batch_size 8 \
  --shap_output_dir "output_multiseed/seed_0/partition_shap_final" \
  > output/partition_shap_full_seed0.log 2>&1 &
```

<!-- STUDY2_AUDIT_START -->
## Study 2 documentation and Study 2b setup

This branch retains the BERT multiseed interpretability implementation. Study 2a
compares BERT and RoBERTa on CCC -> ADReSS (seeds 0–4); the newer post-hoc
significance scripts and results live on `paired-error-analysis`, while
`robustness` documents cross-backbone visualizations and faithfulness.

**Study 2b (planned):** preserve DAPF BERT training code and change dataset
arguments only: `--src_data pitt_single_utt`,
`--trg_data participant_all_ccc_transcript_cut`, `--crossvalidation True`,
`--val_file_dir latest_tmp_dir/five_fold.json`, `--val_fold_idx 0..4`.
Use original checkpoint selection and a distinct output directory. No new
validation algorithm or model change is proposed.
<!-- STUDY2_AUDIT_END -->


```
BASE="./output/paper2b_pitt477_ccc105"
mkdir -p "$BASE"

# Verify the full Pitt dataset and CCC folds exist
test -f ./data/pitt/pitt.csv || { echo "Full Pitt CSV missing"; exit 1; }
test -f ./data/ccc_train_all.csv || exit 1
test -f ./latest_tmp_dir/five_fold_ccc105.json || exit 1

if [ -e "$BASE/launcher.log" ]; then
  echo "Existing launcher log found. Refusing to overwrite."
else

nohup caffeinate -i bash -c '
set -euo pipefail

test "$(git branch --show-current)" = "multiseed-interpretability"

BASE="./output/paper2b_pitt477_ccc105"

# Prevent overwriting existing runs
for FOLD in 0 1 2 3 4; do
  for SEED in 0 1 2 3 4; do
    RUN_DIR="$BASE/fold_${FOLD}/seed_${SEED}"

    if [ -e "$RUN_DIR" ]; then
      echo "Existing run: $RUN_DIR"
      exit 1
    fi
  done
done

for FOLD in 0 1 2 3 4; do
  for SEED in 0 1 2 3 4; do

    RUN_DIR="$BASE/fold_${FOLD}/seed_${SEED}"
    mkdir -p "$RUN_DIR"

    echo "======================================"
    echo "Paper 2b FULL PITT: Fold $FOLD, Seed $SEED"
    echo "======================================"

    .venv/bin/python -u prompt_finetune.py \
      --project_root ./ \
      --logs_root "$RUN_DIR/" \
      --off_line_model_dir ./model/bert-base-uncased \
      --data_dir ./data/ \
      --src_data pitt/pitt \
      --trg_data ccc_train_all \
      --crossvalidation True \
      --val_file_dir latest_tmp_dir/five_fold_ccc105.json \
      --val_fold_idx "$FOLD" \
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
      --plm_lr 1e-5 \
      --ce_class_weights \
      --no_tensorboard \
      --save_wrapped_inputs \
      --save_hidden_states \
      --save_attentions \
      > "$RUN_DIR/train.log" 2>&1

    CKPT=(find"RUN_DIR" -type f \
      -name "best-checkpoint_cv1_fold${FOLD}.ckpt" \
      -print -quit)

    RESULTS=(find"RUN_DIR" -type f \
      -path "*/epoch-1/test_results_cv1_fold${FOLD}.csv" \
      -print -quit)

    test -n "$CKPT" || { echo "Missing checkpoint"; exit 1; }
    test -n "$RESULTS" || { echo "Missing predictions"; exit 1; }

    echo "Completed fold $FOLD, seed $SEED"
    echo "Checkpoint: $CKPT"
    echo "Predictions: $RESULTS"

  done
done

echo "ALL 25 FULL-PITT DAPF-BERT RUNS FINISHED"
' > "$BASE/launcher.log" 2>&1 < /dev/null &

echo !>"BASE/launcher.pid"
echo "Training started. PID: (cat"BASE/launcher.pid")"

fi

```


Running interpretability:

```

cd "/Users/pardisranjbar/Desktop/Dementia Detection/projects/final/DAPF-main"
source .venv/bin/activate

(
set -e

TRAIN_ROOT="./output_paper2b/bert/pitt105_to_ccc105"
INTERP_ROOT="./output_interpretability_paper2b/bert/pitt105_to_ccc105"

for FOLD in 0 1 2 3 4; do
  for SEED in 0 1 2 3 4; do

    RUN="$TRAIN_ROOT/fold_${FOLD}/seed_${SEED}"
    OUT="$INTERP_ROOT/fold_${FOLD}/seed_${SEED}"

    if [ -e "$OUT" ]; then
      echo "Output already exists: $OUT"
      exit 1
    fi

    RESULTS=$(find "$RUN" -type f \
      -path "*/epoch-1/test_results_cv1_fold${FOLD}.csv" \
      -print -quit)

    if [ -z "$RESULTS" ]; then
      echo "Missing predictions for fold $FOLD, seed $SEED"
      exit 1
    fi

    TENSORS=$(find "$(dirname "$RESULTS")" \
      -maxdepth 1 -type f -name "*interpretability*.pt" \
      -print -quit)

    if [ -z "$TENSORS" ]; then
      echo "Missing interpretability tensor"
      exit 1
    fi

    mkdir -p "$OUT"

    echo "Analyzing fold $FOLD, seed $SEED"

    if ! python -m interpretability.run_interpretability \
      --results_csv "$RESULTS" \
      --interpretability_pt "$TENSORS" \
      --output_dir "$OUT" \
      > "$OUT/run.log" 2>&1; then

      tail -n 40 "$OUT/run.log"
      exit 1
    fi

    echo "Completed fold $FOLD, seed $SEED"

  done
done

echo "ALL 25 POST-HOC ANALYSES FINISHED"
)
```

## Summarize final-layer probing
```

python - <<'PY'
from pathlib import Path
import pandas as pd

ROOT = Path(
    "output_interpretability_paper2b/bert/pitt105_to_ccc105"
)
OUT = Path(
    "output_interpretability_paper2b/bert/pitt105_to_ccc105_summary"
)

rows = []

for seed in range(5):
    for fold in range(5):
        path = (
            ROOT / f"fold_{fold}" / f"seed_{seed}"
            / "representation_probe_results.csv"
        )

        df = pd.read_csv(path).set_index("representation")

        assert set(df.index) == {
            "cls_hidden_states", "mask_hidden_states"
        }, path

        cls = float(df.loc["cls_hidden_states", "score_mean"])
        mask = float(df.loc["mask_hidden_states", "score_mean"])

        rows.append({
            "seed": seed,
            "fold": fold,
            "cls_macro_f1": cls,
            "mask_macro_f1": mask,
            "mask_minus_cls": mask - cls,
        })

fold_results = pd.DataFrame(rows)

seed_results = (
    fold_results.groupby("seed", as_index=False)[
        ["cls_macro_f1", "mask_macro_f1", "mask_minus_cls"]
    ].mean()
)

print("\n=== FOLD-LEVEL RESULTS ===")
print(fold_results.to_string(
    index=False, float_format=lambda x: f"{x:.4f}"
))

print("\n=== PER-SEED MEANS ACROSS 5 FOLDS ===")
print(seed_results.to_string(
    index=False, float_format=lambda x: f"{x:.4f}"
))

print("\n=== OVERALL MEAN ± SD ACROSS 5 SEEDS ===")

for col in ["cls_macro_f1", "mask_macro_f1", "mask_minus_cls"]:
    print(
        f"{col}: {seed_results[col].mean():.4f}"
        f" ± {seed_results[col].std(ddof=1):.4f}"
    )

print(
    "\nSeeds where MASK > CLS:",
    int((seed_results.mask_minus_cls > 0).sum()),
    "/ 5"
)

if OUT.exists():
    print("\nSummary directory already exists; no files overwritten.")
else:
    OUT.mkdir(parents=True)
    fold_results.to_csv(
        OUT / "final_layer_fold_results.csv", index=False
    )
    seed_results.to_csv(
        OUT / "final_layer_seed_results.csv", index=False
    )
    print("\nSaved results to:", OUT)
PY

```

## Run layer-wise probing on all 25 models

```
bash <<'BASH'
set -euo pipefail

TRAIN_ROOT="./output_paper2b/bert/pitt105_to_ccc105"
INTERP_ROOT="./output_interpretability_paper2b/bert/pitt105_to_ccc105"

for FOLD in 0 1 2 3 4; do
  for SEED in 0 1 2 3 4; do

    RUN="$TRAIN_ROOT/fold_${FOLD}/seed_${SEED}"
    OUT="$INTERP_ROOT/fold_${FOLD}/seed_${SEED}/layerwise_probing"

    if [ -f "$OUT/completed.ok" ]; then
      echo "Already complete: fold $FOLD, seed $SEED"
      continue
    fi

    if [ -e "$OUT" ]; then
      echo "Incomplete or existing output: $OUT"
      echo "Stopping to avoid overwriting."
      exit 1
    fi

    CHECKPOINT=(find"RUN" -type f \
      -name "best-checkpoint_cv1_fold${FOLD}.ckpt" \
      -print -quit)

    RESULTS=(find"RUN" -type f \
      -path "*/epoch-1/test_results_cv1_fold${FOLD}.csv" \
      -print -quit)

    if [ -z "CHECKPOINT"]||[-z"RESULTS" ]; then
      echo "Missing checkpoint or results: fold $FOLD, seed $SEED"
      exit 1
    fi

    mkdir -p "$OUT"

    echo "======================================"
    echo "Layer-wise probing: Fold $FOLD, Seed $SEED"
    echo "======================================"

    if ! python -m interpretability.run_layerwise_probing \
      --project_root ./ \
      --off_line_model_dir ./model \
      --model bert \
      --model_name bert-base-uncased \
      --template_id 7 \
      --checkpoint "$CHECKPOINT" \
      --results_csv "$RESULTS" \
      --output_dir "$OUT" \
      --batch_size 8 \
      > "$OUT/run.log" 2>&1; then

      tail -n 60 "$OUT/run.log"
      exit 1
    fi

    touch "$OUT/completed.ok"

    echo "Completed fold $FOLD, seed $SEED"

  done
done

echo "ALL 25 LAYER-WISE PROBING RUNS FINISHED"
BASH

```

## Aggregate layer-wise probing

```

cd "/Users/pardisranjbar/Desktop/Dementia Detection/projects/final/DAPF-main"
source .venv/bin/activate

python - <<'PY'
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("output_interpretability_paper2b/bert/pitt105_to_ccc105")
OUT = Path("output_interpretability_paper2b/bert/pitt105_to_ccc105_summary")

outputs = [
    OUT / "layerwise_fold_results.csv",
    OUT / "layerwise_seed_results.csv",
    OUT / "layerwise_summary.csv",
    OUT / "layerwise_probe_f1.png",
]

assert not any(p.exists() for p in outputs), "Summary outputs already exist; refusing to overwrite."

all_results = []

for seed in range(5):
    for fold in range(5):
        path = (
            ROOT / f"fold_{fold}" / f"seed_{seed}"
            / "layerwise_probing" / "layerwise_probe_results.csv"
        )

        assert path.exists(), f"Missing: {path}"
        df = pd.read_csv(path)

        required = {"layer", "representation", "score_mean"}
        if not required.issubset(df.columns):
            raise ValueError(
                f"Unexpected columns in {path}: {df.columns.tolist()}"
            )

        df = df[["layer", "representation", "score_mean"]].copy()

        def normalize_rep(value):
            name = str(value).lower()
            if "cls" in name:
                return "CLS"
            if "mask" in name:
                return "MASK"
            raise ValueError(f"Unknown representation: {value}")

        df["representation"] = df["representation"].map(normalize_rep)
        df["score_mean"] = pd.to_numeric(df["score_mean"])
        df["seed"] = seed
        df["fold"] = fold

        assert len(df) == 24, (path, len(df))
        assert df.groupby(["layer", "representation"]).size().eq(1).all()
        assert set(df["representation"]) == {"CLS", "MASK"}
        assert df["layer"].nunique() == 12
        assert df["score_mean"].between(0, 1).all()

        all_results.append(df)

fold_results = pd.concat(all_results, ignore_index=True)

seed_results = (
    fold_results
    .groupby(["seed", "layer", "representation"], as_index=False)
    ["score_mean"].mean()
)

summary = (
    seed_results
    .groupby(["layer", "representation"])["score_mean"]
    .agg(["mean", "std"])
    .reset_index()
    .sort_values(["layer", "representation"])
)

print("\n=== LAYER-WISE MACRO-F1: MEAN ACROSS 5 SEEDS ===")
table = summary.pivot(
    index="layer", columns="representation", values="mean"
)
print(table.to_string(float_format=lambda x: f"{x:.4f}"))

print("\n=== LAYER-WISE MASK MINUS CLS ===")
print(
    (table["MASK"] - table["CLS"]).to_string(
        float_format=lambda x: f"{x:+.4f}"
    )
)

print("\n=== PEAK LAYER ===")
for rep in ["CLS", "MASK"]:
    peak = table[rep].idxmax()
    print(f"{rep}: layer {peak}, macro-F1 {table.loc[peak, rep]:.4f}")

OUT.mkdir(parents=True, exist_ok=True)

fold_results.to_csv(outputs[0], index=False)
seed_results.to_csv(outputs[1], index=False)
summary.to_csv(outputs[2], index=False)

fig, ax = plt.subplots(figsize=(9, 5))

for rep in ["CLS", "MASK"]:
    subset = summary[summary.representation == rep]
    ax.plot(
        subset["layer"],
        subset["mean"],
        marker="o",
        label=f"DAPF [{rep}]"
    )

ax.set_xlabel("BERT layer")
ax.set_ylabel("Probe macro-F1")
ax.set_title("DAPF-BERT: Pitt-105 to CCC-105")
ax.set_xticks(sorted(summary["layer"].unique()))
ax.legend()
ax.grid(alpha=0.25)

fig.tight_layout()
fig.savefig(outputs[3], dpi=200)
plt.close(fig)

print("\nPASS: Aggregated all 25 runs.")
print("Saved summary and plot to:", OUT)
PY
```

## Run Integrated Gradients on all 25 models

```

cd "/Users/pardisranjbar/Desktop/Dementia Detection/projects/final/DAPF-main"
source .venv/bin/activate

caffeinate -i bash <<'BASH'
set -euo pipefail

TRAIN_ROOT="./output_paper2b/bert/pitt105_to_ccc105"
INTERP_ROOT="./output_interpretability_paper2b/bert/pitt105_to_ccc105"

for FOLD in 0 1 2 3 4; do
  for SEED in 0 1 2 3 4; do

    RUN="$TRAIN_ROOT/fold_${FOLD}/seed_${SEED}"
    OUT="$INTERP_ROOT/fold_${FOLD}/seed_${SEED}/integrated_gradients"

    if [ -f "$OUT/integrated_gradients_token_attributions_clean.csv" ] &&
       [ -f "$OUT/integrated_gradients_token_attributions.csv" ] &&
       [ -f "$OUT/top_positive_attributions_clean.csv" ] &&
       [ -f "$OUT/top_negative_attributions_clean.csv" ]; then
      echo "Already complete: fold $FOLD, seed $SEED"
      continue
    fi

    if [ -e "$OUT" ]; then
      echo "Existing incomplete output: $OUT"
      exit 1
    fi

    CHECKPOINT=$(find "$RUN" -type f \
      -name "best-checkpoint_cv1_fold${FOLD}.ckpt" \
      -print -quit)

    RESULTS=$(find "$RUN" -type f \
      -path "*/epoch-1/test_results_cv1_fold${FOLD}.csv" \
      -print -quit)

    if [ -z "$CHECKPOINT" ] || [ -z "$RESULTS" ]; then
      echo "Missing checkpoint or predictions: fold $FOLD, seed $SEED"
      exit 1
    fi

    mkdir -p "$OUT"

    echo "======================================"
    echo "Integrated Gradients: Fold $FOLD, Seed $SEED"
    echo "======================================"

    if ! python -m interpretability.run_attribution \
      --project_root ./ \
      --off_line_model_dir ./model \
      --model bert \
      --model_name bert-base-uncased \
      --template_id 7 \
      --checkpoint "$CHECKPOINT" \
      --results_csv "$RESULTS" \
      --output_dir "$OUT" \
      --n_steps 32 \
      > "$OUT/run.log" 2>&1; then

      tail -n 60 "$OUT/run.log"
      exit 1
    fi

    test -f "$OUT/integrated_gradients_token_attributions_clean.csv"

    echo "Completed fold $FOLD, seed $SEED"

  done
done

echo "ALL 25 INTEGRATED GRADIENTS RUNS FINISHED"
BASH
```

## Run faithfulness for all 25 models

```

cd "/Users/pardisranjbar/Desktop/Dementia Detection/projects/final/DAPF-main"
source .venv/bin/activate

caffeinate -i bash <<'BASH'
set -euo pipefail

TRAIN_ROOT="./output_paper2b/bert/pitt105_to_ccc105"
INTERP_ROOT="./output_interpretability_paper2b/bert/pitt105_to_ccc105"

for FOLD in 0 1 2 3 4; do
  for SEED in 0 1 2 3 4; do

    RUN="$TRAIN_ROOT/fold_${FOLD}/seed_${SEED}"
    IG="$INTERP_ROOT/fold_${FOLD}/seed_${SEED}/integrated_gradients_scaffold_filtered"

    CHECKPOINT=$(find "$RUN" -type f \
      -name "best-checkpoint_cv1_fold${FOLD}.ckpt" -print -quit)

    RESULTS=$(find "$RUN" -type f \
      -path "*/epoch-1/test_results_cv1_fold${FOLD}.csv" -print -quit)

    ATTR="$IG/integrated_gradients_token_attributions_clean.csv"

    test -n "$CHECKPOINT"
    test -n "$RESULTS"
    test -f "$ATTR"

    for RANKING in positive absolute; do

      OUT="$IG/faithfulness_${RANKING}"

      if [ -f "$OUT/deletion_curve.csv" ] &&
         [ -f "$OUT/insertion_curve.csv" ] &&
         [ -f "$OUT/faithfulness_summary.csv" ]; then
        echo "Already complete: fold $FOLD, seed $SEED, $RANKING"
        continue
      fi

      if [ -e "$OUT" ]; then
        echo "Existing incomplete output: $OUT"
        exit 1
      fi

      mkdir -p "$OUT"

      echo "======================================"
      echo "Faithfulness: Fold $FOLD, Seed $SEED, $RANKING"
      echo "======================================"

      if ! python -m interpretability.run_faithfulness \
        --project_root ./ \
        --off_line_model_dir ./model \
        --model bert \
        --model_name bert-base-uncased \
        --template_id 7 \
        --checkpoint "$CHECKPOINT" \
        --results_csv "$RESULTS" \
        --attributions_csv "$ATTR" \
        --output_dir "$OUT" \
        --ranking "$RANKING" \
        > "$OUT/run.log" 2>&1; then

        tail -n 60 "$OUT/run.log"
        exit 1
      fi

      echo "Completed fold $FOLD, seed $SEED, $RANKING"

    done
  done
done

echo "ALL 50 FAITHFULNESS ANALYSES FINISHED"
BASH
```

## Aggregate faithfulness results

Because Paper 2b has five CCC evaluation folds, we'll first average the fold-specific results within each seed. All five folds contain 21 participants.

```

cd "/Users/pardisranjbar/Desktop/Dementia Detection/projects/final/DAPF-main"
source .venv/bin/activate

python - <<'PY'
from pathlib import Path
import pandas as pd

ROOT = Path("output_interpretability_paper2b/bert/pitt105_to_ccc105")
OUT = Path("output_interpretability_paper2b/bert/pitt105_to_ccc105_summary/faithfulness_scaffold_filtered")

outputs = [
    OUT / "faithfulness_fold_results.csv",
    OUT / "faithfulness_seed_results.csv",
    OUT / "faithfulness_summary.csv",
]

assert not any(p.exists() for p in outputs), \
    "Summary files already exist; refusing to overwrite."

records = []

for seed in range(5):
    for fold in range(5):
        for ranking in ["positive", "absolute"]:

            base = (
                ROOT / f"fold_{fold}" / f"seed_{seed}"
                / "integrated_gradients_scaffold_filtered"
                / f"faithfulness_{ranking}"
            )

            for test, filename, metric in [
                ("deletion", "deletion_curve.csv", "probability_drop"),
                ("insertion", "insertion_curve.csv", "probability_recovered"),
            ]:
                path = base / filename
                assert path.exists(), f"Missing: {path}"

                df = pd.read_csv(path)

                required = {"fraction", metric}
                assert required.issubset(df.columns), (
                    path, df.columns.tolist()
                )

                df["fraction"] = pd.to_numeric(df["fraction"])
                df["effect"] = pd.to_numeric(df[metric])

                assert df[["fraction", "effect"]].notna().all().all()
                assert df["fraction"].between(0, 1).all()

                fold_scores = (
                    df.groupby("fraction", as_index=False)["effect"]
                    .agg(["mean", "count"])
                    .reset_index()
                )

                for _, row in fold_scores.iterrows():
                    records.append({
                        "seed": seed,
                        "fold": fold,
                        "ranking": ranking,
                        "test": test,
                        "fraction": row["fraction"],
                        "effect": row["mean"],
                        "n_rows": int(row["count"]),
                    })

fold_results = pd.DataFrame(records)

assert fold_results.groupby(
    ["ranking", "test", "fraction"]
).size().eq(25).all(), "Incomplete fold/seed coverage"

seed_results = (
    fold_results.groupby(
        ["seed", "ranking", "test", "fraction"],
        as_index=False
    )["effect"].mean()
)

summary = (
    seed_results.groupby(
        ["ranking", "test", "fraction"]
    )["effect"]
    .agg(["mean", "std", "count"])
    .reset_index()
    .sort_values(["ranking", "test", "fraction"])
)

assert summary["count"].eq(5).all()

print("\n=== FAITHFULNESS: MEAN ± SD ACROSS FIVE SEEDS ===")

for ranking in ["positive", "absolute"]:
    print(f"\n=== {ranking.upper()} RANKING ===")

    subset = summary[summary.ranking == ranking]

    for test in ["deletion", "insertion"]:
        print(f"\n{test.upper()}")

        rows = subset[subset.test == test]

        for _, row in rows.iterrows():
            print(
                f"Fraction {row['fraction']:.0%}: "
                f"{row['mean']:+.4f} ± {row['std']:.4f}"
            )

print("\n=== OBSERVATIONS PER FOLD AND FRACTION ===")
print(
    fold_results.groupby(["ranking", "test"])["n_rows"]
    .agg(["min", "max"])
    .to_string()
)

OUT.mkdir(parents=True, exist_ok=True)

fold_results.to_csv(outputs[0], index=False)
seed_results.to_csv(outputs[1], index=False)
summary.to_csv(outputs[2], index=False)

print("\nPASS: All 50 analyses aggregated.")
print("Saved to:", OUT)
PY
```



## Aggregate Integrated Gradients across Folds and Seeds

Paper 2b uses Pitt-105 as the source domain and CCC-105 as the target domain, with five CCC cross-validation folds and five random seeds.

Integrated Gradients is computed separately for each fold-specific DAPF-BERT model. Before aggregating token attributions, we remove the DAPF prompt scaffold so that only transcript tokens contribute to the reported rankings.

### Step 1 — Remove prompt-scaffold tokens and combine attribution files by seed

The original cleaned attribution files retained some tokens from the opening domain prompt, including `health`, `and`, `well`, and `##bei`.

These tokens occupy positions 6–9 in the prompt-wrapped inputs. The transcript starts at position 12, while the final diagnosis prompt begins immediately before `[MASK]`.

We filter by token position, not token text, to preserve genuine transcript occurrences of the same words.

The following command validates the prompt boundaries, saves corrected attribution files separately for all 25 fold/seed runs, and combines the five folds into one attribution file per seed.

```bash
cd "/Users/pardisranjbar/Desktop/Dementia Detection/projects/final/DAPF-main"
source .venv/bin/activate

python - <<'PY'
from pathlib import Path
import pandas as pd

ROOT = Path("output_interpretability_paper2b/bert/pitt105_to_ccc105")
SUMMARY = Path("output_interpretability_paper2b/bert/pitt105_to_ccc105_summary")

NAME = "integrated_gradients_token_attributions_clean.csv"

prefix = [
    "[CLS]", "participant", "'", "s", "narration",
    "on", "health", "and", "well", "##bei", "##ng", "."
]

pending = []

for seed in range(5):
    frames = []
    all_ids = set()

    for fold in range(5):
        base = (
            ROOT / f"fold_{fold}" / f"seed_{seed}"
            / "integrated_gradients"
        )

        raw = pd.read_csv(
            base / "integrated_gradients_token_attributions.csv",
            dtype={"id": str}
        )

        clean = pd.read_csv(base / NAME, dtype={"id": str})

        boundaries = {}

        for pid, group in raw.groupby("id"):
            tokens = dict(zip(
                group.token_position.astype(int),
                group.token.astype(str)
            ))

            assert [tokens.get(i) for i in range(12)] == prefix, (
                "Unexpected opening prompt", seed, fold, pid
            )

            mask_positions = [
                pos for pos, token in tokens.items()
                if token == "[MASK]"
            ]

            assert len(mask_positions) == 1

            m = mask_positions[0]

            assert [
                tokens.get(i) for i in range(m - 3, m + 3)
            ] == [
                "patient", "has", "diagnosis",
                "[MASK]", ".", "[SEP]"
            ], ("Unexpected closing prompt", seed, fold, pid)

            boundaries[pid] = m - 3

        assert set(clean.id) == set(boundaries)

        clean = clean[
            (clean.token_position >= 12) &
            (clean.token_position < clean.id.map(boundaries))
        ].copy()

        assert clean.id.nunique() == 21
        assert not (all_ids & set(clean.id))

        all_ids.update(clean.id.unique())
        frames.append(clean)

        destination = (
            ROOT / f"fold_{fold}" / f"seed_{seed}"
            / "integrated_gradients_scaffold_filtered"
            / NAME
        )

        pending.append((destination, clean))

    assert len(all_ids) == 105

    combined = pd.concat(frames, ignore_index=True)

    destination = (
        SUMMARY / "ig_per_seed_scaffold_filtered"
        / f"seed_{seed}" / NAME
    )

    pending.append((destination, combined))

    print(
        f"Seed {seed}: {len(all_ids)} participants, "
        f"{len(combined)} transcript-token rows"
    )

existing = [str(p) for p, _ in pending if p.exists()]
assert not existing, f"Refusing to overwrite: {existing}"

for path, df in pending:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

print("\nPASS: All 25 corrected fold files and 5 per-seed files saved.")
print("Original attribution files remain unchanged.")
PY
```

Expected verification:

- Five random seeds, each covering all 105 CCC participants.
- Twenty-five corrected fold-level attribution files.
- Five combined per-seed attribution files.
- No modifications to the original Integrated Gradients results.

### Step 2 — Run Paper 1's original attribution aggregation module

Use the original `interpretability.run_attribution_aggregated` module with the corrected, transcript-only attribution files.

The token frequency thresholds remain identical to Paper 1.

```bash
python -m interpretability.run_attribution_aggregated \
  --interp_root ./output_interpretability_paper2b/bert/pitt105_to_ccc105_summary/ig_per_seed_scaffold_filtered \
  --output_dir ./output_interpretability_paper2b/bert/pitt105_to_ccc105_summary/attribution_aggregated_scaffold_filtered \
  --seeds 0,1,2,3,4 \
  --min_count 10 \
  --min_samples 5 \
  --loose_min_count 5 \
  --loose_min_samples 3
```

This produces pooled and cross-seed attribution summaries, including top-positive, top-negative, and highest-absolute-attribution token rankings.

The main reporting thresholds are at least 10 token occurrences and presence in at least five transcripts. The relaxed thresholds are retained for sensitivity analysis.

### Step 3 — Use corrected attributions for faithfulness analysis

Deletion and insertion faithfulness tests must use the corrected fold-level cleaned attribution files:

```text
output_interpretability_paper2b/bert/pitt105_to_ccc105/
  fold_{fold}/seed_{seed}/
    integrated_gradients_scaffold_filtered/
      integrated_gradients_token_attributions_clean.csv
```

The faithfulness methodology remains unchanged from Paper 1:

- Positive and absolute attribution rankings.
- Deletion and insertion perturbation tests.
- The same perturbation fractions and model checkpoints.
- Aggregation across five CCC folds and five random seeds.

Original attribution and faithfulness outputs are retained separately for reproducibility.

## Partition SHAP — Paper 2b

Use representative seed 0, selected because its classification
macro-F1 (0.9037) is closest to the five-seed mean (0.9000).

For Pitt-105 → CCC-105, evaluate the five fold-specific
checkpoints for seed 0 to cover all 105 held-out CCC participants.

Use the existing Paper 1 Partition SHAP implementation without
retraining DAPF.

```

cd "/Users/pardisranjbar/Desktop/Dementia Detection/projects/final/DAPF-main"
source .venv/bin/activate

python - <<'PY'
from pathlib import Path
import shutil

p = Path("prompt_finetune.py")
source = p.read_text()

old = '''elif not do_training:
    logger.warning("No training will be performed!")'''

addition = '''

    if args.crossvalidation:
        trg_validation_data_loader = PromptDataLoader(
            dataset=trg_dataset["validation"],
            tokenizer=tokenizer,
            template=mytemplate,
            tokenizer_wrapper_class=WrapperClass,
            max_seq_length=max_seq_l,
            decoder_max_length=3,
            batch_size=args.batch_size,
        )'''

assert source.count(old) == 1, "Unexpected source structure; stopping."

block = source.split(old, 1)[1].split(
    "# run on test set if desired", 1
)[0]

if "trg_validation_data_loader = PromptDataLoader(" in block:
    print("Validation-loader fix already present.")
else:
    updated = source.replace(old, old + addition, 1)
    compile(updated, str(p), "exec")

    backup = Path("prompt_finetune.py.before_shap_cv_fix.bak")
    assert not backup.exists(), "Backup already exists; stopping."

    shutil.copy2(p, backup)
    p.write_text(updated)
    print("Applied evaluation-only validation-loader fix.")
    print("Backup:", backup)
PY

caffeinate -i bash <<'BASH'
set -euo pipefail

TRAIN_ROOT="./output_paper2b/bert/pitt105_to_ccc105"
INTERP_ROOT="./output_interpretability_paper2b/bert/pitt105_to_ccc105"
STAGE_ROOT="./output_paper2b_shap_staging"

test "$(git branch --show-current)" = "multiseed-interpretability"
test -f "./data/pitt/pitt_paper2b_105.csv"
test -f "./data/ccc_train_all.csv"
test -f "./latest_tmp_dir/five_fold_ccc105.json"

for FOLD in 0 1 2 3 4; do

  RUN="$TRAIN_ROOT/fold_${FOLD}/seed_0"
  STAGE="$STAGE_ROOT/fold_${FOLD}/seed_0"
  OUT="$INTERP_ROOT/fold_${FOLD}/seed_0/partition_shap"

  CHECKPOINT=$(find "$RUN" -type f \
    -name "best-checkpoint_cv1_fold${FOLD}.ckpt" \
    -print -quit)

  test -n "$CHECKPOINT"
  test -f "$CHECKPOINT"

  if [ -d "$STAGE" ] && [ -d "$OUT" ] &&
     [ -f "$STAGE/shap.log" ] &&
     grep -Fq "NameError: name 'trg_validation_data_loader' is not defined" "$STAGE/shap.log"; then

    test ! -e "${STAGE}_failed_loader"
    test ! -e "${OUT}_failed_loader"

    mv "$STAGE" "${STAGE}_failed_loader"
    mv "$OUT" "${OUT}_failed_loader"

    echo "Archived previous failed attempt for fold $FOLD"
  fi

  if [ -e "$STAGE" ] || [ -e "$OUT" ]; then
    echo "Existing SHAP staging or output for fold $FOLD"
    echo "Stopping to avoid overwriting."
    exit 1
  fi

  REL="${CHECKPOINT#"$RUN"/}"
  test "$REL" != "$CHECKPOINT"

  mkdir -p "$(dirname "$STAGE/$REL")"
  mkdir -p "$OUT"

  ln -s "$(pwd)/${CHECKPOINT#./}" "$STAGE/$REL"

  echo "======================================"
  echo "Partition SHAP: Fold $FOLD, Seed 0"
  echo "======================================"

  if ! python -u prompt_finetune.py \
    --project_root ./ \
    --logs_root "$STAGE/" \
    --off_line_model_dir ./model/bert-base-uncased \
    --data_dir ./data/ \
    --src_data pitt/pitt_paper2b_105 \
    --trg_data ccc_train_all \
    --crossvalidation True \
    --val_file_dir ./latest_tmp_dir/five_fold_ccc105.json \
    --val_fold_idx "$FOLD" \
    --model bert \
    --model_name bert-base-uncased \
    --template_type manual \
    --verbalizer_type manual \
    --template_id 7 \
    --meta 2 \
    --seed 0 \
    --gpu_num 0 \
    --batch_size 4 \
    --trg_batch_size 4 \
    --num_epochs 10 \
    --plm_lr 1e-5 \
    --ce_class_weights \
    --no_tensorboard \
    --no_training \
    --run_partition_shap \
    --shap_max_samples 21 \
    --shap_max_evals 500 \
    --shap_batch_size 8 \
    --shap_output_dir "$OUT" \
    > "$STAGE/shap.log" 2>&1; then

    echo "SHAP failed for fold $FOLD"
    tail -n 80 "$STAGE/shap.log"
    exit 1
  fi

  echo "Completed SHAP: fold $FOLD, seed 0"

done

echo "ALL FIVE PARTITION SHAP FOLDS FINISHED"
BASH
```

### Aggregate and rank SHAP tokens
```

python - <<'PY'
from pathlib import Path
import pandas as pd

ROOT = Path(
    "output_interpretability_paper2b/bert/"
    "pitt105_to_ccc105_summary"
)

SHAP = ROOT / "partition_shap_seed0"
IG = ROOT / "attribution_aggregated_scaffold_filtered" / "pooled_ranked"

comparisons = [
    (
        "POSITIVE",
        SHAP / "top_positive_shap_tokens.csv",
        IG / "pooled_main_top_positive_mean_attribution.csv",
    ),
    (
        "NEGATIVE",
        SHAP / "top_negative_shap_tokens.csv",
        IG / "pooled_main_top_negative_mean_attribution.csv",
    ),
]

for name, shap_path, ig_path in comparisons:
    shap = pd.read_csv(shap_path)
    ig = pd.read_csv(ig_path)

    shap_tokens = (
        shap["token_norm"].astype(str).str.lower().head(20).tolist()
    )
    ig_tokens = (
        ig["token_norm"].astype(str).str.lower().head(20).tolist()
    )

    shared = sorted(set(shap_tokens) & set(ig_tokens))

    print(f"\n=== {name} ATTRIBUTIONS ===")
    print("\nSHAP top 20:")
    print(", ".join(shap_tokens))

    print("\nIntegrated Gradients top 20:")
    print(", ".join(ig_tokens))

    print("\nShared tokens:")
    print(", ".join(shared) if shared else "None")

    print(f"\nOverlap: {len(shared)} / 20")

print("\nPASS: Descriptive SHAP–IG comparison complete.")
PY
```

### Compare SHAP with Integrated Gradients

```

python - <<'PY'
from pathlib import Path
import pandas as pd

ROOT = Path(
    "output_interpretability_paper2b/bert/"
    "pitt105_to_ccc105_summary"
)

SHAP = ROOT / "partition_shap_seed0"
IG = ROOT / "attribution_aggregated_scaffold_filtered" / "pooled_ranked"

comparisons = [
    (
        "POSITIVE",
        SHAP / "top_positive_shap_tokens.csv",
        IG / "pooled_main_top_positive_mean_attribution.csv",
    ),
    (
        "NEGATIVE",
        SHAP / "top_negative_shap_tokens.csv",
        IG / "pooled_main_top_negative_mean_attribution.csv",
    ),
]

for name, shap_path, ig_path in comparisons:
    shap = pd.read_csv(shap_path)
    ig = pd.read_csv(ig_path)

    shap_tokens = (
        shap["token_norm"].astype(str).str.lower().head(20).tolist()
    )
    ig_tokens = (
        ig["token_norm"].astype(str).str.lower().head(20).tolist()
    )

    shared = sorted(set(shap_tokens) & set(ig_tokens))

    print(f"\n=== {name} ATTRIBUTIONS ===")
    print("\nSHAP top 20:")
    print(", ".join(shap_tokens))

    print("\nIntegrated Gradients top 20:")
    print(", ".join(ig_tokens))

    print("\nShared tokens:")
    print(", ".join(shared) if shared else "None")

    print(f"\nOverlap: {len(shared)} / 20")

print("\nPASS: Descriptive SHAP–IG comparison complete.")
PY
```