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
---

## Robustness analysis: cross-architecture probing, attribution, and faithfulness figures

Run the following commands **from the `DAPF-main` repository root** on the `robustness` branch, with the project Python environment activated. These commands consume **existing CSV results**: they do not retrain models, recompute Integrated Gradients, or re-run probes.

### Model variants and saved results

| Model | Saved five-seed interpretability root | Layer-wise representations |
| --- | --- | --- |
| DAPF-RoBERTa | `./output_roberta_interpretability` | CLS and supervised MASK |
| RoBERTa-CLS | `./output_roberta_cls_interpretability_multiseed` | CLS only |
| RoBERTa-CLS+Prompt | `./output_roberta_cls_prompt_interpretability_multiseed` | CLS and auxiliary MASK |

For each model, seeds 0–4 contain `layerwise_probe_results.csv` either directly in the seed folder or under `layerwise_probing/`. DAPF layer-wise files use the column `score_mean`; the two CLS-based pipelines use `mean_score`. The plotting script supports both formats.

### 1. Compare layer-wise representation probing

```bash
python -m interpretability.plot_roberta_layerwise_comparison \
  --dapf-root ./output_roberta_interpretability \
  --cls-root ./output_roberta_cls_interpretability_multiseed \
  --prompt-root ./output_roberta_cls_prompt_interpretability_multiseed \
  --out-dir ./output_roberta_comparison/figures
```

Key outputs, each as PNG and PDF:

- `roberta_layerwise_cls_mask` — three panels: DAPF-RoBERTa (CLS/supervised MASK), plain RoBERTa-CLS (CLS), and RoBERTa-CLS+Prompt (CLS/auxiliary MASK).
- `roberta_cls_across_models` — comparison of CLS probes in all three models.
- `roberta_mask_minus_cls_by_layer` — paired MASK-minus-CLS macro-F1 within the two models that contain a MASK position.
- `roberta_final_layer_probe_comparison` — all five available final-layer representations, including plain RoBERTa-CLS.
- `roberta_dapf_layerwise_cls_mask`, `roberta_cls_layerwise_cls`, and `roberta_cls_prompt_layerwise_cls_mask` — per-model plots.

The script also saves `roberta_layerwise_probe_by_seed.csv`, `roberta_layerwise_mean_sd.csv`, `roberta_mask_minus_cls_by_seed.csv`, `roberta_mask_minus_cls_mean_sd.csv`, and `roberta_final_layer_mean_sd.csv`.

The probing curves are **mean logistic-regression cross-validated diagnosis macro-F1 across seeds 0–4**, with error bars indicating the **sample SD across the five seed-level means**. Probing measures label recoverability from frozen representations, **not the predictive accuracy of the model's trained classification head**.

### 2. Compare DAPF-BERT and DAPF-RoBERTa token attribution directions

```bash
python -m interpretability.plot_dapf_attribution_comparison \
  --bert-csv ./output_interpretability_multiseed_summary/attribution_aggregated/cross_seed_token_attribution_summary.csv \
  --roberta-csv ./output_roberta_interpretability_summary/attribution_aggregated/cross_seed_token_attribution_summary.csv \
  --out-dir ./output_roberta_comparison/figures
```

Key outputs (PNG/PDF):

- `dapf_bert_roberta_ig_sign_scatter` — mean signed IG scores for shared normalized token types.
- `dapf_ig_sign_agreement_thresholds` — sign agreement as the minimum absolute mean IG magnitude is increased in both models.

The script also saves `dapf_ig_shared_tokens.csv` and `dapf_ig_sign_agreement_thresholds.csv`.

Five-seed coverage filtering requires `n_seeds == 5`, `total_count >= 10`, and `mean_n_samples_per_seed >= 5` **within each backbone**. The observed overlap comprised **156** shared token types: **69 (44.2%)** agreed in mean attribution sign and **87 (55.8%)** disagreed. Agreement at minimum mean-attribution magnitudes 0, 0.005, 0.01, 0.02, and 0.05 was **44.2% (n=156), 45.2% (n=104), 45.8% (n=72), 50.0% (n=20), and 100.0% (n=2)**, respectively. The highest-threshold result is based on only two tokens.

**Interpretation:** this is a descriptive cross-backbone attribution-sensitivity comparison, not a controlled clinical biomarker analysis. Tokenization, IG reference construction, and model objectives can differ. Attribution signs are relative to the respective model's dementia output and chosen reference.

### 3. Compare overall RoBERTa deletion/insertion faithfulness

```bash
python -m interpretability.plot_roberta_faithfulness_comparison \
  --dapf-root ./output_roberta_interpretability \
  --cls-root ./output_roberta_cls_interpretability_multiseed \
  --prompt-root ./output_roberta_cls_prompt_interpretability_multiseed \
  --out-dir ./output_roberta_comparison/figures
```

Key outputs (PNG/PDF):

- `roberta_faithfulness_positive_deletion`
- `roberta_faithfulness_positive_insertion`
- `roberta_faithfulness_absolute_deletion`
- `roberta_faithfulness_absolute_insertion`

Numeric outputs: `roberta_faithfulness_per_seed.csv` and `roberta_faithfulness_mean_sd.csv`.

All effects refer to **dementia probability**, not predicted-class probability:

- **Deletion:** `P(dementia, original) − P(dementia, perturbed)`.
- **Insertion:** `P(dementia, restored) − P(dementia, masked baseline)`.

A positive value means the change is in the expected direction for a token selected as positively contributing to the dementia output; a negative value means the reverse. Absolute-IG ranking combines positive and negative contributions and should not be interpreted as a directional test in the same way. Report five-seed means ± sample SD, acknowledging that the same 48 participants are reused across seeds.

### 4. Stratify positive-IG faithfulness by true diagnosis

```bash
python -m interpretability.plot_roberta_faithfulness_by_diagnosis \
  --dapf-root ./output_roberta_interpretability \
  --cls-root ./output_roberta_cls_interpretability_multiseed \
  --prompt-root ./output_roberta_cls_prompt_interpretability_multiseed \
  --out-dir ./output_roberta_comparison/figures \
  --ranking positive
```

Key outputs (PNG/PDF):

- `roberta_faithfulness_positive_by_diagnosis` — four-panel figure with healthy-control and dementia groups, each showing deletion and insertion effects.

Numeric outputs:

- `roberta_faithfulness_diagnosis_seed_means.csv`
- `roberta_faithfulness_diagnosis_mean_sd.csv`
- `roberta_faithfulness_intervention_counts.csv`

Diagnosis labels are `0 = healthy control`, `1 = dementia`. Each seed's effect is first averaged across the **24 participants** in each diagnosis group, then averaged across five seeds; error bars are the **sample SD of five seed-level group means**, not SD across 120 independent observations.

Observed mean signed effects at 50% requested perturbation:

| Model | Healthy deletion | Dementia deletion | Healthy insertion | Dementia insertion |
| --- | ---: | ---: | ---: | ---: |
| DAPF-RoBERTa | -0.0129 | -0.0737 | -0.1083 | -0.0084 |
| RoBERTa-CLS | -0.0078 | -0.0327 | +0.0031 | +0.0068 |
| RoBERTa-CLS+Prompt | -0.0040 | -0.0091 | +0.0014 | +0.0101 |

**Intervention-count comparability caveat:** fractions are defined relative to each pipeline's available/ranked token positions, so the requested 20% or 50% need not correspond to an equal number of perturbed tokens across architectures. Inspect `roberta_faithfulness_intervention_counts.csv` alongside plots before interpreting model-to-model effect magnitudes. Perturbation also produces potentially out-of-distribution inputs. These curves demonstrate sensitivity under the specified intervention, not validated clinical explanations.

### Figure selection for reports

Suggested main-text figures: `roberta_layerwise_cls_mask`, `roberta_mask_minus_cls_by_layer`, `dapf_bert_roberta_ig_sign_scatter`, and `roberta_faithfulness_positive_by_diagnosis`. Suggested appendix figures: `dapf_ig_sign_agreement_thresholds`, `roberta_faithfulness_absolute_deletion`, `roberta_faithfulness_absolute_insertion`, and `roberta_cls_across_models`. The three-model final-layer bar chart can substitute for, rather than duplicate, parts of the layer-wise results.

All plotting commands save figures to `./output_roberta_comparison/figures/` as `.png` and `.pdf`. The plot scripts belong in `interpretability/` (Python); `scripts/` is reserved for shell scripts in this project.
