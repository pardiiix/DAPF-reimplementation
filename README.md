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









