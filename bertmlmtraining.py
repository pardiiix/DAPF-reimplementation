import argparse
import os

from datasets import Dataset, DatasetDict
from safetensors import torch
from transformers import AutoModelForMaskedLM, DataCollatorForLanguageModeling
from transformers import AutoTokenizer
from argparse import Namespace
from loguru import logger
import pandas as pd
import collections
import numpy as np
from transformers import Trainer

import sys
from transformers import default_data_collator
from transformers import TrainingArguments
from safetensors.torch import load_model, save_model
from safetensors.torch import load_model, save_model

#create a args parser with required arguments.
from FewShotSampler import  FewShotSampler

parser = argparse.ArgumentParser("")
parser.add_argument("--project_root",  default='./')
parser.add_argument("--logs_root", default='./output/', help="whether to turn off the dropout in the freezed model. Set to true to turn off.")
parser.add_argument("--off_line_model_dir",default='./model/')
parser.add_argument("--src_data",default = "participant_all_ccc_transcript_cut.csv")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--no_tensorboard", default=True,action="store_true")
#parser.add_argument("--trg_data",default = "adress-train_all")
parser.add_argument("--trg_data",default = 'adress-train_all.csv')
parser.add_argument("--trg_test_data",default = "adress-test_all.csv")
parser.add_argument("--get_attention",default = False)
parser.add_argument("--MAX_EPOCH",type=int,default=25)
parser.add_argument("--WARMUP_TYPE",default="linear")
parser.add_argument("--plm_eval_mode", action="store_true", help="whether to turn off the dropout in the freezed model. Set to true to turn off.")
parser.add_argument("--tune_plm",default=False, action="store_true")
parser.add_argument("--freeze_verbalizer_plm", default=False,action = "store_true")
parser.add_argument("--zero_shot", action="store_true")
parser.add_argument("--few_shot_n", type=int, default = 100)
parser.add_argument("--no_training", default=False, action="store_true")
parser.add_argument("--run_evaluation",default=True,action="store_true")
parser.add_argument("--model", type=str, default='bert', help="The plm to use e.g. t5-base, roberta-large, bert-base, emilyalsentzer/Bio_ClinicalBERT")
parser.add_argument("--model_name", type=str, default='bert-base')
parser.add_argument("--data_dir", type=str,default="./data/")
parser.add_argument("--data_not_saved",default=False,action="store_true")
parser.add_argument("--scripts_path", type=str, default="template/")
parser.add_argument("--max_steps", default=50000, type=int)
parser.add_argument("--plm_lr", type=float, default=1e-05)
parser.add_argument("--plm_warmup_steps", type=float, default=5)
parser.add_argument("--warmup_step_prompt", type=int, default=5)
parser.add_argument("--num_epochs", default=10, type=int)
parser.add_argument("--batch_size",default=16, type=int)
parser.add_argument("--trg_batch_size",default=16, type=int)
parser.add_argument("--init_from_vocab", default=False,action="store_true")
# parser.add_argument("--eval_every_steps", type=int, default=100)
parser.add_argument("--soft_token_num", type=int, default=16)
# parser.add_argument("--optimizer", type=str, default="adamw")
parser.add_argument("--optimizer", type=int, default=0)
parser.add_argument("--gradient_accum_steps", type = int, default = 1)
# parser.add_argument("--dev_run",action="store_true")
parser.add_argument("--gpu_num", type=int, default = 0)
# parser.add_argument("--balance_data", action="store_true") # whether to downsample data to majority class
parser.add_argument("--mode",default="joint", type=str) # whether to train with both src and target dataset
parser.add_argument("--sampler_weights", action="store_true") # apply weights to weighted data sampler
parser.add_argument("--training_size", type=str, default="full") # or fewshot or zero
parser.add_argument("--no_ckpt",default=False, type=bool)
parser.add_argument("--last_ckpt", action="store_true",default=True)
parser.add_argument("--crossvalidation", default=False, type=bool)
parser.add_argument("--val_file_dir", default='latest_tmp_dir/five_fold.json', type=str)
parser.add_argument("--chunk_size", type=int, default=128)
parser.add_argument("--wwm_probability", type=float, default=0.15)


args: Namespace = parser.parse_args()
# if args.baseline:
#     prompt_model=get_baseline_model(args.model,model_dict[args.model_name],load_model_path=baseline_plm)
# model = AutoModelForMaskedLM.from_pretrained(os.path.join(args.off_line_model_dir,args.model_name))
# tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.off_line_model_dir,args.model_name))
# if args.last_ckpt:
#     load_model(model, f"{args.logs_root}mlm/{args.model_name}-finetuned/")
#     print(model)
    # sys.exit()
if args.num_epochs==28:
    model = AutoModelForMaskedLM.from_pretrained(os.path.join(args.off_line_model_dir,""))#last3rd
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.off_line_model_dir,""))

elif args.num_epochs==29:
    model = AutoModelForMaskedLM.from_pretrained(f"{args.logs_root}mlm/{args.model_name}-last3rd_finetuned")
    tokenizer = AutoTokenizer.from_pretrained(f"{args.logs_root}mlm/{args.model_name}-last3rd_finetuned")
elif args.num_epochs==30:
    model = AutoModelForMaskedLM.from_pretrained(f"{args.logs_root}mlm/{args.model_name}-last2_finetuned")
    tokenizer = AutoTokenizer.from_pretrained(f"{args.logs_root}mlm/{args.model_name}-last2_finetuned")
logger.info(f" arguments provided were: {args}")
columns = ['text', 'ad']
args.logs_root = args.logs_root.rstrip("/") + "/"
args.project_root = args.project_root.rstrip("/") + "/"
wwm_probability = args.wwm_probability

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=args.wwm_probability)
def whole_word_masking_data_collator(features):
    for feature in features:
        word_ids = feature.pop("word_ids")

        # Create a map between words and corresponding token indices
        mapping = collections.defaultdict(list)
        current_word_index = -1
        current_word = None
        for idx, word_id in enumerate(word_ids):
            if word_id is not None:
                if word_id != current_word:
                    current_word = word_id
                    current_word_index += 1
                mapping[current_word_index].append(idx)

        # Randomly mask words
        mask = np.random.binomial(1, wwm_probability, (len(mapping),))
        input_ids = feature["input_ids"]
        labels = feature["labels"]
        new_labels = [-100] * len(labels)
        for word_id in np.where(mask)[0]:
            word_id = word_id.item()
            for idx in mapping[word_id]:
                new_labels[idx] = labels[idx]
                input_ids[idx] = tokenizer.mask_token_id
        feature["labels"] = new_labels

    return default_data_collator(features)
def tokenize_function(examples):
    result = tokenizer(examples["text"])
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result


def group_texts(examples):
    # Concatenate all texts
    print(type(examples))
    print(type(examples.keys()))
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys() if isinstance(examples[k], list)}
    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // chunk_size) * chunk_size
    # Split by chunks of max_len
    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # Create a new labels column
    result["labels"] = result["input_ids"].copy()
    return result
# Use batched=True to activate fast multithreading!
if args.mode=='joint':
    sre_data_file = os.path.join(args.data_dir, args.src_data)  # src_data_file will be there w/wo DA during train
    trg_data_file = os.path.join(args.data_dir, args.trg_data)  # src_data_file will be there w/wo DA during train
    src_data=pd.read_csv(sre_data_file)
    # src_data=src_data.loc[:,columns]
    trg_data=pd.read_csv(trg_data_file)
    # trg_data = trg_data.loc[:,columns]
    if args.training_size == "fewshot":
        logger.warning(f"Will be performing few shot learning.")
        # create the few_shot sampler for when we want to run training and testing with few shot learning
        # not implemented for DA yet
        support_sampler = FewShotSampler(num_examples_per_label=args.few_shot_n, also_sample_dev=False)


        trg_data = support_sampler(trg_data, seed=args.seed)  # for DA only change the

    src_data = src_data.loc[:, columns]
    trg_data = trg_data.loc[:, columns]
    trg_data=trg_data.reset_index()
    print('trg data size')
    print(trg_data)
    dataset=pd.concat([src_data,trg_data],axis=0)
    dataset = Dataset.from_pandas(dataset)
    Dict={'train':{'text':dataset['text'],'ad':dataset['ad']}}
    dataset = DatasetDict()
    # using your `Dict` object
    for k, v in Dict.items():
        dataset[k] = Dataset.from_dict(v)

else:
    sre_data_file = os.path.join(args.data_dir, args.src_data)  # src_data_file will be there w/wo DA during train
    src_data=pd.read_csv(sre_data_file)
    dataset=src_data.loc[:,columns]
    dataset = Dataset.from_pandas(dataset)

print(dataset)
tokenized_datasets = dataset.map(
    tokenize_function, batched=True, remove_columns=columns
)
print(tokenized_datasets)
chunk_size = args.chunk_size

lm_datasets = tokenized_datasets.map(group_texts, batched=True)
print(lm_datasets)

# Show the training loss with every epoch
logging_steps = len(lm_datasets['train']) // args.batch_size
#epoch_num=28  last3rd
#epoch_num=29 last2
#epoch_num=30 last
if 'roberta' in args.model:
    train_size = 13_00  # roberta
    epoch_num=28
else:
    train_size = 10_00#bert
    epoch_num = 28

test_size = int(0.01 * train_size)
print(train_size)
print(test_size)
downsampled_dataset = lm_datasets["train"].train_test_split(
    train_size=train_size, test_size=test_size, seed=42
)
print('final train dataset size')
print(len(downsampled_dataset['train']))
if args.num_epochs==28:
    output_dir=f"{args.logs_root}mlm/{args.model_name}-last3rd_finetuned"
    epoch_num_run=args.num_epochs
elif args.num_epochs==29:
    output_dir=f"{args.logs_root}mlm/{args.model_name}-last2_finetuned"
    epoch_num_run = 1 #how many times to run the training
elif args.num_epochs==30:
    output_dir=f"{args.logs_root}mlm/{args.model_name}-last_finetuned"
    epoch_num_run = 1  # how many times to run the training
# print(downsampled_dataset)
epoch_num_run
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=epoch_num_run,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    fp16=True,
    logging_steps=logging_steps,
)

if not args.no_training:
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=downsampled_dataset["train"],
        eval_dataset=downsampled_dataset["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    trainer.train()

    trainer.save_model(output_dir) #worked

    # torch.save(trainer.model.state_dict(), './output/mlm/')

    # trainer._save_checkpoint(trainer.model)