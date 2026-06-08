# copied and modified from  https://github.com/yiwang454/prompt_ad_code
from argparse import Namespace
from typing import Dict

from tqdm import tqdm
from openprompt.data_utils import PROCESSORS
import math
import torch
from openprompt.data_utils.utils import InputExample
import argparse
import numpy as np
import pandas as pd

from openprompt import PromptDataLoader
from openprompt.prompts import ManualVerbalizer, ManualTemplate, SoftVerbalizer, PTRTemplate, PTRVerbalizer, \
    PrefixTuningTemplate

from openprompt.prompts import SoftTemplate, MixedTemplate
from openprompt import PromptForClassification
# from openprompt.utils.logging import logger
from loguru import logger

import time
import os
from datetime import datetime
#from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, roc_auc_score 

from torch.utils.data.sampler import RandomSampler, WeightedRandomSampler

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

import json
import itertools
from collections import Counter

import os 
import sys
# # Kill all processes on GPU 6 and 7
# os.system("""kill $(nvidia-smi | awk '$5=="PID" {p=1} p && $2 >= 6 && $2 < 7 {print $5}')""")

'''
Script to run different setups of prompt learning.
'''

# create a args parser with required arguments.
parser = argparse.ArgumentParser("")
# parser.add_argument("--project_root",  default='./')
# parser.add_argument("--logs_root", default='./output/', help="whether to turn off the dropout in the freezed model. Set to true to turn off.")
# parser.add_argument("--off_line_model_dir",default='')
# parser.add_argument("--data_dir",default = "./data/")
# parser.add_argument("--num_epochs",default = 1)
parser.add_argument("--src_data",default = "participant_all_ccc_transcript_cut")
parser.add_argument("--src_data_mult",default = None)
parser.add_argument("--trg_data",default = "adress-train_all")
parser.add_argument("--trg_test_data",default = "adress-test_all")

# Interpretability options
parser.add_argument("--get_attention", action="store_true")
parser.add_argument("--save_wrapped_inputs", action="store_true")
parser.add_argument("--save_hidden_states", action="store_true")
parser.add_argument("--save_attentions", action="store_true")

parser.add_argument("--U",type=float,default = 1.05)
parser.add_argument("--SOURCE_DOMAINS",default=["health and wellbeing"])
parser.add_argument("--TARGET_DOMAINS",default=["picture description"])
parser.add_argument("--LR_SCHEDULER",default="cosine")
parser.add_argument("--MAX_EPOCH",type=int,default=25)
parser.add_argument("--WARMUP_TYPE",default="linear")
parser.add_argument("--N_CTX",type=int,default=0) #context vector size, fo DA
parser.add_argument("--N_DMX",type=int,default=0)#Domain-Specific Context (DSC) to capture unique features of each domain vector size; for DA
parser.add_argument("--da",type=bool,default=True)
parser.add_argument("--CSC",type=bool,default=False,help="class specific context")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--plm_eval_mode", action="store_true", help="whether to turn off the dropout in the freezed model. Set to true to turn off.")
parser.add_argument("--tune_plm",default=True, action="store_true")
parser.add_argument("--freeze_verbalizer_plm", default=False,action = "store_true")
parser.add_argument("--zero_shot", action="store_true")
parser.add_argument("--few_shot_n", type=int, default = 100)
parser.add_argument("--no_training", default=False, action="store_true")
parser.add_argument("--run_evaluation",default=True,action="store_true")
parser.add_argument("--model", type=str, default='t5', help="The plm to use e.g. t5-base, roberta-large, bert-base, emilyalsentzer/Bio_ClinicalBERT")
parser.add_argument("--model_name", type=str, default='t5-large')
parser.add_argument("--project_root", type=str, help="The project root in the file system, i.e. the absolute path of OpenPrompt")
parser.add_argument("--logs_root", type=str, help="The dir in which project results are stored in, i.e. the absolute path of OpenPrompt")
parser.add_argument("--off_line_model_dir", type=str, help="The dir in which pre-trained model are stored in")
parser.add_argument("--meta", type=int,default=2,help='whether to ad domain info as meta')
parser.add_argument("--template_id", type=int, default = 0)
parser.add_argument("--template_type", type=str, default ="manual")
parser.add_argument("--verbalizer_type", type=str, default ="manual")
parser.add_argument("--manual_type", type=str, default ="A")
parser.add_argument("--data_dir", type=str)
parser.add_argument("--data_not_saved",default=False,action="store_true")
parser.add_argument("--scripts_path", type=str, default="template/")
parser.add_argument("--max_steps", default=50000, type=int)
parser.add_argument("--plm_lr", type=float, default=1e-05)
parser.add_argument("--plm_warmup_steps", type=float, default=5)
parser.add_argument("--prompt_lr", type=float, default=0.5)
parser.add_argument("--warmup_step_prompt", type=int, default=5)
parser.add_argument("--num_epochs", default=10, type=int)
parser.add_argument("--batch_size",default=4, type=int)
parser.add_argument("--trg_batch_size",default=4, type=int)
parser.add_argument("--init_from_vocab", default=True,action="store_true")
# parser.add_argument("--eval_every_steps", type=int, default=100)
parser.add_argument("--soft_token_num", type=int, default=16)
parser.add_argument("--optimizer", type=int, default=0)
parser.add_argument("--gradient_accum_steps", type = int, default = 1)
# parser.add_argument("--dev_run",action="store_true")
parser.add_argument("--gpu_num", type=int, default = 0)
# parser.add_argument("--balance_data", action="store_true") # whether to downsample data to majority class
parser.add_argument("--ce_class_weights", action="store_true")
parser.add_argument("--sampler_weights", action="store_true") # apply weights to weighted data sampler
parser.add_argument("--training_size", type=str, default="full") # or fewshot or zero
parser.add_argument("--no_ckpt",default=False, type=bool)
parser.add_argument("--last_ckpt", action="store_true",default=False)
parser.add_argument("--crossvalidation", default=False, type=bool)
parser.add_argument("--val_file_dir", default='latest_tmp_dir/five_fold.json', type=str)
parser.add_argument("--val_fold_idx", type=int, default=0)
parser.add_argument("--no_tensorboard", default=True,action="store_true")
parser.add_argument("--part_tuning", action="store_true")
parser.add_argument("--tunable_layernum", type=int, default=0)
parser.add_argument("--pause", action="store_true")
parser.add_argument("--pause_threshold", type=str)
parser.add_argument("--transcription", type=str, default='chas')
parser.add_argument("--asr_format", type=int, default=3)
parser.add_argument("--adrc_label_col", type=int, default=0)


parser.add_argument(
        '--sensitivity',
        default=False,
        type=bool,
        help='Run sensitivity trials - investigating the influence of classifier hidden dimension on performance in frozen plm setting.'
    )

# parser.add_argument(
#     '--optimized_run',
#     default=False,
#     type=bool,"
#     help='Run the optimized frozen model after hp search '
# )
# instatiate args and set to variable

args: Namespace = parser.parse_args()
logger.info(f" arguments provided were: {args}")

import random

from openprompt.utils.reproduciblity import set_seed
set_seed(args.seed)

from openprompt.plms.seq2seq import T5TokenizerWrapper, T5LMTokenizerWrapper
from transformers import T5Config, T5Tokenizer, T5ForConditionalGeneration
from openprompt.data_utils.data_sampler import FewShotSampler
from openprompt.plms import load_plm
from prompt_ad_utils import read_input_text_len_control, read_input_no_len_control
import ast
from transformers import  get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup,get_constant_schedule_with_warmup  # use AdamW is a standard practice for transformer
from transformers.optimization import Adafactor, AdafactorSchedule  # use Adafact
max_seq_l =512 #1024  for t5-basecauses oom error
torch.cuda.empty_cache()
class_labels = [
    "healthy",
    "dementia"
]
label_col=['ad','AB42_AB40Positivity','tTau_AB42Positivity','MOCA_impairment','pTau_Positivity']
# label_col=['AB42_AB40Positivity']

optimizer=['adamw','adafactor','sgd']
args.optimizer=optimizer[args.optimizer]
adrc_label_col=label_col[args.adrc_label_col]
if args.training_size=='fewshot':
    args.few_shot_n=args.soft_token_num
elif args.training_size=='full':
    args.few_shot_n = 100

#55 t+d+c
def get_domain_column():
    # according to manual template, domain column name, for running in pod, the tepmate idx strat frm 1,
    # for local or lab gpu the code is as such, template start frm 0
    if args.template_id in [2,3,36,37,25,26,27,28,29,47,55,64,67,71]: #Last ones are for mixed temp
        return 'domain'
    elif args.template_id in [4,5,6,7,35,38,20,21,22,23,24,41,42,43,44,45,46,57,58,59,60,61,62,69,70,73,74]:
        return 'domain1'
    elif args.template_id in [8,9,10,11,12,13,39,40,41,30,31,32,33,34,49,63,65,66,72]:
        return 'domain2'

domain_col=""
if 'manual' in args.template_type or 'mixed' in args.template_type  and args.meta > -1:
    domain_col = get_domain_column()
print(domain_col)
mult=False

if args.src_data_mult is not None:
    mult=True

def normalize_label(label):
    """
    Convert dataset labels to the integer format expected by OpenPrompt.

    Expected mapping:
        healthy/control -> 0
        dementia/AD     -> 1
    """
    label_map = {
        "healthy": 0,
        "control": 0,
        "cn": 0,
        "normal": 0,
        "0": 0,
        0: 0,
        0.0: 0,

        "dementia": 1,
        "ad": 1,
        "alzheimer": 1,
        "alzheimers": 1,
        "alzheimer's": 1,
        "1": 1,
        1: 1,
        1.0: 1,
    }

    if pd.isna(label):
        return None

    if isinstance(label, str):
        cleaned = label.strip().lower()
    else:
        cleaned = label

    if cleaned not in label_map:
        raise ValueError(f"Unknown label value: {label}")

    return label_map[cleaned]


def create_input_example(org_data, include_domain, label_col='ad'):
    # include_domain: whether to include domain information.
    # if > -1, means domain-adaptation experiment; otherwise no domain prompt.
    data_list = []
    label_list = []

    if label_col:
        print(label_col)
        print(org_data.columns.tolist())
        org_data = org_data[~org_data[label_col].isna()]

    for index, data in org_data.iterrows():
        numeric_label = normalize_label(data[label_col])

        if numeric_label is None:
            continue

        if include_domain > -1:
            meta = {
                domain_col: data[domain_col],
            }

            input_example = InputExample(
                text_a=data['text'],
                label=numeric_label,
                meta=meta,
                guid=data["filename"]
            )

        else:
            input_example = InputExample(
                text_a=data['text'],
                label=numeric_label,
                guid=data["filename"]
            )

        data_list.append(input_example)
        label_list.append(numeric_label)

    return data_list, Counter(label_list)

def loading_data_asexample(data_save_dir, sample_size, classes, model, mode='train', validation_dict=None):
    print(mode)
    src_data_file=None
    src_data_mult_file = None
    trg_data_file=None
    raw_src_df=None
    raw_src_mult_df = None
    raw_trg_df=None
    load_src_data_df=None
    load_src_data_mult_df = None
    load_trg_data_df=None
    org_src_data=None
    org_src_mult_data = None
    org_trg_data=None

    if 'cv' in mode:
        # data_file = os.path.join(data_save_dir, 'train_chas_A') # .format(trans_type, manual_type)
        # data_file = os.path.join(data_save_dir, 'participant_all_ccc_transcript_cut')
        src_data_file = os.path.join(data_save_dir, args.src_data)  # src_data_file will be there w/wo DA
        if  args.src_data_mult is not None:
            # DA condition
            src_data_mult_file = os.path.join(data_save_dir, args.src_data_mult)
        if  args.trg_data is not None:
            # DA condition
            trg_data_file = os.path.join(data_save_dir, args.trg_data)



    else:
        #tran/test

        # data_file  = os.path.join(data_save_dir, 'participant_all_ccc_transcript_cut'.format(mode))
        if 'train' in mode:
            src_data_file = os.path.join(data_save_dir, args.src_data)  # src_data_file will be there w/wo DA during train
            if args.src_data_mult is not None:
                # DA condition
                src_data_mult_file = os.path.join(data_save_dir, args.src_data_mult)
            if  args.trg_data is not None:
            #DA condition
            #if 'train' in mode:
                trg_data_file = os.path.join(data_save_dir, args.trg_data) #tearget train set
        else:
            if args.trg_test_data is not None:
                trg_data_file = os.path.join(data_save_dir, args.trg_test_data) #target test set





    if args.data_not_saved :
        if src_data_file is not None:
            src_data_file += '.csv'
        if src_data_mult_file is not None:
            src_data_mult_file += '.csv'
        # if args.trg_data is not None:
        #     # w DA
        #     trg_data_file += '.csv'

    else:
        if src_data_file is not None:

            src_data_file += '.csv'
        if src_data_mult_file is not None:
            src_data_mult_file += '.csv'
        if trg_data_file is not None :
            # w DA
            trg_data_file += '.csv'
    if src_data_file is not None:

        raw_src_df = pd.read_csv(src_data_file) # transcripts
    if src_data_mult_file is not None:
        raw_src_mult_df = pd.read_csv(src_data_mult_file) # transcripts

    if trg_data_file is not None:
        #DA
        raw_trg_df = pd.read_csv(trg_data_file)  # transcripts


    if "cv" in mode:
        # for DA, only target data will be split and  in test set and src data will be added fully in each fold
        # otherwise, no tgr data, src data will be split and have validation folds

        if validation_dict == None:
            raise ValueError("Cross validation mode requires validation_dict input")
        # if args.data_not_saved:
        #     raise ValueError("Data proprocessing (when data_not_saved == True) is only supported by test mode")
        train_speaker = validation_dict['train_speaker']
        validation_speaker = validation_dict['test_speaker']
        if mode == "train_cv":
            if args.trg_data is  None:
                #wo DA, the cv split will be for src data
                load_src_data_df = raw_src_df[raw_src_df["filename"].apply(lambda x: True if x in train_speaker else False)]
                if args.src_data_mult is not None:
                    load_src_data_mult_df=raw_src_mult_df

            else:
                #DA,  the cv split will be for trg data, trg data will be used full

                load_src_data_df = raw_src_df
                if args.src_data_mult is not None:
                    load_src_data_mult_df=raw_src_mult_df


                load_trg_data_df = raw_trg_df[raw_trg_df["filename"].apply(lambda x: True if x in train_speaker else False)]


        elif mode == "test_cv":
            if args.trg_data is  None:
                # wo DA, src data will have separate test cv set
                load_src_data_df = raw_src_df[raw_src_df["filename"].apply(lambda x: True if x in validation_speaker else False)]


            else:
                #DA cv test with trg data, for DA, only target data will be split and  in test set
                load_trg_data_df = raw_trg_df[raw_trg_df["filename"].apply(lambda x: True if x in validation_speaker else False)]




    else:
        #no cv, only train or test
        if 'train' in mode:
            print(raw_src_mult_df)
            #for train, if DA, both src and trg train set will be available
            load_src_data_df = raw_src_df #full src train set
            if args.src_data_mult is not None:
                load_src_data_mult_df=raw_src_mult_df #multi src
            if raw_trg_df is not None:
                #DA
                load_trg_data_df = raw_trg_df #load full trg data train set
        elif 'test' in mode:
            #for train, if DA, both  trg test set will be available, wo DA, src test set

            if raw_trg_df is not None :
                # DA
                load_trg_data_df = raw_trg_df #load full trg test data

            else:
                load_src_data_df = raw_src_df #src datar test set





    if args.data_not_saved:
        #not implemented for DA
        # cut_data_save_path = os.path.join(data_save_dir, 'participant_all_ccc_transcript_cut.csv'.format(mode))
        # org_data = read_input_no_len_control(load_src_data_df, sample_size=sample_size, max_len=max_seq_l, model=model, save_trans=cut_data_save_path,domain_col=domain_col) #sample_size is actually used for hether to add domain info
        if  args.src_data_mult is not None:
            cut_data_save_path = os.path.join(data_save_dir, args.src_data_mult+'.csv')
            # print(load_src_data_mult_df)
            org_mult_data = read_input_no_len_control(load_src_data_mult_df, sample_size=sample_size, max_len=max_seq_l,
                                                 model=model, save_trans=cut_data_save_path,
                                                 domain_col=domain_col,label_col='ad')  # sam


    else:
        if load_src_data_df is not None:
            org_src_data = read_input_no_len_control(load_src_data_df, sample_size=sample_size, max_len=max_seq_l, model=model, save_trans=None,domain_col=domain_col,label_col='ad')
        if load_src_data_mult_df is not None:
            org_src_mult_data = read_input_no_len_control(load_src_data_mult_df, sample_size=sample_size, max_len=max_seq_l, model=model, save_trans=None,domain_col=domain_col,label_col='ad')


        if load_trg_data_df is not None:
            #DA
            org_trg_data = read_input_no_len_control(load_trg_data_df, sample_size=sample_size, max_len=max_seq_l, model=model, save_trans=None,domain_col=domain_col,label_col=adrc_label_col)


    data_src_list = None
    label_src_list = None
    data_trg_list = None
    label_trg_list = None
    data_src_mult_list = None
    label_src_mult_list=None
    # print(sample_size)
    if org_src_data is not None:
        data_src_list,label_src_list=create_input_example(org_src_data, sample_size,label_col='ad')
    if org_src_mult_data is not None:
        data_src_mult_list,label_src_mult_list=create_input_example(org_src_mult_data, sample_size,label_col='ad')
    if org_trg_data is not None:
        data_trg_list,label_trg_list=create_input_example(org_trg_data, sample_size,label_col=adrc_label_col)
    return data_src_list,label_src_list,data_trg_list,label_trg_list,data_src_mult_list,label_src_mult_list

# set up variables to add to checkpoint and logs filenames
time_now = str(datetime.now().strftime("%d-%m-%Y--%H-%M"))
raw_time_now = time_now.split('--')[0]
print('crossv')
print(args.crossvalidation)
print(args.batch_size)
# torch.cuda.empty_cache()
if adrc_label_col!='ad':
    args.val_file_dir='latest_tmp_dir/'+adrc_label_col+'five_fold.json'
if args.crossvalidation:
    if args.val_file_dir == None:
        raise ValueError("Need to specify val_file_dir")
    assert args.val_file_dir.split(".")[-1] == 'json'
   # run_idx = int(args.val_file_dir.split("fold_")[-1].split('.')[0])
    run_idx=1
    assert run_idx in range(1, 11)
    version = f"version_{args.seed}_val"
else:
    version = f"version_{args.seed}"

model_dict = {'bert-base-uncased': os.path.join(args.off_line_model_dir, 'bert-base-uncased'),
            'roberta-base': os.path.join(args.off_line_model_dir, ''),
             args.model_name: os.path.join(args.off_line_model_dir,''),}

print('model path')
print(model_dict[args.model_name])

plm, tokenizer, model_config, WrapperClass = load_plm(args.model, model_dict[args.model_name])

# For interpretability: allow the backbone PLM to return hidden states and attention weights.

if args.save_hidden_states:
    plm.config.output_hidden_states = True

if args.save_attentions or args.get_attention:
    plm.config.output_attentions = True

if args.save_hidden_states or args.save_attentions or args.get_attention:
    plm.config.return_dict = True

# edit based on whether or not plm was frozen during training
# actually want to save the checkpoints and logs in same place now. Becomes a lot easier to manage later
args.logs_root = args.logs_root.rstrip("/") + "/"
args.project_root = args.project_root.rstrip("/") + "/"
if args.trg_data is not None:
    train_str='joint'
else:
    train_str = 'trg-only'
result_dir=""
if args.tune_plm == True:
    logger.warning("Unfreezing the plm - will be updated during training")
    freeze_plm = False
    # set checkpoint, logs and params save_dirs
    if args.sensitivity:
        logger.warning(f"performing sensitivity analysis experiment!")
        logs_dir = f"{args.logs_root}sensitivity/{args.model_name}_temp{args.template_type}{args.template_id}_verb{args.verbalizer_type}_epoch{args.num_epochs}_optim{args.optimizer}_stk{args.soft_token_num}_lr{args.plm_lr}_{train_str}_tune{args.tune_plm}/{version}"
        ckpt_dir = f"{logs_dir}/checkpoints/"
    elif args.part_tuning:
        logger.warning(f"part parameters tuning run!")
        logs_dir = f"{args.logs_root}parttuning/{args.model_name}_temp{args.template_type}{args.template_id}_verb{args.verbalizer_type}_epoch{args.num_epochs}_optim{args.optimizer}_stk{args.soft_token_num}_lr{args.plm_lr}_layernum{args.tunable_layernum}_{train_str}_tune{args.tune_plm}/{version}"
        ckpt_dir = f"{logs_dir}/checkpoints/"
    # elif args.pause:
    #     logger.warning(f"pause info tuning run!")
    #     if not args.ce_class_weights:
    #         logs_dir = f"{args.logs_root}{args.model_name}_temp{args.template_type}{args.template_id}_verb{args.verbalizer_type}_{args.training_size}_pause{args.pause_threshold}/{version}"
    #     else:
    #         logs_dir = f"{args.logs_root}ce_class_weights/{args.model_name}_temp{args.template_type}{args.template_id}_verb{args.verbalizer_type}_{args.training_size}_pause{args.pause_threshold}/{version}"
    #     ckpt_dir = f"{logs_dir}/checkpoints/"

    else:
        if args.manual_type == "A":
            if adrc_label_col!='ad':
            # ckpt_dir='bert-base-uncased_tempmanual4_verbmanual_epoch10_optimadamw_stk0_100_domain2_bs16_prlr0.5_joint'
                logs_dir = f"{args.logs_root}adrcpitt{adrc_label_col}/{args.model_name}_temp{args.template_type}{args.template_id}_verb{args.verbalizer_type}_epoch{args.num_epochs}_optim{args.optimizer}_stk{args.soft_token_num}_{args.few_shot_n}_domain{args.meta}_bs{args.batch_size}_prlr{args.prompt_lr}_{train_str}/{version}"
            else:
                logs_dir = f"{args.logs_root}{args.model_name}_temp{args.template_type}{args.template_id}_verb{args.verbalizer_type}_epoch{args.num_epochs}_optim{args.optimizer}_stk{args.soft_token_num}_{args.few_shot_n}_domain{args.meta}_bs{args.batch_size}_prlr{args.prompt_lr}_{train_str}_cv{args.crossvalidation}/{version}"

            # logs_dir = f"{args.logs_root}{args.model_name}_temp{args.template_type}{args.template_id}_verb{args.verbalizer_type}_epoch{args.num_epochs}_optim{args.optimizer}_stk{args.soft_token_num}_{args.few_shot_n}_domain{args.meta}_bs{args.batch_size}_prlr{args.prompt_lr}_{train_str}_tune{args.tune_plm}_cv{args.crossvalidation}_m{mult}/{version}"
            ckpt_dir = f"{logs_dir}/checkpoints/"
            if adrc_label_col != 'ad':
                result_dir = f"{args.logs_root}adrcpitt{adrc_label_col}/{args.model_name}_temp{args.template_type}{args.template_id}_verb{args.verbalizer_type}_epoch{args.num_epochs}_optim{args.optimizer}_stk{args.soft_token_num}_{args.few_shot_n}_domain{args.meta}_bs{args.batch_size}_prlr{args.prompt_lr}_{train_str}"
            else:
                result_dir=f"{args.logs_root}{args.model_name}_temp{args.template_type}{args.template_id}_verb{args.verbalizer_type}_epoch{args.num_epochs}_optim{args.optimizer}_stk{args.soft_token_num}_{args.few_shot_n}_domain{args.meta}_bs{args.batch_size}_prlr{args.prompt_lr}_{train_str}_cv{args.crossvalidation}"
        else:
            logs_dir = f"{args.logs_root}{args.model_name}_temp{args.template_type}{args.template_id}_verb{args.verbalizer_type}_epoch{args.num_epochs}_optim{args.optimizer}_stk{args.soft_token_num}_{args.few_shot_n}_{args.manual_type}_domain{args.meta}_bs{args.batch_size}_prlr{args.prompt_lr}_{train_str}_tune{args.tune_plm}_cv{args.crossvalidation}_m{mult}/{version}"
            ckpt_dir = f"{logs_dir}/checkpoints/"
            result_dir=f"{args.logs_root}{args.model_name}_temp{args.template_type}{args.template_id}_verb{args.verbalizer_type}_epoch{args.num_epochs}_optim{args.optimizer}_stk{args.soft_token_num}_{args.few_shot_n}_{args.manual_type}_domain{args.meta}_bs{args.batch_size}_prlr{args.prompt_lr}_{train_str}_tune{args.tune_plm}_cv{args.crossvalidation}_m{mult}"
else:
    logger.warning("Freezing the plm")
    freeze_plm = True
    # we have to account for the slight issue with softverbalizer being wrongly implemented by openprompt
    # here we have an extra agument which will correctly freeze the PLM parts of softverbalizer if set to true
    if args.freeze_verbalizer_plm and args.verbalizer_type == "soft":
        logger.warning("also will be explicitly freezing plm parts of the soft verbalizer")
        if args.sensitivity:
            logger.warning(f"performing sensitivity analysis experiment!")
            logs_dir = f"{args.logs_root}sensitivity/frozen_plm/{args.model_name}_temp{args.template_type}{args.template_id}_frozenverb{args.verbalizer_type}_epoch{args.num_epochs}_optim{args.optimizer}_stk{args.soft_token_num}_{args.few_shot_n}_bs{args.batch_size}_{train_str}/{version}"
            ckpt_dir = f"{logs_dir}/checkpoints/"
        else:
            logs_dir = f"{args.logs_root}frozen_plm/{args.model_name}_temp{args.template_type}{args.template_id}_frozenverb{args.verbalizer_type}_epoch{args.num_epochs}_optim{args.optimizer}_stk{args.soft_token_num}_{args.few_shot_n}_bs{args.batch_size}_{train_str}/{version}"
            ckpt_dir = f"{logs_dir}/checkpoints/" 

    else:# set checkpoint, logs and params save_dirs    
        if args.sensitivity:
            logger.warning(f"performing sensitivity analysis experiment!")
            logs_dir = f"{args.logs_root}sensitivity/frozen_plm/{args.model_name}_temp{args.template_type}{args.template_id}_verb{args.verbalizer_type}_epoch{args.num_epochs}_optim{args.optimizer}_stk{args.soft_token_num}_{args.few_shot_n}_prlr{args.prompt_lr}_{train_str}/{version}"
            ckpt_dir = f"{logs_dir}/checkpoints/"
        else:
            # logs_dir = f"{args.logs_root}frozen_plm/t5-large_tempsoft1_verbsoft_full_100/version_0/" #just to check for test

            logs_dir = f"{args.logs_root}frozen_plm/{args.model_name}_temp{args.template_type}{args.template_id}_verb{args.verbalizer_type}_epoch{args.num_epochs}_optim{args.optimizer}_stk{args.soft_token_num}_{args.few_shot_n}_bs{args.batch_size}_prlr{args.prompt_lr}_{train_str}_tune{args.tune_plm}_cv{args.crossvalidation}_m{mult}_vocab{args.init_from_vocab}/{version}"
            ckpt_dir = f"{logs_dir}/checkpoints/"
            result_dir=f"{args.logs_root}frozen_plm/{args.model_name}_temp{args.template_type}{args.template_id}_verb{args.verbalizer_type}_epoch{args.num_epochs}_optim{args.optimizer}_stk{args.soft_token_num}_{args.few_shot_n}_bs{args.batch_size}_prlr{args.prompt_lr}_{train_str}_tune{args.tune_plm}_cv{args.crossvalidation}_m{mult}_vocab{args.init_from_vocab}"

# check if the checkpoint and params dir exists

if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# set up tensorboard logger
#if not args.no_tensorboard:
#    writer = SummaryWriter(logs_dir)

# initialise empty dataset
DATASET = 'pitt'
scriptsbase = f"{args.project_root}{args.scripts_path}/"
scriptformat = "txt"
src_dataset = {}
trg_dataset = {}
src_dataset_mult={}

# crude setting of sampler to None - changed for mortality with umbalanced dataset

sampler = None
# Below are multiple dataset examples, although right now just mimic ic9-top50.
# decide which template and verbalizer to use
if args.template_type == "manual":
    print(f"manual template selected, with id :{args.template_id}")
    mytemplate = ManualTemplate(tokenizer=tokenizer).from_file(f"{scriptsbase}/manual_template.txt", choice=args.template_id)
elif args.template_type == "prefix":
    print(f"manual template selected, with id :{args.template_id}")
    mytemplate = PrefixTuningTemplate(model=plm,tokenizer=tokenizer,num_token=args.soft_token_num).from_file(f"{scriptsbase}/manual_template.txt", choice=args.template_id)

elif args.template_type == "soft":
    print(f"soft template selected, with id :{args.template_id}")
    mytemplate = SoftTemplate(args.SOURCE_DOMAINS,
                 args.TARGET_DOMAINS,
                 args.N_DMX,
                 args.N_CTX,
                 args.CSC,
                 args.da,
                 class_labels,model=plm, tokenizer=tokenizer, num_tokens=args.soft_token_num, initialize_from_vocab=args.init_from_vocab).from_file(f"{scriptsbase}/soft_manual_template.txt", choice=args.template_id)

elif args.template_type == "ptr":
    print(f"ptr template selected, with id :{args.template_id}")
    mytemplate = ManualTemplate(tokenizer=tokenizer).from_file(f"{scriptsbase}/ptr_template.txt", choice=args.template_id)

elif args.template_type == "mixed":
    print(f"mixed template selected, with id :{args.template_id}")
    mytemplate = MixedTemplate(model=plm, tokenizer=tokenizer).from_file(f"{scriptsbase}/mixed_template.txt", choice=args.template_id)
# now set verbalizer
if args.verbalizer_type == "manual" :
    print(class_labels)
    # myverbalizer = ManualVerbalizer(tokenizer, classes=class_labels).from_file(f"{scriptsbase}/manual_verbalizer.{scriptformat}", choice=args.verbalizer_id)
    myverbalizer = ManualVerbalizer(
        classes = class_labels,
        label_words = {
            
            "dementia": ["dementia","alzheimer's","disfluent","disordered"],
             "healthy": ["healthy"]

        },
        tokenizer = tokenizer,
    )
    print(myverbalizer)

elif args.verbalizer_type == "soft":
    print(f"soft verbalizer selected!")
    # myverbalizer = SoftVerbalizer(tokenizer, plm, num_classes=len(class_labels))
    myverbalizer = SoftVerbalizer(
            classes = class_labels,
            label_words = {
                "dementia": ["dementia"],
                "healthy": ["healthy"],
            },
            tokenizer = tokenizer,
            model = plm,
            num_classes = len(class_labels)
            )

    # we noticed a bug where soft verbalizer was technically not freezing alongside the PLM - meaning it had considerably greater number of trainable parameters
    # so if we want to properly freeze the verbalizer plm components as described here: https://github.com/thunlp/OpenPrompt/blob/4ba7cb380e7b42c19d566e9836dce7efdb2cc235/openprompt/prompts/soft_verbalizer.py#L82
    # we now need to actively set grouped_parameters_1 to requires_grad = False
    if args.freeze_verbalizer_plm and freeze_plm:
        logger.warning(f"We have a soft verbalizer and want to freeze it alongside the PLM!")
        # now set the grouped_parameters_1 require grad to False
        for param in myverbalizer.group_parameters_1:
            param.requires_grad = False
if  "ccc" in DATASET or "adress" in DATASET or 'pitt' in DATASET:
    logger.warning(f"Using the following dataset: {DATASET} ")
    # update data_dir
    data_dir = args.data_dir


    # are we doing any downsampling or balancing etc
    ce_class_weights = args.ce_class_weights
    sampler_weights = args.sampler_weights

    # get different splits
    SAMPLE_SIZE =args.meta # whether to addd meta info, that is domain infor for DA experiment,
    print('cross-valid')
    print(args.crossvalidation)
    print(args.val_file_dir)
    print(args.ce_class_weights)
    if args.crossvalidation:
        with open(args.val_file_dir) as json_read:
            cv_fold_list = ast.literal_eval(json_read.read())
            # print(len(cv_fold_list))
        validation_dict = cv_fold_list[args.val_fold_idx]
        # print(validation_dict)

        src_dataset['train'], src_train_classes_count,trg_dataset['train'], trg_train_classes_count,src_dataset_mult['train'], src_train_mult_classes_count = loading_data_asexample(data_dir, SAMPLE_SIZE, class_labels, args.model_name, mode='train_cv', validation_dict=validation_dict)
        _,_,trg_dataset['validation'],trg_validation_classes_count,_,_ = loading_data_asexample(data_dir, SAMPLE_SIZE, class_labels, args.model_name, mode='test_cv', validation_dict=validation_dict)
        # print(len(src_dataset['train']))
        # print(len(trg_dataset['train']))
        # print(len(trg_dataset['validation']))
        print('train data')
        print(trg_train_classes_count)
        print(trg_validation_classes_count)
        print(len(trg_dataset['validation']))
        print('adrc label_col')
        print(adrc_label_col)
        if adrc_label_col != 'ad':
            print('test dataset size')
            print(len(trg_dataset['train']))

            print(len(trg_dataset['validation']))
        # dataset['test'], _ = loading_data_asexample(data_dir, SAMPLE_SIZE, class_labels, args.model_name, mode='test_cv', validation_dict=validation_dict)
    else:
        #print(args.transcription)
        # if args.transcription == 'chas':
        src_dataset['train'], src_train_classes_count,trg_dataset['train'], trg_train_classes_count,src_dataset_mult['train'], src_train_mult_classes_count = loading_data_asexample(data_dir, SAMPLE_SIZE, class_labels, args.model_name, mode='train')
        # src_dataset['validation'], src_validation_classes_count,trg_dataset['validation'],trg_validation_classes_count = loading_data_asexample(data_dir, SAMPLE_SIZE, class_labels, args.model_name, mode='test')
        _, _,trg_dataset['test'],trg_test_classes_count,_,_ = loading_data_asexample(data_dir, SAMPLE_SIZE, class_labels, args.model_name, mode='test')

        # if args.trg_data is not None:
        #     src_dataset['train']=src_dataset['train']+trg_dataset['train'] # merge srg and target domain data for training

        # elif args.transcription in ['cnntdnn', 'sys14_26.4', 'sys18_25.9']: # for asr trans
        #     dataset['train'], train_classes_count = loading_data_asexample(data_dir, SAMPLE_SIZE, class_labels, args.model_name, mode='train')
        #     dataset['validation'], _ = loading_data_asexample(data_dir, SAMPLE_SIZE, class_labels, args.model_name, mode='test')
        #     dataset['test'], _ = loading_data_asexample(data_dir, SAMPLE_SIZE, class_labels, args.model_name, mode='test') # , data_saved=False
        #
    # the below class labels should align with the label encoder fitted to training data
    if args.ce_class_weights:
        print(args.ce_class_weights)
        # if args.trg_data is not None:
        #     # print(src_train_classes_count)
        #     for key in src_train_classes_count:
        #         src_train_classes_count[key]+=trg_train_classes_count[key]
        #      #adding trg train data
        #     print(src_train_classes_count)

        src_task_class_weights = [src_train_classes_count[0]/src_train_classes_count[i] for i in range(len(class_labels))]

        if args.trg_data is not None:
            trg_task_class_weights=[]
            for i in range(len(class_labels)):
                if trg_train_classes_count[i]==0:
                    trg_task_class_weights.append(trg_train_classes_count[0] / (trg_train_classes_count[i]+1))
                else:
                    trg_task_class_weights.append(trg_train_classes_count[0] / (trg_train_classes_count[i] ))

            # trg_task_class_weights = [trg_train_classes_count[0] / trg_train_classes_count[i] for i in
            #                           range(len(class_labels))]

    # max_seq_l = 512 # this should be specified according to the running GPU's capacity
    if args.tune_plm: # tune the entire plm will use more gpu-memories, thus we should use a smaller batch_size.
        batchsize_t = args.batch_size 
        batchsize_e = args.batch_size
        gradient_accumulation_steps = args.gradient_accum_steps
        model_parallelize = False # if multiple gpus are available, one can use model_parallelize
    else:
        batchsize_t = args.batch_size
        batchsize_e = args.batch_size
        gradient_accumulation_steps = args.gradient_accum_steps
        model_parallelize = False
else:
    #TODO implement los and mimic readmission
    raise NotImplementedError

if args.data_not_saved:
    sys.exit()
assert model_parallelize == False
# if torch.cuda.device_count() > 1:
#     model_parallelize=True

# write hparams to file
# lets write these arguments to file for later loading alongside the trained models
if not os.path.exists(os.path.join(ckpt_dir, 'hparams.txt')):
    with open(os.path.join(ckpt_dir, 'hparams.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
# add the hparams to tensorboard logs
# print(f"hparams dict: {args.__dict__}")
save_metrics = {"random/metric": 0}
if not args.no_tensorboard:
    writer.add_hparams(args.__dict__, save_metrics)

# Now define the template and verbalizer. 
# Note that soft template can be combined with hard template, by loading the hard template from file. 
# For example, the template in soft_template.txt is {}
# The choice_id 1 is the hard template 


# print('wrapped_example')
# wrapped_example = mytemplate.wrap_one_example(dataset['train'][0])
# print(len(wrapped_example))
# for temp in wrapped_example[0]:
#
#     # print(temp)
#     for key in temp:
#         print(key)
#         if 'text' in key:
#             print(temp[key])
#             print(len(temp[key]))

# Select device safely: CUDA if available, otherwise MPS on Apple Silicon, otherwise CPU.
if torch.cuda.is_available():
    use_cuda = True
    cuda_device = torch.device(f"cuda:{args.gpu_num}")
    print(f"Using CUDA device: {cuda_device}")

elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    use_cuda = True
    cuda_device = torch.device("mps")
    print("Using Apple Silicon MPS device")

else:
    use_cuda = False
    cuda_device = torch.device("cpu")
    print("Using CPU")

# now set the default gpu to this one
# torch.cuda.set_device(cuda_device)


print(f"tune_plm value: {args.tune_plm}")
prompt_model = PromptForClassification(plm=plm,template=mytemplate, verbalizer=myverbalizer, freeze_plm=freeze_plm, plm_eval_mode=args.plm_eval_mode)
# print('check 2')
# torch.cuda.mem_get_info()


if use_cuda:
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     prompt_model=torch.nn.DataParallel(prompt_model)
    prompt_model=  prompt_model.to(cuda_device)
    # print('check 3')
    # torch.cuda.mem_get_info()

# if model_parallelize:
#     prompt_model.parallelize()

# if doing few shot learning - produce the datasets here:
if args.training_size == "fewshot":
    print('fewshot %d'%(args.few_shot_n))
    logger.warning(f"Will be performing few shot learning.")
# create the few_shot sampler for when we want to run training and testing with few shot learning
    #not implemented for DA yet
    support_sampler = FewShotSampler(num_examples_per_label = args.few_shot_n, also_sample_dev=False)

    # create a fewshot dataset from training, val and test. Seems to be what several papers do...
    # src_dataset['train'] = support_sampler(src_dataset['train'], seed=args.seed)
    #
    # # can try without also fewshot sampling val and test sets?
    # src_dataset['validation'] = support_sampler(src_dataset['validation'], seed=args.seed)
    # src_dataset['test'] = support_sampler(src_dataset['test'], seed=args.seed)
    trg_dataset['train'] = support_sampler(trg_dataset['train'], seed=args.seed)
    #
    # # can try without also fewshot sampling val and test sets?
    # src_dataset['validation'] = support_sampler(src_dataset['validation'], seed=args.seed)
    # src_dataset['test'] = support_sampler(src_dataset['test'], seed=args.seed)

#validation_data_loader = PromptDataLoader(
#         dataset = src_dataset['train'],
#         tokenizer = tokenizer,
#         template = mytemplate,
#         tokenizer_wrapper_class=WrapperClass,
#         max_seq_length=max_seq_l,
#         decoder_max_length=3,
#         batch_size=args.batch_size,
#     )
#print('validation_data_loader')
#cnt=0
#import random
#temps=random.sample(range(1, len(validation_data_loader)), 3)
#for i,input in enumerate(validation_data_loader):
#     if i!=temps[cnt]:
#      continue
#     print(i)
#     cnt+=1
       
#     temp=tokenizer.decode(input['input_ids'][0]).split()
#     print(temp)
#     idx=temp.index('[MASK].')
#
#     # print(temp)
#     print(len(temp[:idx]))
#     print(input['label'])
#     print(input['attention_mask'])
#     print(input['guid'])
#     cnt+=1
#     if cnt==2:
#      break

#
#         break

# are we doing training?
do_training = (not args.no_training)
if do_training:
    if args.template_type == 'soft' :
        #
        if args.soft_token_num>0 and args.N_CTX==0 and args.N_DMX==0:
            max_seq_l -= args.soft_token_num
        elif args.N_CTX>0 and args.N_DMX>0:
            max_seq_l -= (args.N_CTX+args.N_DMX)

    # if we have a sampler .e.g weightedrandomsampler. Do not shuffle
    if "WeightedRandom" in type(sampler).__name__:
        logger.warning("Sampler is WeightedRandom - will not be shuffling training data!")
        shuffle = False
    else:
        shuffle = True
    logger.warning(f"Do training is True - creating train and validation dataloders!")
    src_train_data_loader = PromptDataLoader(
        dataset = src_dataset['train'],
        tokenizer = tokenizer, 
        template = mytemplate, 
        tokenizer_wrapper_class=WrapperClass,
        max_seq_length=max_seq_l,
        decoder_max_length=3,
        batch_size=args.batch_size,
    )
    if args.src_data_mult is not None:
        src_train_data_loader_mult = PromptDataLoader(
            dataset=src_dataset_mult['train'],
            tokenizer=tokenizer,
            template=mytemplate,
            tokenizer_wrapper_class=WrapperClass,
            max_seq_length=max_seq_l,
            decoder_max_length=3,
            batch_size=args.batch_size,
        )
        # sys.exit()
    if args.trg_data is not None:
        trg_train_data_loader = PromptDataLoader(
            dataset=trg_dataset['train'],
            tokenizer=tokenizer,
            template=mytemplate,
            tokenizer_wrapper_class=WrapperClass,
            max_seq_length=max_seq_l,
            decoder_max_length=3,
            batch_size=args.trg_batch_size,
        )
    # customPromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer, 
    #     tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l, decoder_max_length=3, 
    #     batch_size=batchsize_t,shuffle=shuffle, sampler = sampler, teacher_forcing=False, predict_eos_token=False,
    #     truncate_method="tail")

    if 'validation'  in src_dataset  :
        src_validation_data_loader = PromptDataLoader(
            dataset = src_dataset['validation'],
            tokenizer = tokenizer,
            template = mytemplate,
            tokenizer_wrapper_class=WrapperClass,
            max_seq_length=max_seq_l,
            decoder_max_length=3,
            batch_size=args.batch_size,
        )
    if args.trg_data is not None and 'validation'  in trg_dataset:
        trg_validation_data_loader = PromptDataLoader(
            dataset=trg_dataset['validation'],
            tokenizer=tokenizer,
            template=mytemplate,
            tokenizer_wrapper_class=WrapperClass,
            max_seq_length=max_seq_l,
            decoder_max_length=3,
            batch_size=args.batch_size,
        )



# zero-shot test / eval on test set after training
# print(trg_dataset['test'])
if args.trg_data is not None and 'test' in trg_dataset:
    test_data_loader = PromptDataLoader(
        dataset = trg_dataset['test'],
        tokenizer = tokenizer,
        template = mytemplate,
        tokenizer_wrapper_class=WrapperClass,
        max_seq_length=max_seq_l,
        decoder_max_length=3,
        batch_size=args.batch_size,
    )
    print('test dataset size %d'%(len(test_data_loader)))
AVAI_SCHEDS = ["single_step", "multi_step", "cosine"]


#from transformers import  AdamW, get_linear_schedule_with_warmup,get_constant_schedule_with_warmup  # use AdamW is a standard practice for transformer 
#from transformers.optimization import Adafactor, AdafactorSchedule  # use Adafactor is the default setting for T5

#TODO update this to handle class weights for imabalanced datasets
if ce_class_weights:
    print("we have some task specific class weights - passing to CE loss")
    logger.warning("we have some task specific class weights - passing to CE loss")
    # get from the class_weight function
    src_task_class_weights = torch.tensor(src_task_class_weights, dtype=torch.float)

    if args.trg_data is not None:
        trg_task_class_weights = torch.tensor(trg_task_class_weights, dtype=torch.float)

    if use_cuda:
        src_task_class_weights = src_task_class_weights.to(cuda_device)
        if args.trg_data is not None:
            trg_task_class_weights = trg_task_class_weights.to(cuda_device)

    # set manually cause above didnt work
    # task_class_weights = torch.tensor([1,16.1], dtype=torch.float).to(cuda_device)
    loss_func = torch.nn.CrossEntropyLoss(weight = src_task_class_weights, reduction = 'mean')
else:
    loss_func = torch.nn.CrossEntropyLoss()

# get total steps as a function of the max epochs, batch_size and len of dataloader
tot_step = args.max_steps
print('plm tune')
print(args.tune_plm)
if args.tune_plm:
    logger.warning("We will be tuning the PLM!") # normally we freeze the model when using soft_template. However, we keep the option to tune plm
    no_decay = ['bias', 'LayerNorm.weight'] # it's always good practice to set no decay to biase and LayerNorm parameters
    if not args.part_tuning:
        if isinstance(prompt_model, torch.nn.DataParallel):
            optimizer_grouped_parameters_plm = [
                {'params': [p for n, p in prompt_model.plm.named_parameters() if (not any(nd in n for nd in no_decay))],
                 'weight_decay': 0.01},
                {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0}
            ]

        else:


            optimizer_grouped_parameters_plm = [
                {'params': [p for n, p in prompt_model.plm.named_parameters() if (not any(nd in n for nd in no_decay))], 'weight_decay': 0.01},
                {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
    else:
        tunable_layers = []
        for layer_num in range(12-args.tunable_layernum, 12):
            tunable_layers.append("bert.encoder.layer.{}".format(layer_num))
        tunable_layers.append('cls.predictions')
        if isinstance(prompt_model, torch.nn.DataParallel):

            optimizer_grouped_parameters_plm = [
                {'params': [p for n, p in prompt_model.plm.named_parameters() if (not any(nd in n for nd in no_decay)) and (any(fl in n for fl in tunable_layers))], 'weight_decay': 0.01},
                {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay) and (any(fl in n for fl in tunable_layers))], 'weight_decay': 0.0}
            ]
        else:
            optimizer_grouped_parameters_plm = [
                {'params': [p for n, p in prompt_model.plm.named_parameters() if
                            (not any(nd in n for nd in no_decay)) and (any(fl in n for fl in tunable_layers))],
                 'weight_decay': 0.01},
                {'params': [p for n, p in prompt_model.plm.named_parameters() if
                            any(nd in n for nd in no_decay) and (any(fl in n for fl in tunable_layers))],
                 'weight_decay': 0.0}
            ]
    if args.optimizer.lower() == "adafactor":
        optimizer_plm = Adafactor(optimizer_grouped_parameters_plm,
                                lr=args.plm_lr,relative_step=False,scale_parameter=False, warmup_init=False)
    elif args.optimizer.lower() == "sgd":
        optimizer_plm=torch.optim.SGD(optimizer_grouped_parameters_plm, lr=args.plm_lr)
    else:
        optimizer_plm = torch.optim.AdamW(optimizer_grouped_parameters_plm, lr=args.plm_lr)

    scheduler_plm = get_linear_schedule_with_warmup(
        optimizer_plm, 
        num_warmup_steps=args.plm_warmup_steps, num_training_steps=tot_step)
else:
    logger.warning("We will not be tunning the plm - i.e. the PLM layers are frozen during training")
    optimizer_plm = None
    scheduler_plm = None

# if using soft template
if args.template_type == "soft" or args.template_type == "mixed" or  args.template_type == "prefix":
    logger.warning(f"{args.template_type} template used - will be fine tuning the prompt embeddings!")
    optimizer_grouped_parameters_template = [{'params': [p for name, p in prompt_model.template.named_parameters() if 'raw_embedding' not in name]}] # note that you have to remove the raw_embedding manually from the optimization
    if args.optimizer.lower() == "adafactor":
        optimizer_template = Adafactor(optimizer_grouped_parameters_template,  
                                lr=args.prompt_lr,
                                relative_step=False,
                                scale_parameter=False,
                                warmup_init=False)  # when lr is 0.3, it is the same as the configuration of https://arxiv.org/abs/2104.08691
        scheduler_template = get_constant_schedule_with_warmup(optimizer_template, num_warmup_steps=args.warmup_step_prompt) # when num_warmup_steps is 0, it is the same as the configuration of https://arxiv.org/abs/2104.08691
    elif args.optimizer.lower() == "adamw":
        optimizer_template = torch.optim.AdamW(optimizer_grouped_parameters_template, lr=args.prompt_lr) # usually lr = 0.5
        # optimizer_template=torch.optim.SGD(optimizer_grouped_parameters_plm, lr=args.plm_lr)
        scheduler_template = get_linear_schedule_with_warmup(
                        optimizer_template, 
                        num_warmup_steps=args.warmup_step_prompt, num_training_steps=tot_step) # usually num_warmup_steps is 500
    elif args.optimizer.lower() == "sgd":

        optimizer_template = torch.optim.SGD(
            optimizer_grouped_parameters_template,
            lr=args.prompt_lr
        )
        scheduler_template = get_constant_schedule_with_warmup(optimizer_template, num_warmup_steps=args.warmup_step_prompt) # when num_warmup_steps is 0, it is the same as the configuration of https://arxiv.org/abs/2104.08691

        # scheduler_template=get_cosine_schedule_with_warmup(optimizer_template, num_warmup_steps=args.warmup_step_prompt, num_training_steps=tot_step)
        # scheduler_template = get_linear_schedule_with_warmup(
        #     optimizer_template,
        #     num_warmup_steps=args.warmup_step_prompt, num_training_steps=tot_step)
        # scheduler_template = build_lr_scheduler(optimizer_template)

elif args.template_type == "manual" or args.template_type == "ptr":
    optimizer_template = None
    scheduler_template = None

if args.verbalizer_type == "soft" or args.verbalizer_type == "mixed":
    logger.warning("Soft verbalizer used - will be fine tuning the verbalizer/answer embeddings!")
    optimizer_grouped_parameters_verb = [
    {'params': prompt_model.verbalizer.group_parameters_1, "lr":args.plm_lr},
    {'params': prompt_model.verbalizer.group_parameters_2, "lr":args.plm_lr},
    
    ]

    if args.optimizer.lower() == "sgd":

        optimizer_verb = torch.optim.SGD(
            optimizer_grouped_parameters_template

        )
        scheduler_verb=get_cosine_schedule_with_warmup(optimizer_verb, num_warmup_steps=args.warmup_step_prompt,num_training_steps=tot_step)
    elif args.optimizer.lower() == "adamw":
        optimizer_verb = torch.optim.AdamW(optimizer_grouped_parameters_verb)
        scheduler_verb = get_linear_schedule_with_warmup(
            optimizer_verb,
            num_warmup_steps=args.warmup_step_prompt, num_training_steps=tot_step)  # usually num_warmup_steps is 500


elif args.verbalizer_type == "manual" or args.verbalizer_type == "ptr":
    optimizer_verb = None
    scheduler_verb = None

best_val_epoch=-1
def train(prompt_model, src_train_data_loader,src_train_data_loader_mult=None,trg_train_data_loader=None, mode = "train", ckpt_dir = ckpt_dir):

    # logger.warning(f"cuda current device inside training is: {torch.cuda.current_device()}")
    # set model to train 
    prompt_model.train()

    # set up some counters
    actual_step = 0
    glb_step = 0

    # some validation metrics to monitor
    best_val_acc = 0
    best_val_f1 = 0
    best_val_prec = 0    
    best_val_recall = 0
    #best_val_epoch=-1

    # this will be set to true when max steps are reached
    leave_training = False
    print('num epochs')
    print(args.num_epochs)
    # print('check 4')
    # print(torch.cuda.mem_get_info())
    # trg_train_loader_iter = iter(trg_train_data_loader)
    # if args.trg_data is not None:
    #     if len(src_train_data_loader)>len(trg_train_data_loader):
    #         batch_num=int(math.ceil((len(src_train_data_loader)/args.batch_size)))
    #     else:
    #         batch_num = int(math.ceil((len(trg_train_data_loader) / args.trg_batch_size)))
    # else:
    #     batch_num = int(math.ceil((len(src_train_data_loader) / args.batch_size)))
    batch_num = len(src_train_data_loader)
    # print('template id')
    # print(args.template_id)
    for epoch in tqdm(range(args.num_epochs)):
        print(f"On epoch: {epoch}")
        tot_loss = 0 
        epoch_loss = 0
        # for step, trg_inputs in enumerate(trg_train_data_loader):
        t0=time.time()
        src_train_loader_iter = iter(src_train_data_loader)

        if args.src_data_mult is not None and src_train_data_loader_mult is not None:
            src_train_loader_iter_mult = iter(src_train_data_loader_mult)
        # else:
        #     print('mult none')
        if args.trg_data is not None and trg_train_data_loader is not None:
            trg_train_loader_iter = iter(trg_train_data_loader)
        for step in range(batch_num):

            #print('step')
            #print(step)
            try:
                src_inputs = next(src_train_loader_iter)
                # decoded=tokenizer.decode(src_inputs["input_ids"][0])
                # print(decoded)
                # sys.exit()
                #print(src_inputs)
            except StopIteration:
                print('ekhane src')
                print(step)
                #print(len(src_train_loader_iter))
                src_train_loader_iter = iter(src_train_data_loader)
                src_inputs = next(src_train_loader_iter)
            if use_cuda:
                src_inputs = src_inputs.to(cuda_device)
            # print('check 5')
            # torch.cuda.mem_get_info()
            # torch.cuda.mem_get_info()
            # try:
            if args.src_data_mult is not None and src_train_loader_iter_mult is not None :
                try:
                    src_inputs_mult = next(src_train_loader_iter_mult)
                    print('src mult size')
                    print(len(src_inputs_mult))
                    # print(src_inputs_mult)
                    #
                    # decoded=tokenizer.decode(src_inputs_mult["input_ids"][0])
                    # print(decoded)
                    # sys.exit()

                except StopIteration:
                    print('ekhane src mult')
                    print(step)
                    #print(len(trg_train_loader_iter))
                    src_train_loader_iter_mult = iter(src_train_data_loader_mult)
                    src_inputs_mult = next(src_train_loader_iter_mult)
                if use_cuda:
                    src_inputs_mult = src_inputs_mult.to(cuda_device)

            if args.trg_data is not None:
                try:
                    trg_inputs = next(trg_train_loader_iter)
                    # print('trg_inputs')


                except StopIteration:
                    print('ekhane trg')
                    print(step)
                    #print(len(trg_train_loader_iter))
                    trg_train_loader_iter = iter(trg_train_data_loader)
                    trg_inputs = next(trg_train_loader_iter)
                if use_cuda:
                    trg_inputs = trg_inputs.to(cuda_device)


            src_logits = prompt_model(src_inputs)
            if args.src_data_mult is not None and src_inputs_mult is not None and len(src_inputs_mult)>0:
                src_logits_mult = prompt_model(src_inputs_mult)
            if args.trg_data is not None and trg_inputs is not None and len(trg_inputs)>0:

                trg_logits = prompt_model(trg_inputs)

            # print('check 6')
            # torch.cuda.mem_get_info()
            src_labels = src_inputs['label']
            if args.src_data_mult is not None and src_inputs_mult is not None and len(src_inputs_mult)>0:
                src_labels_mult = src_inputs_mult['label']
            if args.trg_data is not None and trg_inputs is not None  and len(trg_inputs)>0:

                trg_labels = trg_inputs['label']
            # print('src tensor type')
            # print(type(src_logits))
            # print(type(src_labels))
            src_loss = loss_func(src_logits, src_labels)
            src_loss_mult=0
            if args.src_data_mult is not None and src_labels_mult is not None  and len(src_labels_mult)>0:
                src_loss_mult = loss_func(src_logits_mult, src_labels_mult)
            #print(src_loss)
            trg_loss=0
            if args.trg_data is not None and trg_labels is not None and len(trg_labels)>0:
                # print('trg tensor type')
                # print(type(trg_logits))
                # print(type(trg_labels))
                # trg_labels = torch.as_tensor(trg_labels, dtype=torch.long)
                # trg_logits = torch.as_tensor(trg_logits, dtype=torch.long)
                # print(trg_logits)
                # print(trg_labels)
                trg_labels = trg_labels.to(torch.int64)
                trg_loss = loss_func(trg_logits, trg_labels)
                #print(trg_loss)

            # loss_x = F.cross_entropy(output_x[:, :self.n_cls], label)
            # loss_u = (F.cross_entropy(
            #     output_u[:, self.n_cls:2 * self.n_cls],
            #     label_p,
            #     reduction="none") * mask).sum() / mask.sum()
            loss = src_loss +src_loss_mult+ args.U * trg_loss

            # normalize loss to account for gradient accumulation
            loss = loss / gradient_accumulation_steps
            # print('check 7')
            # torch.cuda.mem_get_info()
            # propogate backward to calculate gradients
            loss.backward()
            tot_loss += loss.item()

            actual_step+=1
            # log loss to tensorboard  every 50 steps    

            # clip gradients based on gradient accumulation steps
            if actual_step % gradient_accumulation_steps == 0:
                # log loss
                aveloss = tot_loss/(step+1)
                if not args.no_tensorboard:
                    # write to tensorboard
                    writer.add_scalar("train/batch_loss", aveloss, glb_step)        

                # clip grads            
                torch.nn.utils.clip_grad_norm_(prompt_model.parameters(), 1.0)
                glb_step += 1
                # print('check 8')
                # torch.cuda.mem_get_info()
                # backprop the loss and update optimizers and then schedulers too
                # plm
                if optimizer_plm is not None:
                    optimizer_plm.step()
                    optimizer_plm.zero_grad()
                # print('check 9')
                # torch.cuda.mem_get_info()
                if scheduler_plm is not None:
                    scheduler_plm.step()
                # template
                if optimizer_template is not None:
                    optimizer_template.step()
                    optimizer_template.zero_grad()
                if scheduler_template is not None:
                    scheduler_template.step()
                # verbalizer
                if optimizer_verb is not None:
                    optimizer_verb.step()
                    optimizer_verb.zero_grad()
                if scheduler_verb is not None:
                    scheduler_verb.step()

                # check if we are over max steps
                if glb_step > args.max_steps:
                    logger.warning("max steps reached - stopping training!")
                    leave_training = True
                    break
                #print('check 10')
                #torch.cuda.mem_get_info()
        # get epoch loss and write to tensorboard
        if args.trg_data is not None and args.src_data_mult is not None:
            epoch_loss = tot_loss/(len(src_train_data_loader)+len(trg_train_data_loader)+len(src_train_data_loader_mult))
        else:
            epoch_loss = tot_loss / (len(src_train_data_loader) )
        print("{} sec. for each epoch".format(time.time() - t0))

        if not args.no_tensorboard:
            writer.add_scalar("train/epoch_loss", epoch_loss, epoch)

        
        # run a run through validation set to get some metrics
        if args.crossvalidation:
            val_loss, val_acc, val_prec_weighted, val_prec_macro, val_recall_weighted,val_recall_macro, val_f1_weighted,val_f1_macro, val_auc_weighted,val_auc_macro = evaluate(prompt_model, trg_validation_data_loader, epoch=epoch,mode='validation')
        else:
            val_loss, val_acc, val_prec_weighted, val_prec_macro, val_recall_weighted,val_recall_macro, val_f1_weighted,val_f1_macro, val_auc_weighted,val_auc_macro = evaluate(prompt_model, test_data_loader, epoch=epoch,mode='test')

        if not args.no_tensorboard:
            writer.add_scalar("valid/loss", val_loss, epoch)
            writer.add_scalar("valid/balanced_accuracy", val_acc, epoch)
            writer.add_scalar("valid/precision_weighted", val_prec_weighted, epoch)
            writer.add_scalar("valid/precision_macro", val_prec_macro, epoch)
            writer.add_scalar("valid/recall_weighted", val_recall_weighted, epoch)
            writer.add_scalar("valid/recall_macro", val_recall_macro, epoch)
            writer.add_scalar("valid/f1_weighted", val_f1_weighted, epoch)
            writer.add_scalar("valid/f1_macro", val_f1_macro, epoch)
            
            #TODO add binary classification metrics e.g. roc/auc
            writer.add_scalar("valid/auc_weighted", val_auc_weighted, epoch)
            writer.add_scalar("valid/auc_macro", val_auc_macro, epoch)        

            # # add cm to tensorboard
            # writer.add_figure("valid/Confusion_Matrix", cm_figure, epoch)
        print("Epoch {}, train loss: {}, valid loss:{}, valid acc:{} ".format(epoch, epoch_loss,val_loss,val_acc), flush=True)
        # save checkpoint if validation accuracy improved
        if val_acc > best_val_acc:
            # only save ckpts if no_ckpt is False - we do not always want to save - especially when developing code
            #print('Accuracy improved! Saving checkpoint')
            if not args.no_ckpt:
                print('acc improved')
                logger.warning(f"Accuracy improved! Saving checkpoint at :{ckpt_dir}!")
                if not args.crossvalidation:
                    torch.save(prompt_model.state_dict(),os.path.join(ckpt_dir, "best-checkpoint.ckpt"))
                else:
                    torch.save(prompt_model.state_dict(),os.path.join(ckpt_dir, "best-checkpoint_cv{}_fold{}.ckpt".format(run_idx, args.val_fold_idx)))
            best_val_acc = val_acc
            best_val_epoch=epoch

        if glb_step > args.max_steps:
            leave_training = True
            break
    
        if leave_training:
            logger.warning("Leaving training as max steps have been met!")
            break 
    if args.crossvalidation:
        acc,prec,recall,f1,auc=test_evaluation(prompt_model,ckpt_dir,trg_validation_data_loader,epoch_num=best_val_epoch)
        result=str(args.val_fold_idx)+" "+str(args.seed)+" "+str(acc)+" "+str(prec)+" "+str(recall)+" "+str(f1)+" "+str(auc)+"\n"
    else:
        acc, prec, recall, f1, auc = test_evaluation(prompt_model, ckpt_dir, test_data_loader, epoch_num=best_val_epoch)
        result = str(args.val_fold_idx) + " " + str(args.seed) + " " + str(acc) + " " + str(prec) + " " + str(
            recall) + " " + str(f1) + " " + str(auc) + "\n"
    # if len(result)>10:
    #     with open(result_dir + "/" + 'result.txt', 'a+') as run_write:
    #         run_write.write(result)
# ## evaluate

# %%
# def get_attention_weight(attention_scores,text_mask):
#     # Extract attention scores with shape (batch_size, num_heads, sequence_length, sequence_length)
#     # attention_scores = outputs['attentions']

#     # grab attention from final layer of the model
#     attention = attention_scores[-1]

#     # this subsets the attention matrix to size `(12, n_query_tokens, n_response_tokens)`
#     # this has to be a for loop because the values of `n_query_tokens` and `n_response_tokens` will be different for different batch items
#     bs = attention.shape[0]
#     extracted_attentions = []
#     for i in range(bs):
#         # q_mask = query_mask[i]
#         t_mask = text_mask[i]

#         masked_attention = attention[i, :, t_mask, :][:, :, t_mask]

#         # extracted_attentions.append(masked_attention)
#         # average over number of heads and query tokens
#         extracted_attentions.append(torch.mean(masked_attention, (0,1)).detach().cpu().numpy().flatten())


#     # grab attention submatrix for first batch item, as an example
#     # extracted_attention = extracted_attentions[0]

#     # average over number of heads and query tokens
#     # mean_attentions = torch.mean(extracted_attention, (0, 1)).detach().numpy().flatten()
#     return extracted_attentions


def get_output_field(outputs, field_name):
    """
    Safely extract hidden_states or attentions from HuggingFace ModelOutput.
    Works for dict-like and attribute-like outputs.
    """
    if outputs is None:
        return None

    if hasattr(outputs, field_name):
        return getattr(outputs, field_name)

    if isinstance(outputs, dict):
        return outputs.get(field_name, None)

    return None


def get_mask_positions(inputs, tokenizer):
    """
    Find the [MASK] position for each input in the batch.

    Prefer OpenPrompt's loss_ids if available, because that marks the prediction token.
    Otherwise fall back to tokenizer.mask_token_id.
    """
    input_ids = inputs["input_ids"]

    if "loss_ids" in inputs:
        loss_ids = inputs["loss_ids"]
        mask_positions = []

        for i in range(loss_ids.shape[0]):
            positions = torch.nonzero(loss_ids[i] == 1, as_tuple=False).view(-1)
            if len(positions) > 0:
                mask_positions.append(int(positions[0].item()))
            else:
                mask_positions.append(0)

        return mask_positions

    mask_token_id = tokenizer.mask_token_id
    mask_positions = []

    for i in range(input_ids.shape[0]):
        positions = torch.nonzero(input_ids[i] == mask_token_id, as_tuple=False).view(-1)
        if len(positions) > 0:
            mask_positions.append(int(positions[0].item()))
        else:
            mask_positions.append(0)

    return mask_positions


def extract_hidden_state_summaries(plm_outputs, inputs, tokenizer):
    """
    Extract compact hidden-state representations for probing.

    We save:
    1. final-layer [CLS] vector
    2. final-layer [MASK] vector

    This is much smaller than saving all layers for all tokens.
    """
    hidden_states = get_output_field(plm_outputs, "hidden_states")

    if hidden_states is None:
        return None, None

    # Final layer hidden states: [batch_size, seq_len, hidden_dim]
    final_hidden = hidden_states[-1]

    cls_vectors = final_hidden[:, 0, :].detach().cpu()

    mask_positions = get_mask_positions(inputs, tokenizer)
    mask_vectors = []

    for i, pos in enumerate(mask_positions):
        mask_vectors.append(final_hidden[i, pos, :].detach().cpu())

    mask_vectors = torch.stack(mask_vectors, dim=0)

    return cls_vectors, mask_vectors


def extract_mask_attention(plm_outputs, inputs, tokenizer):
    """
    Extract final-layer attention from the [MASK] token to all input tokens.

    Output per sample:
    vector of length seq_len, averaged across attention heads.
    """
    attentions = get_output_field(plm_outputs, "attentions")

    if attentions is None:
        return None

    # Final layer attention: [batch_size, num_heads, seq_len, seq_len]
    final_attention = attentions[-1]

    mask_positions = get_mask_positions(inputs, tokenizer)
    attention_mask = inputs["attention_mask"]

    batch_attention_vectors = []

    for i, pos in enumerate(mask_positions):
        # Attention from [MASK] query token to all key tokens, averaged over heads
        attn_vec = final_attention[i, :, pos, :].mean(dim=0)

        # Keep only real tokens, not padding
        valid_len = int(attention_mask[i].sum().item())
        attn_vec = attn_vec[:valid_len].detach().cpu()

        batch_attention_vectors.append(attn_vec)

    return batch_attention_vectors

def save_interpretability_tensors(
    save_dir,
    file_prefix,
    allids,
    all_cls_hidden_states,
    all_mask_hidden_states,
    all_mask_attentions
):
    """
    Save hidden states and attention summaries separately from CSV metrics.
    """
    save_obj = {
        "ids": allids,
    }

    if len(all_cls_hidden_states) > 0:
        save_obj["cls_hidden_states"] = torch.cat(all_cls_hidden_states, dim=0)

    if len(all_mask_hidden_states) > 0:
        save_obj["mask_hidden_states"] = torch.cat(all_mask_hidden_states, dim=0)

    if len(all_mask_attentions) > 0:
        # Variable-length attention vectors because sequences may have different valid lengths.
        # Keep them as a list.
        save_obj["mask_attentions"] = all_mask_attentions

    if len(save_obj.keys()) > 1:
        torch.save(save_obj, os.path.join(save_dir, f"{file_prefix}_interpretability.pt"))


def evaluate(prompt_model, dataloader, mode = "validation", class_labels = class_labels, epoch=None):
    print('evaluating in %s mode'%(mode))
    prompt_model.eval()

    tot_loss = 0
    allpreds = []
    alllabels = []
    alllogits = []     #record logits from the the model
    allscores = []    # store probabilties i.e. softmax applied to logits
    allids = []

    all_input_ids = []
    all_attention_masks = []

    all_cls_hidden_states = []
    all_mask_hidden_states = []
    all_mask_attentions = []

    captured_plm_outputs = {}

    hook_handle = None

    if args.save_hidden_states or args.save_attentions or args.get_attention:
        def capture_plm_outputs(module, module_inputs, module_outputs):
            captured_plm_outputs["outputs"] = module_outputs

        hook_handle = prompt_model.plm.register_forward_hook(capture_plm_outputs)

    with torch.no_grad():
        for step, inputs in enumerate(dataloader):

            if use_cuda:
                inputs = inputs.to(cuda_device)

            if args.save_wrapped_inputs:
                all_input_ids.extend(inputs["input_ids"].detach().cpu().tolist())
                all_attention_masks.extend(inputs["attention_mask"].detach().cpu().tolist())

            logits = prompt_model(inputs)

            plm_outputs = captured_plm_outputs.get("outputs", None)

            if args.save_hidden_states:
                cls_vecs, mask_vecs = extract_hidden_state_summaries(
                    plm_outputs=plm_outputs,
                    inputs=inputs,
                    tokenizer=tokenizer
                )

                if cls_vecs is not None:
                    all_cls_hidden_states.append(cls_vecs)

                if mask_vecs is not None:
                    all_mask_hidden_states.append(mask_vecs)

            if args.save_attentions or args.get_attention:
                mask_attention = extract_mask_attention(
                    plm_outputs=plm_outputs,
                    inputs=inputs,
                    tokenizer=tokenizer
                )

                if mask_attention is not None:
                    all_mask_attentions.extend(mask_attention)


            labels = inputs['label']
            labels = labels.to(torch.int64)
            loss = loss_func(logits, labels)
            tot_loss += loss.item()

            # add labels to list
            alllabels.extend(labels.cpu().tolist())

            # add ids to list - they are already a list so no need to send to cpu
            allids.extend(inputs['guid'])

            # add logits to list
            alllogits.extend(logits.cpu().tolist())
            #use softmax to normalize, as the sum of probs should be 1
            # if binary classification we just want the positive class probabilities
            if len(class_labels) > 2:  
                allscores.extend(torch.nn.functional.softmax(logits, dim = -1).cpu().tolist())
            else:
                allscores.extend(torch.nn.functional.softmax(logits, dim = -1)[:,1].cpu().tolist())

            # add predicted labels    
            allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())

    if hook_handle is not None:
        hook_handle.remove()
    
    val_loss = tot_loss/len(dataloader)    
    # get sklearn based metrics
    acc = balanced_accuracy_score(alllabels, allpreds)
    f1_weighted = f1_score(alllabels, allpreds, average = 'weighted')
    f1_macro = f1_score(alllabels, allpreds, average = 'macro')
    prec_weighted = precision_score(alllabels, allpreds, average = 'weighted')
    prec_macro = precision_score(alllabels, allpreds, average = 'macro')
    recall_weighted = recall_score(alllabels, allpreds, average = 'weighted')
    recall_macro = recall_score(alllabels, allpreds, average = 'macro')


    # roc_auc  - only really good for binary classification but can try for multiclass too
    # use scores instead of predicted labels to give probs
    
    if len(class_labels) > 2:   
        roc_auc_weighted = roc_auc_score(alllabels, allscores, average = "weighted", multi_class = "ovr")
        roc_auc_macro = roc_auc_score(alllabels, allscores, average = "macro", multi_class = "ovr")
                  
    else:
        roc_auc_weighted = roc_auc_score(alllabels, allscores, average = "weighted")
        roc_auc_macro = roc_auc_score(alllabels, allscores, average = "macro")         

    # get confusion matrix
    cm = confusion_matrix(alllabels, allpreds)

    # plot using custom function defined below
    # cm_figure = plotConfusionMatrix(cm, class_labels)
    # below makes a slightly nicer plot 
    # if not args.crossvalidation:
    #     cm_figure = plot_confusion_matrix(cm, class_labels)
    if epoch is not None:
        epoch_dir = os.path.join(ckpt_dir, "epoch{}".format(epoch))
        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir)
    else:
        epoch_dir = ckpt_dir
    if not os.path.exists(epoch_dir):
        os.makedirs(epoch_dir)
    # if we are doing final evaluation on test data - save labels, pred_labels, logits and some plots
    if mode == 'test':
        assert epoch != None

        # create empty dict to store labels, pred_labels, logits
        results_dict = {}
        logger.warning(f"mode was: {mode} so will be saving evaluation results to file as well as tensorboard!")
        # classification report
        print(classification_report(alllabels, allpreds, target_names=class_labels))
        
        # save to dict
        test_report = classification_report(alllabels, allpreds, target_names=class_labels, output_dict=True)
        # now to file
        test_report_df = pd.DataFrame(test_report).transpose()
        if args.crossvalidation:
            test_report_name = "test_class_report_cv{}_fold{}.csv".format(run_idx, args.val_fold_idx)
            test_results_name = "test_results_cv{}_fold{}.csv".format(run_idx, args.val_fold_idx)
            figure_name = "test_cm_cv{}_fold{}.png".format(run_idx, args.val_fold_idx)
        else:
            if args.transcription in ['cnntdnn', 'fusionshujie', "sys14_26.4", "sys18_25.9"]: # "cnntdnn":
                test_report_name = "test_{}_{}_class_report.csv".format(args.transcription, args.asr_format)
                test_results_name = "test_{}_{}_results.csv".format(args.transcription, args.asr_format)
                figure_name = "test_{}_{}_cm.png".format(args.transcription, args.asr_format)
            # elif args.transcription == "chas":
            else:
                test_report_name = "test_class_report.csv"
                test_results_name = "test_results.csv"
                figure_name = "test_cm.png"
        test_report_df.to_csv(os.path.join(epoch_dir, test_report_name), index = False)
        print(test_report_df)
        # save logits etc
        
        results_dict = {}
        results_dict['id'] = allids
        results_dict['labels'] = alllabels
        results_dict['pred_labels'] = allpreds
        results_dict['logits'] = alllogits
        results_dict['probas'] = allscores
        if args.save_wrapped_inputs:
            results_dict["input_ids"] = all_input_ids
            results_dict["attention_mask"] = all_attention_masks
        # save dataframe and to csv
        pd.DataFrame(results_dict).to_csv(os.path.join(epoch_dir, test_results_name), index =False)
        save_interpretability_tensors(
            save_dir=epoch_dir,
            file_prefix=test_results_name.replace(".csv", ""),
            allids=allids,
            all_cls_hidden_states=all_cls_hidden_states,
            all_mask_hidden_states=all_mask_hidden_states,
            all_mask_attentions=all_mask_attentions
        )

    if mode == 'last':
    
        # create empty dict to store labels, pred_labels, logits
        results_dict = {}
        logger.warning(f"mode was: {mode} so will be saving evaluation results to file as well as tensorboard!")

        # save to dict
        test_report = classification_report(alllabels, allpreds, target_names=class_labels, output_dict=True)
        # now to file
        test_report_df = pd.DataFrame(test_report).transpose()
        if args.crossvalidation:
            test_report_name = "test_class_report_last_cv{}_fold{}.csv".format(run_idx, args.val_fold_idx)
            test_results_name = "test_results_last_cv{}_fold{}.csv".format(run_idx, args.val_fold_idx)
            figure_name = "test_cm_last_cv{}_fold{}.png".format(run_idx, args.val_fold_idx)
        else:
            test_report_name = "test_class_report_last.csv"
            test_results_name = "test_results_last.csv"
            figure_name = "test_cm_last.png"
        test_report_df.to_csv(os.path.join(ckpt_dir, test_report_name), index=False)               
        
        results_dict = {}
        results_dict['id'] = allids
        results_dict['labels'] = alllabels
        results_dict['pred_labels'] = allpreds
        results_dict['logits'] = alllogits
        results_dict['probas'] = allscores
        if args.save_wrapped_inputs:
            results_dict["input_ids"] = all_input_ids
            results_dict["attention_mask"] = all_attention_masks
        # save dataframe and to csv
        pd.DataFrame(results_dict).to_csv(os.path.join(ckpt_dir, test_results_name), index =False)
        save_interpretability_tensors(
            save_dir=ckpt_dir,
            file_prefix=test_results_name.replace(".csv", ""),
            allids=allids,
            all_cls_hidden_states=all_cls_hidden_states,
            all_mask_hidden_states=all_mask_hidden_states,
            all_mask_attentions=all_mask_attentions
        )     
        # # save confusion matrix
        # if not args.crossvalidation:
        #     cm_figure.savefig(os.path.join(ckpt_dir, figure_name))
    assert args.part_tuning == False
    if (mode == 'validation') and (epoch >= 0) and (epoch < 10):
        print('calculating validation result')   
        assert epoch != None
        # epoch_dir = os.path.join(ckpt_dir, "epoch{}".format(epoch))
        # cv_dir_name = "./output/cv_{}_temp_{}/".format(args.model_name,args.template_id)




        
        # create empty dict to store labels, pred_labels, logits
        results_dict = {}
        logger.warning(f"mode was: {mode} so will be saving evaluation results to file!")

        # save to dict
        test_report = classification_report(alllabels, allpreds, target_names=class_labels, output_dict=True)
        # now to file
        test_report_df = pd.DataFrame(test_report).transpose()
        if args.crossvalidation:
            test_report_name = "test_class_report_cv{}_fold{}.csv".format(run_idx, args.val_fold_idx)
            test_results_name = "test_results_cv{}_fold{}.csv".format(run_idx, args.val_fold_idx)
            figure_name = "test_cm_cv{}_fold{}.png".format(run_idx, args.val_fold_idx)
        else:
            if args.transcription in ['cnntdnn', 'fusionshujie', "sys14_26.4", "sys18_25.9"]: #"cnntdnn":
                test_report_name = "test_{}_{}_class_report.csv".format(args.transcription, args.asr_format)
                test_results_name = "test_{}_{}_results.csv".format(args.transcription, args.asr_format)
                figure_name = "test_{}_{}_cm.png".format(args.transcription, args.asr_format)
            elif args.transcription == "chas":
                test_report_name = "test_class_report.csv"
                test_results_name = "test_results.csv"
                figure_name = "test_cm.png"
            else:
                NotImplemented
        if '/' in args.model_name:
            modelname=args.model_name.split('/')[1]
        else:
            modelname=args.model_name

        #logger.warning('save to {}'.format(os.path.join(cv_dir_name_name, test_report_name)))
        test_report_df.to_csv(os.path.join(epoch_dir, test_report_name), index = False)
        print('resulst saved in %s'%(epoch_dir))
        print(test_report_df)

        # save logits etc
        
        results_dict = {}
        results_dict['id'] = allids
        results_dict['labels'] = alllabels
        results_dict['pred_labels'] = allpreds
        results_dict['logits'] = alllogits
        results_dict['probas'] = allscores
        if args.save_wrapped_inputs:
            results_dict["input_ids"] = all_input_ids
            results_dict["attention_mask"] = all_attention_masks
        pd.DataFrame(results_dict).to_csv(os.path.join(epoch_dir, test_results_name), index=False)
        # save tensors
        save_interpretability_tensors(
            save_dir=epoch_dir,
            file_prefix=test_results_name.replace(".csv", ""),
            allids=allids,
            all_cls_hidden_states=all_cls_hidden_states,
            all_mask_hidden_states=all_mask_hidden_states,
            all_mask_attentions=all_mask_attentions
        )
        if args.last_ckpt:
            if not args.crossvalidation:
                torch.save(prompt_model.state_dict(),os.path.join(epoch_dir, "checkpoint.ckpt"))
            else:
                torch.save(prompt_model.state_dict(),os.path.join(epoch_dir, "checkpoint_cv{}_fold{}.ckpt".format(run_idx, args.val_fold_idx)))

    return val_loss, acc, prec_weighted, prec_macro, recall_weighted, recall_macro, f1_weighted, f1_macro, roc_auc_weighted, roc_auc_macro


# TODO - add a test function to load the best checkpoint and obtain metrics on all test data. Can do this post training but may be nicer to do after training to avoid having to repeat.

def test_evaluation(prompt_model, ckpt_dir, dataloader, epoch_num=None):
    # once model is trained we want to load best checkpoint back in and evaluate on test set - then log to tensorboard and save logits and few other things to file?

    # first load the state_dict using the ckpt_dir of the best model i.e. should be best-checkpoint.ckpt found in the ckpt_dir
    if args.last_ckpt:
        assert epoch_num != None
        # ckpt_dir='bert-base-uncased_tempmanual4_verbmanual_epoch10_optimadamw_stk0_100_domain2_bs16_prlr0.5_joint'
        epoch_dir = os.path.join(ckpt_dir, "epoch{}".format(epoch_num))
        if not args.crossvalidation:
            loaded_model = torch.load(os.path.join(epoch_dir, "checkpoint.ckpt"))
        else:
            loaded_model = torch.load(os.path.join(epoch_dir, "checkpoint_cv{}_fold{}.ckpt".format(run_idx, args.val_fold_idx)))
    elif not args.no_ckpt:
        if not args.crossvalidation:
            loaded_model = torch.load(os.path.join(ckpt_dir, "best-checkpoint.ckpt"))
        else:
            loaded_model = torch.load(os.path.join(ckpt_dir, "best-checkpoint_cv{}_fold{}.ckpt".format(run_idx, args.val_fold_idx)))
    else:
        NotImplemented
    # now load this into the already create PromptForClassification object i.e. the supplied prompt_model
    prompt_model.load_state_dict(state_dict = loaded_model)
    # print("cuda_device", cuda_device)
    if use_cuda:
        prompt_model.to(cuda_device)

    # then run evaluation on test_dataloader

    if args.last_ckpt:
        test_loss, test_acc, test_prec_weighted, test_prec_macro, test_recall_weighted,test_recall_macro, test_f1_weighted,test_f1_macro, test_auc_weighted,test_auc_macro = evaluate(prompt_model,
                                                                                                                    mode = 'validation', dataloader = dataloader, epoch=epoch_num)
    else:
        test_loss, test_acc, test_prec_weighted, test_prec_macro, test_recall_weighted,test_recall_macro, test_f1_weighted,test_f1_macro, test_auc_weighted,test_auc_macro = evaluate(prompt_model,
                                                                                                                    mode = 'test', dataloader = dataloader,epoch=epoch_num)
    if not args.no_tensorboard:
        # write to tensorboard

        writer.add_scalar("test/loss", test_loss, 0)
        writer.add_scalar("test/balanced_accuracy", test_acc, 0)
        writer.add_scalar("test/precision_weighted", test_prec_weighted, 0)
        writer.add_scalar("test/precision_macro", test_prec_macro, 0)
        writer.add_scalar("test/recall_weighted", test_recall_weighted, 0)
        writer.add_scalar("test/recall_macro", test_recall_macro, 0)
        writer.add_scalar("test/f1_weighted", test_f1_weighted, 0)
        writer.add_scalar("test/f1_macro", test_f1_macro, 0)

        #TODO add binary classification metrics e.g. roc/auc
        writer.add_scalar("test/auc_weighted", test_auc_weighted, 0)
        writer.add_scalar("test/auc_macro", test_auc_macro, 0)        

        # # add cm to tensorboard
        # writer.add_figure("test/Confusion_Matrix", cm_figure, 0)
    return test_acc,test_prec_macro, test_recall_macro,  test_f1_macro,  test_auc_macro

def last_epoch_evaluation(prompt_model, ckpt_dir, test_data_loader):

    # then run evaluation on test_dataloader

    test_loss, test_acc, test_prec_weighted, test_prec_macro, test_recall_weighted,test_recall_macro, test_f1_weighted,test_f1_macro, test_auc_weighted,test_auc_macro = evaluate(prompt_model,
                                                                                                                    mode = 'last', dataloader = test_data_loader)


# nicer plot
def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    
    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    credit: https://towardsdatascience.com/exploring-confusion-matrix-evolution-on-tensorboard-e66b39f4ac12
    """

   
    font = FontProperties()
    font.set_family('serif')
    font.set_name('Times New Roman')
    font.set_style('normal')

    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Confusion matrix: ADReSS")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    
    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    
    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() * 0.90
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # figure.savefig(f'experiments/{model}/test_mtx.png')

    return figure


# if refactor this has to be run before any training has occured
if args.zero_shot:
    logger.info("Obtaining zero shot performance on test set!")
    
    zero_loss, zero_acc, zero_prec_weighted, zero_prec_macro, zero_recall_weighted,zero_recall_macro, zero_f1_weighted,zero_f1_macro, zero_auc_weighted,zero_auc_macro = evaluate(prompt_model, test_data_loader, mode='last')

    if not args.no_tensorboard:
        writer.add_scalar("zero_shot/loss", zero_loss, 0)
        writer.add_scalar("zero_shot/balanced_accuracy", zero_acc, 0)
        writer.add_scalar("zero_shot/precision_weighted", zero_prec_weighted, 0)
        writer.add_scalar("zero_shot/precision_macro", zero_prec_macro, 0)
        writer.add_scalar("zero_shot/recall_weighted", zero_recall_weighted, 0)
        writer.add_scalar("zero_shot/recall_macro", zero_recall_macro, 0)
        writer.add_scalar("zero_shot/f1_weighted", zero_f1_weighted, 0)
        writer.add_scalar("zero_shot/f1_macro", zero_f1_macro, 0)

        #TODO add binary classification metrics e.g. roc/auc
        writer.add_scalar("zero_shot/auc_weighted", zero_auc_weighted, 0)
        writer.add_scalar("zero_shot/auc_macro", zero_auc_macro, 0)
        

        # # add cm to tensorboard
        # writer.add_figure("zero_shot/Confusion_Matrix", zero_cm_figure, 0)

# run training

logger.warning(f"do training : {do_training}")
print('start')
if do_training:
    logger.warning("Beginning full training!")
    if args.trg_data is not None and args.src_data_mult is not None:
        train(prompt_model, src_train_data_loader=src_train_data_loader,src_train_data_loader_mult=src_train_data_loader_mult,trg_train_data_loader=trg_train_data_loader, mode="train", ckpt_dir=ckpt_dir)
    if args.trg_data is not None:
        train(prompt_model, src_train_data_loader=src_train_data_loader,trg_train_data_loader=trg_train_data_loader, mode="train", ckpt_dir=ckpt_dir)
    else:
        train(prompt_model, src_train_data_loader=src_train_data_loader, mode="train", ckpt_dir=ckpt_dir)

    torch.cuda.empty_cache()

elif not do_training:
    logger.warning("No training will be performed!")

# run on test set if desired

if args.run_evaluation:
    logger.warning("Running evaluation on test set using best checkpoint!")
    print('seed', args.seed)
    for epoch_num in [best_val_epoch]:
        if args.crossvalidation:
            acc, prec, recall, f1, auc = test_evaluation(prompt_model, ckpt_dir, trg_validation_data_loader, epoch_num)
        else:
            acc,prec,recall,f1,auc=test_evaluation(prompt_model, ckpt_dir, test_data_loader, epoch_num)
        result=str(args.val_fold_idx)+" "+str(args.seed)+" "+str(acc)+" "+str(prec)+" "+str(recall)+" "+str(f1)+" "+str(auc)+"\n"
        with open(result_dir+"/"+'result_'+adrc_label_col+'.txt', 'a+') as run_write:
            run_write.write(result)
    # last_epoch_evaluation(prompt_model, ckpt_dir, test_data_loader)

# write the contents to file
if not args.no_tensorboard:
    writer.flush()
