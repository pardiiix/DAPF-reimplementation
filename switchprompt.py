# copied and modified from  NtaylorOX/Public_Clinical_Prompt
from argparse import Namespace
from typing import Dict
from torch.utils.data import DataLoader

from tqdm import tqdm
from openprompt.data_utils import PROCESSORS
import torch
from openprompt.data_utils.utils import InputExample
import argparse
import numpy as np
import pandas as pd
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from openprompt.prompts import ManualVerbalizer, ManualTemplate, SoftVerbalizer, PTRTemplate, PTRVerbalizer

from openprompt.prompts import SoftTemplate, MixedTemplate
from openprompt import PromptForClassification
# from openprompt.utils.logging import logger
from loguru import logger

import time
import os
from datetime import datetime
# from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score, classification_report, \
    confusion_matrix, roc_auc_score

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
parser.add_argument("--num_dynamic_keyword",type=int, default=0)
parser.add_argument("--num_static_keyword",type=int, default=10)
parser.add_argument("--src_data", default="participant_all_ccc_transcript_cut")
parser.add_argument("--trg_data", default="adress-train_all")
parser.add_argument("--trg_test_data", default=None)
parser.add_argument("--get_attention", default=False)
parser.add_argument("--U", type=float, default=1.05)
parser.add_argument("--SOURCE_DOMAINS", default=["health and wellbeing"])
parser.add_argument("--TARGET_DOMAINS", default=["picture description"])
parser.add_argument("--LR_SCHEDULER", default="cosine")
parser.add_argument("--MAX_EPOCH", type=int, default=25)
parser.add_argument("--WARMUP_TYPE", default="linear")
parser.add_argument("--seed", type=int, default=11)
parser.add_argument("--plm_eval_mode", action="store_true",
                    help="whether to turn off the dropout in the freezed model. Set to true to turn off.")
parser.add_argument("--tune_plm", default=False, action="store_true")
parser.add_argument("--freeze_verbalizer_plm", default=False, action="store_true")
parser.add_argument("--zero_shot", action="store_true")
parser.add_argument("--few_shot_n", type=int, default=100)
parser.add_argument("--no_training", default=False, action="store_true")
parser.add_argument("--run_evaluation", default=True, action="store_true")
parser.add_argument("--model", type=str, default='',
                    help="The plm to use e.g. t5-base, roberta-large, bert-base, emilyalsentzer/Bio_ClinicalBERT")
parser.add_argument("--model_name", type=str, default='roberta-base')
parser.add_argument("--project_root", type=str,default='./',
                    help="The project root in the file system, i.e. the absolute path of OpenPrompt")
parser.add_argument("--logs_root", type=str,default='./output/',
                    help="The dir in which project results are stored in, i.e. the absolute path of OpenPrompt")
parser.add_argument("--off_line_model_dir", type=str, default="",help="The dir in which pre-trained model are stored in")
parser.add_argument("--manual_type", type=str, default="A")
parser.add_argument("--data_dir", type=str,default = "./data/")
parser.add_argument("--data_not_saved", default=False, action="store_true")
parser.add_argument("--scripts_path", type=str, default="template/")
parser.add_argument("--max_steps", default=50000, type=int)
parser.add_argument("--plm_lr", type=float, default=1e-05)
parser.add_argument("--plm_warmup_steps", type=float, default=5)
parser.add_argument("--prompt_lr", type=float, default=0.5)
parser.add_argument("--warmup_step_prompt", type=int, default=5)
parser.add_argument("--num_epochs", default=10, type=int)
parser.add_argument("--batch_size", default=8, type=int)
parser.add_argument("--trg_batch_size", default=16, type=int)
parser.add_argument("--init_from_vocab", default=False, action="store_true")
# parser.add_argument("--eval_every_steps", type=int, default=100)
parser.add_argument("--soft_token_num", type=int, default=32)
# parser.add_argument("--optimizer", type=str, default="adamw")
parser.add_argument("--optimizer", type=int, default=0)
parser.add_argument("--gradient_accum_steps", type=int, default=1)
# parser.add_argument("--dev_run",action="store_true")
parser.add_argument("--gpu_num", type=int, default=0)
# parser.add_argument("--balance_data", action="store_true") # whether to downsample data to majority class
parser.add_argument("--ce_class_weights", default="False",
                    action="store_true")  # whether to apply class weights to cross entropy loss fn
parser.add_argument("--sampler_weights", action="store_true")  # apply weights to weighted data sampler
parser.add_argument("--training_size", type=str, default="full")  # or fewshot or zero
parser.add_argument("--no_ckpt", default=False, type=bool)
parser.add_argument("--last_ckpt", action="store_true", default=False)
parser.add_argument("--crossvalidation", default=True, type=bool)
parser.add_argument("--val_file_dir", default='latest_tmp_dir/five_fold.json', type=str)
parser.add_argument("--val_fold_idx", type=int, default=0)
parser.add_argument("--no_tensorboard", default=True, action="store_true")
parser.add_argument("--part_tuning", action="store_true")
parser.add_argument("--tunable_layernum", type=int, default=0)
parser.add_argument("--transcription", type=str, default='chas')
parser.add_argument("--prompt", type=bool, default=False)
parser.add_argument("--prefix", type=bool, default=True)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--hidden_dropout_prob", type=float, default=0.1)
parser.add_argument("--prefix_projection", type=bool, default=False)
parser.add_argument("--prefix_hidden_size", type=int, default=512)
parser.add_argument("--baseline", type=bool, default=False)
parser.add_argument("--data_parallel", type=bool, default=True)


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
args.few_shot_n=args.num_static_keyword
print(' fewshot')
print(args.few_shot_n)
from openprompt.plms.seq2seq import T5TokenizerWrapper, T5LMTokenizerWrapper
from transformers import T5Config, T5Tokenizer, T5ForConditionalGeneration
from openprompt.data_utils.data_sampler import FewShotSampler
from openprompt.plms import get_model,get_baseline_model
from prompt_ad_utils import read_input_text_len_control, read_input_no_len_control
import ast
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup, \
    get_constant_schedule_with_warmup  # use AdamW is a standard practice for transformer
from transformers.optimization import Adafactor, AdafactorSchedule  # use Adafact
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import ExponentialLR
max_seq_l=512
if args.prompt:
    max_seq_l = max_seq_l - (args.soft_token_num)# 1024  for t5-basecauses oom error
elif args.prefix:
    # max_seq_l = max_seq_l - (args.soft_token_num)-(args.num_static_keyword)-19# 1024  for t5-basecauses oom error, switchprompt
    max_seq_l = max_seq_l - (args.soft_token_num) #, for ptuningv2
torch.cuda.empty_cache()
class_labels = [
    "healthy",
    "dementia"
]
optimizer = ['adamw', 'adafactor', 'sgd']
args.optimizer = optimizer[args.optimizer]
if args.baseline and not (args.prompt or args.prefix):
    learning_rate = args.plm_lr
else:
    learning_rate = args.prompt_lr

# baseline_plm=f"{args.logs_root}mlm/{args.model_name}-last3rd_finetuned/"
# baseline_plm=f"{args.logs_root}mlm/{args.model_name}-last2_finetuned/"
baseline_plm=f"{args.logs_root}mlm/{args.model_name}-last_finetuned/"
# ckpt_name="last3rd"
# ckpt_name="last2"
ckpt_name="last"


print('baseline plm')
print(baseline_plm)
def create_input_example(org_data, include_domain):
    # include_domain: whether to include domain information for. if >-1, means DA experiemnt,
    # otherwise joint training, for manual template, add manual domain prompt
    data_list = []
    label_list = []
    for index, data in org_data.iterrows():
        # print(data['joined_all_par_trans'].split())
        # print(len(data['joined_all_par_trans'].split()))
        # if type(data['filename'])==str:
        #     input_example = InputExample(text_a=data['text'], label=data['ad'], guid=data["id"])
        # else:
        input_example = InputExample(text_a=data['text'], label=data['ad'], guid=data["filename"])##ccc
        # input_example = InputExample(text_a=data['text'], label=data['ad'], guid=data["id"])#adress
        data_list.append(input_example)
        label_list.append(data['ad'])
    return data_list, Counter(label_list)


def loading_data_asexample(data_save_dir, sample_size, classes, model, mode='train', validation_dict=None):
    print(mode)
    src_data_file = None
    trg_data_file = None
    raw_src_df = None
    raw_trg_df = None
    load_src_data_df = None
    load_trg_data_df = None
    org_src_data = None
    org_trg_data = None

    if 'cv' in mode:
        # data_file = os.path.join(data_save_dir, 'train_chas_A') # .format(trans_type, manual_type)
        # data_file = os.path.join(data_save_dir, 'participant_all_ccc_transcript_cut')
        src_data_file = os.path.join(data_save_dir, args.src_data)  # src_data_file will be there w/wo DA

        if args.trg_data is not None:
            # DA condition
            trg_data_file = os.path.join(data_save_dir, args.trg_data)



    else:
        # tran/test

        # data_file  = os.path.join(data_save_dir, 'participant_all_ccc_transcript_cut'.format(mode))
        if 'train' in mode:
            src_data_file = os.path.join(data_save_dir,
                                         args.src_data)  # src_data_file will be there w/wo DA during train

            if args.trg_data is not None:
                # DA condition
                # if 'train' in mode:
                trg_data_file = os.path.join(data_save_dir, args.trg_data)  # tearget train set
        else:
            if args.trg_test_data is not None:
                trg_data_file = os.path.join(data_save_dir, args.trg_test_data)  # target test set

    if args.data_not_saved:
        if src_data_file is not None:
            src_data_file += '.csv'

        if args.trg_data is not None:
            # w DA
            trg_data_file += '.csv'

    else:
        if src_data_file is not None:
            src_data_file += '.csv'
        if trg_data_file is not None:
            # w DA
            trg_data_file += '.csv'
    if src_data_file is not None:
        raw_src_df = pd.read_csv(src_data_file)  # transcripts
    if trg_data_file is not None:
        # DA
        raw_trg_df = pd.read_csv(trg_data_file)  # transcripts

    if "cv" in mode:
        # for DA, only target data will be split and  in test set and src data will be added fully in each fold
        # otherwise, no tgr data, src data will be split and have validation folds

        if validation_dict == None:
            raise ValueError("Cross validation mode requires validation_dict input")
        if args.data_not_saved:
            raise ValueError("Data proprocessing (when data_not_saved == True) is only supported by test mode")
        train_speaker = validation_dict['train_speaker']
        validation_speaker = validation_dict['test_speaker']
        if mode == "train_cv":
            if args.trg_data is None:
                # wo DA, the cv split will be for src data
                load_src_data_df = raw_src_df[
                    raw_src_df["filename"].apply(lambda x: True if x in train_speaker else False)]
            else:
                # DA,  the cv split will be for trg data, trg data will be used full

                load_src_data_df = raw_src_df

                load_trg_data_df = raw_trg_df[
                    raw_trg_df["filename"].apply(lambda x: True if x in train_speaker else False)]


        elif mode == "test_cv":
            if args.trg_data is None:
                # wo DA, src data will have separate test cv set
                load_src_data_df = raw_src_df[
                    raw_src_df["filename"].apply(lambda x: True if x in validation_speaker else False)]


            else:
                # DA cv test with trg data, for DA, only target data will be split and  in test set
                load_trg_data_df = raw_trg_df[
                    raw_trg_df["filename"].apply(lambda x: True if x in validation_speaker else False)]




    else:
        # no cv, only train or test
        if 'train' in mode:
            # for train, if DA, both src and trg train set will be available
            load_src_data_df = raw_src_df  # full src train set
            if raw_trg_df is not None:
                # DA
                load_trg_data_df = raw_trg_df  # load full trg data train set
        elif 'test' in mode:
            # for train, if DA, both  trg test set will be available, wo DA, src test set

            if raw_trg_df is not None:
                # DA
                load_trg_data_df = raw_trg_df  # load full trg test data

            else:
                load_src_data_df = raw_src_df  # src datar test set


    if load_src_data_df is not None:
        org_src_data = read_input_no_len_control(load_src_data_df, sample_size=sample_size, max_len=max_seq_l,
                                                 model=model, )
    if load_trg_data_df is not None:
        # DA
        org_trg_data = read_input_no_len_control(load_trg_data_df, sample_size=sample_size, max_len=max_seq_l,
                                                 model=model)

    data_src_list = None
    label_src_list = None
    data_trg_list = None
    label_trg_list = None
    print(sample_size)
    if org_src_data is not None:
        data_src_list, label_src_list = create_input_example(org_src_data, sample_size)
    if org_trg_data is not None:
        data_trg_list, label_trg_list = create_input_example(org_trg_data, sample_size)
    return data_src_list, label_src_list, data_trg_list, label_trg_list


# set up variables to add to checkpoint and logs filenames
time_now = str(datetime.now().strftime("%d-%m-%Y--%H-%M"))
raw_time_now = time_now.split('--')[0]
print('crossv')
print(args.crossvalidation)
print(args.batch_size)
# torch.cuda.empty_cache()
if args.crossvalidation:
    if args.val_file_dir == None:
        raise ValueError("Need to specify val_file_dir")
    assert args.val_file_dir.split(".")[-1] == 'json'
    # run_idx = int(args.val_file_dir.split("fold_")[-1].split('.')[0])
    run_idx = 1
    assert run_idx in range(1, 11)
    version = f"version_{args.seed}_val"
else:
    version = f"version_{args.seed}"

model_dict = {'bert-base-uncased': os.path.join(args.off_line_model_dir, 'bert-base-uncased'),
              'roberta-base': os.path.join(args.off_line_model_dir, ''),
              args.model_name: os.path.join(args.off_line_model_dir, args.model_name), }

args.logs_root = args.logs_root.rstrip("/") + "/"
args.project_root = args.project_root.rstrip("/") + "/"
if args.trg_data is not None:
    train_str = 'joint'
else:
    train_str = 'trg-only'
result_dir = ""
if args.baseline:
    logs_dir = f"{args.logs_root}mlm/{args.model_name}_epoch{args.num_epochs}_optim{args.optimizer}_pre{args.soft_token_num}_bs{args.batch_size}_{train_str}_lr{learning_rate}_cv{args.crossvalidation}_{ckpt_name}/{version}"
    ckpt_dir = f"{logs_dir}/checkpoints/"
    result_dir = f"{args.logs_root}mlm/{args.model_name}_epoch{args.num_epochs}_optim{args.optimizer}_pre{args.soft_token_num}_bs{args.batch_size}_{train_str}_lr{learning_rate}_cv{args.crossvalidation}_{ckpt_name}"
elif args.prefix and args.model=='bertprefixv2' :
    logs_dir = f"{args.logs_root}ptuningv2/{args.model_name}_epoch{args.num_epochs}_optim{args.optimizer}_pre{args.soft_token_num}_bs{args.batch_size}_{train_str}_lr{learning_rate}_cv{args.crossvalidation}/{version}"
    ckpt_dir = f"{logs_dir}/checkpoints/"
    result_dir = f"{args.logs_root}ptuningv2/{args.model_name}_epoch{args.num_epochs}_optim{args.optimizer}_pre{args.soft_token_num}_bs{args.batch_size}_{train_str}_lr{learning_rate}_cv{args.crossvalidation}"

elif args.prefix and  args.model=='bertprefix':
    logs_dir = f"{args.logs_root}switchprompt/{args.model_name}_epoch{args.num_epochs}_optim{args.optimizer}_pre{args.soft_token_num}_bs{args.batch_size}_{train_str}_lr{learning_rate}_cv{args.crossvalidation}/{version}"
    ckpt_dir = f"{logs_dir}/checkpoints/"
    result_dir = f"{args.logs_root}switchprompt/{args.model_name}_epoch{args.num_epochs}_optim{args.optimizer}_pre{args.soft_token_num}_bs{args.batch_size}_{train_str}_lr{learning_rate}_cv{args.crossvalidation}"

# check if the checkpoint and params dir exists

if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)


# initialise empty dataset
DATASET = 'ccc'
scriptsbase = f"{args.project_root}{args.scripts_path}/"
scriptformat = "txt"
src_dataset = {}
trg_dataset = {}
# crude setting of sampler to None - changed for mortality with umbalanced dataset

sampler = None
#
if "ccc" in DATASET or "adress" in DATASET or 'pitt' in DATASET:
    logger.warning(f"Using the following dataset: {DATASET} ")
    # update data_dir
    data_dir = args.data_dir

    # are we doing any downsampling or balancing etc
    ce_class_weights = args.ce_class_weights
    sampler_weights = args.sampler_weights

    # get different splits
    SAMPLE_SIZE = 2  # whether to addd meta info, that is domain infor for DA experiment,
    print('cross-valid')
    print(args.crossvalidation)
    print(args.ce_class_weights)
    if args.crossvalidation:
        with open(args.val_file_dir) as json_read:
            cv_fold_list = ast.literal_eval(json_read.read())
            # print(len(cv_fold_list))
        validation_dict = cv_fold_list[args.val_fold_idx]
        # print(validation_dict)
        src_dataset['train'], src_train_classes_count, trg_dataset[
            'train'], trg_train_classes_count = loading_data_asexample(data_dir, SAMPLE_SIZE, class_labels,
                                                                       args.model_name, mode='train_cv',
                                                                       validation_dict=validation_dict)
        _, _, trg_dataset['validation'], trg_validation_classes_count = loading_data_asexample(data_dir, SAMPLE_SIZE,
                                                                                               class_labels,
                                                                                               args.model_name,
                                                                                               mode='test_cv',
                                                                                               validation_dict=validation_dict)

        # dataset['test'], _ = loading_data_asexample(data_dir, SAMPLE_SIZE, class_labels, args.model_name, mode='test_cv', validation_dict=validation_dict)
    else:
        # print(args.transcription)
        # if args.transcription == 'chas':
        src_dataset['train'], src_train_classes_count, trg_dataset[
            'train'], trg_train_classes_count = loading_data_asexample(data_dir, SAMPLE_SIZE, class_labels,
                                                                       args.model_name, mode='train')
        # src_dataset['validation'], src_validation_classes_count,trg_dataset['validation'],trg_validation_classes_count = loading_data_asexample(data_dir, SAMPLE_SIZE, class_labels, args.model_name, mode='test')
        _, _, trg_dataset['test'], trg_test_classes_count = loading_data_asexample(data_dir, SAMPLE_SIZE, class_labels,
                                                                                   args.model_name, mode='test')
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

        src_task_class_weights = [src_train_classes_count[0] / src_train_classes_count[i] for i in
                                  range(len(class_labels))]
        if args.trg_data is not None:
            trg_task_class_weights = [trg_train_classes_count[0] / trg_train_classes_count[i] for i in
                                      range(len(class_labels))]

else:
    # TODO implement los and mimic readmission
    raise NotImplementedError

if args.data_not_saved:
    sys.exit()
if args.baseline:
    prompt_model=get_baseline_model(args.model,model_dict[args.model_name],load_model_path=baseline_plm)
else:
    prompt_model=get_model(args,args.model,model_dict[args.model_name],src_dataset['train'],trg_dataset['train'])
# sys.exit()
 # assert model_parallelize == False
# if torch.cuda.device_count() > 1:
#     model_parallelize=True

# write hparams to file
# lets write these arguments to file for later loading alongside the trained models
if not os.path.exists(os.path.join(ckpt_dir, 'hparams.txt')):
    with open(os.path.join(ckpt_dir, 'hparams.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
# add the hparams to tensorboard logs
# print(f"hparams dict: {args.__dict__}")


# are we using cuda and if so which number of device
use_cuda = True
if use_cuda:
    cuda_device = torch.device(f'cuda:{args.gpu_num}')
    print(cuda_device)
    print('check 1')
    # print(torch.cuda.mem_get_info())

else:
    cuda_device = torch.device('cpu')
    print('cpu')

# now set the default gpu to this one
# torch.cuda.set_device(cuda_device)




if use_cuda:
    if torch.cuda.device_count() > 1 and args.data_parallel:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        prompt_model=torch.nn.DataParallel(prompt_model)
    prompt_model = prompt_model.to(cuda_device)
    # print('check 3')
    # torch.cuda.mem_get_info()



# if doing few shot learning - produce the datasets here:
if args.training_size == "fewshot":
    logger.warning(f"Will be performing few shot learning.")
    # create the few_shot sampler for when we want to run training and testing with few shot learning
    # not implemented for DA yet
    support_sampler = FewShotSampler(num_examples_per_label=args.few_shot_n, also_sample_dev=False)

    # create a fewshot dataset from training, val and test. Seems to be what several papers do...
    # src_dataset['train'] = support_sampler(src_dataset['train'], seed=args.seed)
    trg_dataset['train'] = support_sampler(trg_dataset['train'], seed=args.seed)#for DA only change the


    # can try without also fewshot sampling val and test sets?
    # src_dataset['validation'] = support_sampler(src_dataset['validation'], seed=args.seed)
    # src_dataset['test'] = support_sampler(src_dataset['test'], seed=args.seed)

do_training = (not args.no_training)
def encode(docs):
    '''
    This function takes list of texts and returns input_ids and attention_mask of texts
    '''
    if isinstance(prompt_model, torch.nn.DataParallel):
        encoded_dict = prompt_model.module.tokenizer.batch_encode_plus(docs, add_special_tokens=True, max_length=max_seq_l, padding='max_length',
                                return_attention_mask=True, truncation=True, return_tensors='pt')
    else:
        encoded_dict = prompt_model.tokenizer.batch_encode_plus(docs, add_special_tokens=True,
                                                                       max_length=max_seq_l, padding='max_length',
                                                                       return_attention_mask=True, truncation=True,
                                                                       return_tensors='pt')
    input_ids = encoded_dict['input_ids']
    attention_masks = encoded_dict['attention_mask']
    return input_ids, attention_masks
def get_dataloader(text,labels,bs,guid):
    train_input_ids, train_att_masks = encode(text)
    train_y = torch.LongTensor(labels)
    guid = torch.LongTensor(guid)
    # print(train_input_ids.size())
    # print(train_att_masks.size())
    # print(train_y.size())
    # print(guid.size())
    train_dataset = TensorDataset(train_input_ids, train_att_masks, train_y,guid)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=bs)
    return train_dataloader
if do_training:

    logger.warning(f"Do training is True - creating train and validation dataloders!")
    src_train_data_loader=get_dataloader([elem.text_a for elem in src_dataset['train'] ],[elem.label for elem in src_dataset['train'] ],args.batch_size,[int(elem.guid) for elem in src_dataset['train'] ])
    print('trg train size')
    print(len(trg_dataset['train']))
    if args.trg_data is not None:
        trg_train_data_loader = get_dataloader(
            [elem.text_a for elem in trg_dataset['train']],
                                               [elem.label for elem in trg_dataset['train']],
            args.trg_batch_size,
            [int(elem.guid) for elem in trg_dataset['train']]
        )


    # customPromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer,
    #     tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l, decoder_max_length=3,
    #     batch_size=batchsize_t,shuffle=shuffle, sampler = sampler, teacher_forcing=False, predict_eos_token=False,
    #     truncate_method="tail")
    if 'validation' in src_dataset:
        src_validation_data_loader = get_dataloader([elem.text_A for elem in src_dataset['validation']],
                                               [elem.label for elem in src_dataset['validation']], args.batch_size,[int(elem.guid) for elem in src_dataset['validation'] ])

    if args.trg_data is not None and 'validation' in trg_dataset:
        trg_validation_data_loader = get_dataloader([elem.text_a for elem in trg_dataset['validation']],
                                               [elem.label for elem in trg_dataset['validation']], args.batch_size,[int(elem.guid) for elem in trg_dataset['validation'] ])



# zero-shot test / eval on test set after training
# print(trg_dataset['test'])
if args.trg_data is not None and 'test' in trg_dataset:
    test_data_loader = get_dataloader([elem.text_a for elem in trg_dataset['test']],
                                                   [elem.label for elem in trg_dataset['test']], args.trg_batch_size,[int(elem.guid) for elem in trg_dataset['test']])
elif args.trg_data is not None and 'validation' in trg_dataset and args.crossvalidation:
    test_data_loader = get_dataloader([elem.text_a for elem in trg_dataset['validation']],
                                      [elem.label for elem in trg_dataset['validation']], args.trg_batch_size,[int(elem.guid) for elem in trg_dataset['validation']])


AVAI_SCHEDS = ["single_step", "multi_step", "cosine"]

# from transformers import  AdamW, get_linear_schedule_with_warmup,get_constant_schedule_with_warmup  # use AdamW is a standard practice for transformer
# from transformers.optimization import Adafactor, AdafactorSchedule  # use Adafactor is the default setting for T5

# TODO update this to handle class weights for imabalanced datasets
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
    loss_func = torch.nn.CrossEntropyLoss(weight=src_task_class_weights, reduction='mean')
else:
    loss_func = torch.nn.CrossEntropyLoss()

# get total steps as a function of the max epochs, batch_size and len of dataloader
tot_step = args.max_steps
print('plm tune')
print(args.tune_plm)

ALL_LAYERNORM_LAYERS = [torch.nn.LayerNorm]

def get_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result


def get_decay_parameter_names( model) :
    """
    Get all parameter names that weight decay will be applied to

    Note that some models implement their own layernorm instead of calling nn.LayerNorm, weight decay could still
    apply to those modules since this function only filter out instance of nn.LayerNorm
    """
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    return decay_parameters
if args.prompt or args.prefix or args.baseline:
    decay_parameters = get_decay_parameter_names(prompt_model)
    if not isinstance(prompt_model, torch.nn.DataParallel):

        optimizer_grouped_parameters_prompt = [
            {
                "params": [
                    p for n, p in prompt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                ],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [
                    p for n, p in prompt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                ],
                "weight_decay": 0.0,
            },
        ]
    else:
        optimizer_grouped_parameters_prompt = [
            {
                "params": [
                    p for n, p in prompt_model.module.named_parameters() if (n in decay_parameters and p.requires_grad)
                ],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [
                    p for n, p in prompt_model.module.named_parameters() if (n not in decay_parameters and p.requires_grad)
                ],
                "weight_decay": 0.0,
            },
        ]


    um_training_steps = tot_step
    if args.optimizer.lower() == "adafactor":
        optimizer_prompt = Adafactor(optimizer_grouped_parameters_prompt,
                                lr=learning_rate,
                                relative_step=False,
                                scale_parameter=False,
                                warmup_init=False)  # when lr is 0.3, it is the same as the configuration of https://arxiv.org/abs/2104.08691
        # ExponentialLR
        if args.baseline and not (args.prompt or args.prefix):
            scheduler_prompt = get_linear_schedule_with_warmup(
                optimizer_prompt,
                num_warmup_steps=args.plm_warmup_steps, num_training_steps=tot_step)
        else:
            scheduler_prompt = get_constant_schedule_with_warmup(optimizer_prompt, num_warmup_steps=args.warmup_step_prompt) # when num_warmup_steps is 0, it is the same as the configuration of https://arxiv.org/abs/2104.08691
    elif args.optimizer.lower() == "adamw":
        optimizer_prompt = torch.optim.AdamW(optimizer_grouped_parameters_prompt, lr=learning_rate) # usually lr = 0.5
        # optimizer_template=torch.optim.SGD(optimizer_grouped_parameters_plm, lr=args.plm_lr)
        if args.baseline and not (args.prompt or args.prefix):
            scheduler_prompt = get_linear_schedule_with_warmup(
                optimizer_prompt,
                num_warmup_steps=args.plm_warmup_steps, num_training_steps=tot_step)
        else:
            scheduler_prompt = get_constant_schedule_with_warmup(optimizer_prompt,
                                                                 num_warmup_steps=args.warmup_step_prompt)

    elif args.optimizer.lower() == "sgd":

        optimizer_prompt = torch.optim.SGD(
            optimizer_grouped_parameters_prompt,
            lr=learning_rate
        )
        if args.baseline and not (args.prompt or args.prefix):
            
            
            
             scheduler_prompt= get_linear_schedule_with_warmup(
                optimizer_prompt,
                num_warmup_steps=args.plm_warmup_steps, num_training_steps=tot_step)
        else:
            scheduler_prompt = get_constant_schedule_with_warmup(optimizer_prompt,
                                                                 num_warmup_steps=args.warmup_step_prompt)
logger.warning("We will not be tunning the plm - i.e. the PLM layers are frozen during training")

optimizer_plm = None
scheduler_plm = None

# if using soft template


best_val_epoch = -1


def train(prompt_model, src_train_data_loader, trg_train_data_loader=None, mode="train", ckpt_dir=ckpt_dir):
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
    # best_val_epoch=-1

    # this will be set to true when max steps are reached
    leave_training = False
    print('num epochs')
    print(args.num_epochs)
    # print('check 4')
    # print(torch.cuda.mem_get_info())
    # trg_train_loader_iter = iter(trg_train_data_loader)

    batch_num = len(src_train_data_loader)

    for epoch in tqdm(range(args.num_epochs)):
        print(f"On epoch: {epoch}")
        tot_loss = 0
        epoch_loss = 0
        # for step, trg_inputs in enumerate(trg_train_data_loader):
        t0=time.time()
        src_train_loader_iter = iter(src_train_data_loader)
        if args.trg_data is not None:
            trg_train_loader_iter = iter(trg_train_data_loader)
        for step in range(batch_num):

            # print('step')
            # print(step)
            try:
                src_inputs = next(src_train_loader_iter)
                # print(src_inputs)
            except StopIteration:
                print('ekhane src')
                print(step)
                # print(len(src_train_loader_iter))
                src_train_loader_iter = iter(src_train_data_loader)
                src_inputs = next(src_train_loader_iter)
            if use_cuda:
                # src_inputs = src_inputs.to(cuda_device)
                src_input_ids, src_att_mask, src_labels,_ = [data.to(cuda_device) for data in src_inputs]
            else:
                src_input_ids, src_att_mask, src_labels,_ = [data.to(cuda_device) for data in src_inputs]


            # print('check 5')
            # torch.cuda.mem_get_info()
            # torch.cuda.mem_get_info()
            # try:
            if args.trg_data is not None:
                try:
                    trg_inputs = next(trg_train_loader_iter)
                    # print(trg_inputs)


                except StopIteration:
                    print('ekhane trg')
                    print(step)
                    # print(len(trg_train_loader_iter))
                    trg_train_loader_iter = iter(trg_train_data_loader)
                    trg_inputs = next(trg_train_loader_iter)
                if use_cuda:
                    # trg_inputs = trg_inputs.to(cuda_device)
                    trg_input_ids, trg_att_mask, trg_labels,_ = [data.to(cuda_device) for data in trg_inputs]
                else:
                    trg_input_ids, trg_att_mask, trg_labels,_ = [data.to(cuda_device) for data in trg_inputs]


            src_output = prompt_model(input_ids=src_input_ids,attention_mask=src_att_mask,labels=src_labels,return_dict=True)
            # print(src_output)
            if args.trg_data is not None and len(trg_inputs) > 0:
                trg_output = prompt_model(input_ids=trg_input_ids,attention_mask=trg_att_mask,labels=trg_labels,return_dict=True)
            # print(trg_output)
            # print('check 6')
            # torch.cuda.mem_get_info()
            # src_labels = src_inputs['label']
            # if args.trg_data is not None and len(trg_inputs) > 0:
            #     trg_labels = trg_inputs['label']
            #
            # src_loss = loss_func(src_logits, src_labels)
            # # print(src_loss)
            # trg_loss = 0
            # if args.trg_data is not None and len(trg_labels) > 0:
            #     trg_loss = loss_func(trg_logits, trg_labels)
            #     # print(trg_loss)

            # loss_x = F.cross_entropy(output_x[:, :self.n_cls], label)
            # loss_u = (F.cross_entropy(
            #     output_u[:, self.n_cls:2 * self.n_cls],
            #     label_p,
            #     reduction="none") * mask).sum() / mask.sum()

            if isinstance(prompt_model, torch.nn.DataParallel):
                loss = src_output.loss.sum() + (args.U * trg_output.loss.sum())
            else:
                loss = src_output.loss + (args.U * trg_output.loss)
            # normalize loss to account for gradient accumulation
            loss = loss / args.gradient_accum_steps
            # print('check 7')
            # torch.cuda.mem_get_info()
            # propogate backward to calculate gradients
            # loss.backward()
            if isinstance(prompt_model, torch.nn.DataParallel):
                loss.sum().backward()
            else:
                loss.backward()
            tot_loss += loss.sum().item()

            actual_step += 1
            # log loss to tensorboard  every 50 steps

            # clip gradients based on gradient accumulation steps
            if actual_step % args.gradient_accum_steps == 0:
                # log loss
                aveloss = tot_loss / (step + 1)
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
                if optimizer_prompt is not None:
                    optimizer_prompt.step()
                    optimizer_prompt.zero_grad()
                # print('check 9')
                # torch.cuda.mem_get_info()
                if scheduler_prompt is not None:
                    scheduler_prompt.step()
                # template

                # check if we are over max steps
                if glb_step > args.max_steps:
                    logger.warning("max steps reached - stopping training!")
                    leave_training = True
                    break
                # print('check 10')
                # torch.cuda.mem_get_info()
        # get epoch loss and write to tensorboard
        if args.trg_data is not None:
            epoch_loss = tot_loss / (len(src_train_data_loader) + len(trg_train_data_loader))
        else:
            epoch_loss = tot_loss / (len(src_train_data_loader))
        print("{} sec. for each epoch".format(time.time()-t0))
        if not args.no_tensorboard:
            writer.add_scalar("train/epoch_loss", epoch_loss, epoch)

        # run a run through validation set to get some metrics
        if args.crossvalidation:
            val_loss, val_acc, val_prec_weighted, val_prec_macro, val_recall_weighted, val_recall_macro, val_f1_weighted, val_f1_macro, val_auc_weighted, val_auc_macro = evaluate(
                prompt_model, test_data_loader, epoch=epoch, mode='validation')
        else:
            val_loss, val_acc, val_prec_weighted, val_prec_macro, val_recall_weighted, val_recall_macro, val_f1_weighted, val_f1_macro, val_auc_weighted, val_auc_macro = evaluate(
                prompt_model, test_data_loader, epoch=epoch, mode='test')

        if not args.no_tensorboard:
            writer.add_scalar("valid/loss", val_loss, epoch)
            writer.add_scalar("valid/balanced_accuracy", val_acc, epoch)
            writer.add_scalar("valid/precision_weighted", val_prec_weighted, epoch)
            writer.add_scalar("valid/precision_macro", val_prec_macro, epoch)
            writer.add_scalar("valid/recall_weighted", val_recall_weighted, epoch)
            writer.add_scalar("valid/recall_macro", val_recall_macro, epoch)
            writer.add_scalar("valid/f1_weighted", val_f1_weighted, epoch)
            writer.add_scalar("valid/f1_macro", val_f1_macro, epoch)

            # TODO add binary classification metrics e.g. roc/auc
            writer.add_scalar("valid/auc_weighted", val_auc_weighted, epoch)
            writer.add_scalar("valid/auc_macro", val_auc_macro, epoch)

            # # add cm to tensorboard
            # writer.add_figure("valid/Confusion_Matrix", cm_figure, epoch)
        print("Epoch {}, train loss: {}, valid loss:{}, valid acc:{} ".format(epoch, epoch_loss, val_loss, val_acc),
              flush=True)
        # save checkpoint if validation accuracy improved
        if val_acc > best_val_acc:
            # only save ckpts if no_ckpt is False - we do not always want to save - especially when developing code
            # print('Accuracy improved! Saving checkpoint')
            if not args.no_ckpt:
                print('acc improved')
                logger.warning(f"Accuracy improved! Saving checkpoint at :{ckpt_dir}!")
                if not args.crossvalidation:
                    torch.save(prompt_model.state_dict(), os.path.join(ckpt_dir, "best-checkpoint.ckpt"))
                else:
                    torch.save(prompt_model.state_dict(), os.path.join(ckpt_dir,
                                                                       "best-checkpoint_cv{}_fold{}.ckpt".format(
                                                                           run_idx, args.val_fold_idx)))
            best_val_acc = val_acc
            best_val_epoch = epoch

        if glb_step > args.max_steps:
            leave_training = True
            break

        if leave_training:
            logger.warning("Leaving training as max steps have been met!")
            break
    if args.crossvalidation:
        test_evaluation(prompt_model, ckpt_dir, test_data_loader, epoch_num=best_val_epoch)


# ## evaluate

# %%
def get_attention_weight(attention_scores, text_mask):
    # Extract attention scores with shape (batch_size, num_heads, sequence_length, sequence_length)
    # attention_scores = outputs['attentions']

    # grab attention from final layer of the model
    attention = attention_scores[-1]

    # this subsets the attention matrix to size `(12, n_query_tokens, n_response_tokens)`
    # this has to be a for loop because the values of `n_query_tokens` and `n_response_tokens` will be different for different batch items
    bs = attention.shape[0]
    extracted_attentions = []
    for i in range(bs):
        # q_mask = query_mask[i]
        t_mask = text_mask[i]

        masked_attention = attention[i, :, t_mask, :][:, :, t_mask]

        # extracted_attentions.append(masked_attention)
        # average over number of heads and query tokens
        extracted_attentions.append(torch.mean(masked_attention, (0, 1)).detach().numpy().flatten())

    # grab attention submatrix for first batch item, as an example
    # extracted_attention = extracted_attentions[0]

    # average over number of heads and query tokens
    # mean_attentions = torch.mean(extracted_attention, (0, 1)).detach().numpy().flatten()
    return extracted_attentions


def evaluate(prompt_model, dataloader, mode="validation", class_labels=class_labels, epoch=None):
    print('evaluating in %s mode' % (mode))
    prompt_model.eval()

    tot_loss = 0
    allpreds = []
    alllabels = []
    # record logits from the the model
    alllogits = []
    # store probabilties i.e. softmax applied to logits
    allscores = []

    allids = []
    attentions = []
    with torch.no_grad():
        for step, inputs in enumerate(dataloader):
            if use_cuda:
                # inputs = inputs.to(cuda_device)
                src_input_ids, src_att_mask, src_labels,src_guid = [data.to(cuda_device) for data in inputs]

            eval_output = prompt_model(input_ids=src_input_ids,attention_mask=src_att_mask)
            # print(eval_output)
            labels = src_labels
            # print(labels)
            if args.get_attention:
                # to get the attention weights
                attention_scores = eval_output.attentions
                text_mask = (src_att_mask == 1)
                attentions.append(get_attention_weight(eval_output.logits, text_mask))

            loss = loss_func(eval_output.logits, labels)
            tot_loss += loss.item()

            # add labels to list
            alllabels.extend(labels.cpu().tolist())

            # add ids to list - they are already a list so no need to send to cpu
            # allids.extend(inputs['guid'])
            allids.extend(src_guid.detach().cpu().numpy())

            # add logits to list
            alllogits.extend(eval_output.logits.cpu().tolist())
            # use softmax to normalize, as the sum of probs should be 1
            # if binary classification we just want the positive class probabilities
            if len(class_labels) > 2:
                allscores.extend(torch.nn.functional.softmax(eval_output.logits, dim=-1).cpu().tolist())
            else:
                allscores.extend(torch.nn.functional.softmax(eval_output.logits, dim=-1)[:, 1].cpu().tolist())

            # add predicted labels
            allpreds.extend(torch.argmax(eval_output.logits, dim=-1).cpu().tolist())

    val_loss = tot_loss / len(dataloader)
    # get sklearn based metrics
    acc = balanced_accuracy_score(alllabels, allpreds)
    f1_weighted = f1_score(alllabels, allpreds, average='weighted')
    f1_macro = f1_score(alllabels, allpreds, average='macro')
    prec_weighted = precision_score(alllabels, allpreds, average='weighted')
    prec_macro = precision_score(alllabels, allpreds, average='macro')
    recall_weighted = recall_score(alllabels, allpreds, average='weighted')
    recall_macro = recall_score(alllabels, allpreds, average='macro')

    # roc_auc  - only really good for binary classification but can try for multiclass too
    # use scores instead of predicted labels to give probs

    if len(class_labels) > 2:
        roc_auc_weighted = roc_auc_score(alllabels, allscores, average="weighted", multi_class="ovr")
        roc_auc_macro = roc_auc_score(alllabels, allscores, average="macro", multi_class="ovr")

    else:
        roc_auc_weighted = roc_auc_score(alllabels, allscores, average="weighted")
        roc_auc_macro = roc_auc_score(alllabels, allscores, average="macro")

        # get confusion matrix
    cm = confusion_matrix(alllabels, allpreds)

    # plot using custom function defined below
    # cm_figure = plotConfusionMatrix(cm, class_labels)
    # below makes a slightly nicer plot
    # if not args.crossvalidation:
    #     cm_figure = plot_confusion_matrix(cm, class_labels)
    epoch_dir = os.path.join(ckpt_dir, "epoch{}".format(epoch))
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
            if args.transcription in ['cnntdnn', 'fusionshujie', "sys14_26.4", "sys18_25.9"]:  # "cnntdnn":
                test_report_name = "test_{}_{}_class_report.csv".format(args.transcription, args.asr_format)
                test_results_name = "test_{}_{}_results.csv".format(args.transcription, args.asr_format)
                figure_name = "test_{}_{}_cm.png".format(args.transcription, args.asr_format)
            # elif args.transcription == "chas":
            else:
                test_report_name = "test_class_report.csv"
                test_results_name = "test_results.csv"
                figure_name = "test_cm.png"
        test_report_df.to_csv(os.path.join(epoch_dir, test_report_name), index=False)
        print(test_report_df)
        # save logits etc

        results_dict = {}
        results_dict['id'] = allids
        results_dict['labels'] = alllabels
        results_dict['pred_labels'] = allpreds
        results_dict['logits'] = alllogits
        results_dict['probas'] = allscores
        # save dataframe and to csv
        pd.DataFrame(results_dict).to_csv(os.path.join(epoch_dir, test_results_name), index=False)

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

        # save logits etc

        results_dict = {}
        results_dict['id'] = allids
        results_dict['labels'] = alllabels
        results_dict['pred_labels'] = allpreds
        results_dict['logits'] = alllogits
        results_dict['probas'] = allscores
        # save dataframe and to csv
        pd.DataFrame(results_dict).to_csv(os.path.join(ckpt_dir, test_results_name), index=False)

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
            if args.transcription in ['cnntdnn', 'fusionshujie', "sys14_26.4", "sys18_25.9"]:  # "cnntdnn":
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
            modelname = args.model_name.split('/')[1]
        else:
            modelname = args.model_name

        # logger.warning('save to {}'.format(os.path.join(cv_dir_name_name, test_report_name)))
        test_report_df.to_csv(os.path.join(epoch_dir, test_report_name), index=False)
        print('resulst saved in %s' % (epoch_dir))
        print(test_report_df)

        # save logits etc

        results_dict = {}
        results_dict['id'] = allids
        results_dict['labels'] = alllabels
        results_dict['pred_labels'] = allpreds
        results_dict['logits'] = alllogits
        results_dict['probas'] = allscores
        # save dataframe and to csv
        pd.DataFrame(results_dict).to_csv(os.path.join(epoch_dir, test_results_name), index=False)

        if args.last_ckpt:
            if not args.crossvalidation:
                torch.save(prompt_model.state_dict(), os.path.join(epoch_dir, "checkpoint.ckpt"))
            else:
                torch.save(prompt_model.state_dict(),
                           os.path.join(epoch_dir, "checkpoint_cv{}_fold{}.ckpt".format(run_idx, args.val_fold_idx)))

    return val_loss, acc, prec_weighted, prec_macro, recall_weighted, recall_macro, f1_weighted, f1_macro, roc_auc_weighted, roc_auc_macro


# TODO - add a test function to load the best checkpoint and obtain metrics on all test data. Can do this post training but may be nicer to do after training to avoid having to repeat.

def test_evaluation(prompt_model, ckpt_dir, dataloader, epoch_num=None):
    # once model is trained we want to load best checkpoint back in and evaluate on test set - then log to tensorboard and save logits and few other things to file?

    # first load the state_dict using the ckpt_dir of the best model i.e. should be best-checkpoint.ckpt found in the ckpt_dir
    if args.last_ckpt:
        assert epoch_num != None
        epoch_dir = os.path.join(ckpt_dir, "epoch{}".format(epoch_num))
        if not args.crossvalidation:
            loaded_model = torch.load(os.path.join(epoch_dir, "checkpoint.ckpt"))
        else:
            loaded_model = torch.load(
                os.path.join(epoch_dir, "checkpoint_cv{}_fold{}.ckpt".format(run_idx, args.val_fold_idx)))
    elif not args.no_ckpt:
        if not args.crossvalidation:
            loaded_model = torch.load(os.path.join(ckpt_dir, "best-checkpoint.ckpt"))
        else:
            print( os.path.join(ckpt_dir, "best-checkpoint_cv{}_fold{}.ckpt".format(run_idx, args.val_fold_idx)))
            loaded_model = torch.load(
                os.path.join(ckpt_dir, "best-checkpoint_cv{}_fold{}.ckpt".format(run_idx, args.val_fold_idx)))
    else:
        NotImplemented
    # now load this into the already create PromptForClassification object i.e. the supplied prompt_model
    prompt_model.load_state_dict(state_dict=loaded_model)
    # print("cuda_device", cuda_device)
    if use_cuda:
        prompt_model.to(cuda_device)

    # then run evaluation on test_dataloader

    if args.last_ckpt:
        test_loss, test_acc, test_prec_weighted, test_prec_macro, test_recall_weighted, test_recall_macro, test_f1_weighted, test_f1_macro, test_auc_weighted, test_auc_macro = evaluate(
            prompt_model,
            mode='validation', dataloader=dataloader, epoch=epoch_num)
    else:
        test_loss, test_acc, test_prec_weighted, test_prec_macro, test_recall_weighted, test_recall_macro, test_f1_weighted, test_f1_macro, test_auc_weighted, test_auc_macro = evaluate(
            prompt_model,
            mode='test', dataloader=dataloader, epoch=epoch_num)
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

        # TODO add binary classification metrics e.g. roc/auc
        writer.add_scalar("test/auc_weighted", test_auc_weighted, 0)
        writer.add_scalar("test/auc_macro", test_auc_macro, 0)

        # # add cm to tensorboard
        # writer.add_figure("test/Confusion_Matrix", cm_figure, 0)
    return test_acc, test_prec_macro, test_recall_macro, test_f1_macro, test_auc_macro


def last_epoch_evaluation(prompt_model, ckpt_dir, test_data_loader):
    # then run evaluation on test_dataloader

    test_loss, test_acc, test_prec_weighted, test_prec_macro, test_recall_weighted, test_recall_macro, test_f1_weighted, test_f1_macro, test_auc_weighted, test_auc_macro = evaluate(
        prompt_model,
        mode='last', dataloader=test_data_loader)


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

    zero_loss, zero_acc, zero_prec_weighted, zero_prec_macro, zero_recall_weighted, zero_recall_macro, zero_f1_weighted, zero_f1_macro, zero_auc_weighted, zero_auc_macro = evaluate(
        prompt_model, test_data_loader, mode='last')

    if not args.no_tensorboard:
        writer.add_scalar("zero_shot/loss", zero_loss, 0)
        writer.add_scalar("zero_shot/balanced_accuracy", zero_acc, 0)
        writer.add_scalar("zero_shot/precision_weighted", zero_prec_weighted, 0)
        writer.add_scalar("zero_shot/precision_macro", zero_prec_macro, 0)
        writer.add_scalar("zero_shot/recall_weighted", zero_recall_weighted, 0)
        writer.add_scalar("zero_shot/recall_macro", zero_recall_macro, 0)
        writer.add_scalar("zero_shot/f1_weighted", zero_f1_weighted, 0)
        writer.add_scalar("zero_shot/f1_macro", zero_f1_macro, 0)

        # TODO add binary classification metrics e.g. roc/auc
        writer.add_scalar("zero_shot/auc_weighted", zero_auc_weighted, 0)
        writer.add_scalar("zero_shot/auc_macro", zero_auc_macro, 0)

        # # add cm to tensorboard
        # writer.add_figure("zero_shot/Confusion_Matrix", zero_cm_figure, 0)

# run training

logger.warning(f"do training : {do_training}")
print('start')
if do_training:
    logger.warning("Beginning full training!")
    if args.trg_data is not None:
        train(prompt_model, src_train_data_loader, trg_train_data_loader, args.num_epochs, ckpt_dir)
    else:
        train(prompt_model, src_train_data_loader, args.num_epochs, ckpt_dir)

    torch.cuda.empty_cache()

elif not do_training:
    logger.warning("No training will be performed!")

# run on test set if desired

if args.run_evaluation:
    logger.warning("Running evaluation on test set using best checkpoint!")
    print('seed', args.seed)
    for epoch_num in [best_val_epoch]:
        acc, prec, recall, f1, auc = test_evaluation(prompt_model, ckpt_dir, test_data_loader, epoch_num)
        result=str(args.val_fold_idx)+" "+str(args.seed)+" "+str(acc)+" "+str(prec)+" "+str(recall)+" "+str(f1)+" "+str(auc)+"\n"
        with open(result_dir + "/" + 'result.txt', 'a+') as run_write:
            run_write.write(result)
    # last_epoch_evaluation(prompt_model, ckpt_dir, test_data_loader)

# write the contents to file
if not args.no_tensorboard:
    writer.flush()
