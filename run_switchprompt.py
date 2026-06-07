import subprocess
import os
import sys
#mlm model chalanor shomoy bert_baseline use korte hobe
#swithprompt er jonno bertprefix, prompt tuning er jonno bertprompt, ptuningv2 er jonno bertprefixv2
# CWD = sys.argv[1] # Input your working directory
CWD = './'
GPU_idx = 0
cv = True
#mode=bertprefixv2, for ptuningv2, bertprefix for switchprompt (change the prefixencofder in seqclassification.py file, promt, means prompt tuning )
COMMAND_LIST = []
meta_info = -1
tune_plm = False
# for template_id in [13,14,0,1]:
# 3,4,5,6,7,8,9,10,1,2
# [7,4,10]
for bs in [(32,32)]:  # batch_size
    # for seed in [0,1,2]:
    # for static in [32,16,8,4,2]:  # num of static keywords
    for soft_token in [64]: #num of prefix/soft prompt tokens len
        # for static in [2,4,8,16,32]: #num of static keywords
        for seed in [1,2]:
            # for fold_idx in range(5):
            for fold_idx in range(5):
                command = '''python switchprompt.py \
                        --project_root ./ \
                        --logs_root ./output/ \
                        --data_dir ./data/ \
                        --src_data  pitt_single_utt\
                        --trg_data participant_all_ccc_transcript_cut\
                        --seed {:d} \
                        --model bertprefixv2 \
                         --model_name bert-base-uncased \
                        --gpu_num {:d} \
                        --batch_size {:d} \
                        --trg_batch_size {:d} \
                        --val_file_dir latest_tmp_dir/five_fold.json \
                        --val_fold_idx {} \
                        --soft_token_num {:d} \
                        --no_tensorboard '''.format(seed, GPU_idx, bs[0], bs[1], fold_idx,
                                                    soft_token)
                COMMAND_LIST.append(command)

for k in range(len(COMMAND_LIST)):
    subp = subprocess.Popen(COMMAND_LIST[k], shell=True, cwd=CWD, encoding="utf-8")
    subp.wait()

    if subp.poll() == 0:
        print(subp.communicate())
    else:
        print(COMMAND_LIST[k], 'fail')
        with open('./running_status.txt', 'a+') as run_write:
            run_write.write(COMMAND_LIST[k] + '  fail\n')
