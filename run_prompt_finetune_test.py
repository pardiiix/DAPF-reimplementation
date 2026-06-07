import subprocess
import os
import sys
CWD = './' # Input your working directory

GPU_idx = 0

COMMAND_LIST = []
for template_id in [ 3]:
    for seed in [1]:
        command = '''python prompt_finetune.py \
                --project_root ./ \
                --logs_root ./output/ \
                --off_line_model_dir ./model/roberta-base/ \
                --data_dir ./data/ \
                --seed {:d} \
                --tune_plm \
                --model roberta \
                --model_name roberta-base \
                --template_type manual --verbalizer_type manual \
                --template_id {} \
                --gpu_num {} \
                --num_epochs 10 \
                --no_ckpt False \
                --last_ckpt \
                --no_tensorboard'''.format(seed, template_id, GPU_idx)
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
