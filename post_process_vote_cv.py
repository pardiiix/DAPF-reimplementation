import pandas as pd
import sys
import os
import bisect
import numpy as np
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_fscore_support
import re
import random
#prepare to compute n best 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from collections import Counter
import matplotlib.pyplot as plt


def get_validation_idx(current_model_dir):
    # current model dir example: new_models/bert_post_train_total_loss_lr000005_1_cv3_fold2
    data_save_dir = './prompt_ad_code/latest_tmp_dir'
    model_dir_tails = current_model_dir.split('lr')[-1]
    validation_info = model_dir_tails.split('_')[2:]
    if 'cv' in validation_info[0]:
        #validation_mode = cv
        validation_idx_file = os.path.join(data_save_dir, 'ten_fold_{}.json'.format(validation_info[0].lstrip('cv')))
        fold_idx = int(validation_info[-1].lstrip('fold'))
    else:
        raise ValueError('validation info wrong')

    with open(validation_idx_file, 'r') as v_read:
        validation_dict = json.load(v_read)
    
    if fold_idx in [idx for idx in range(0, 10)]:
        train_speaker = validation_dict[fold_idx]['train_speaker']
        validation_speaker = validation_dict[fold_idx]['test_speaker']
        return train_speaker, validation_speaker
    else:
        raise ValueError('validation index wrong')

def plot_wrong_spk(wrong_spk_dict, file_name):
    
    plt.figure()
    plt.bar(wrong_spk_dict.keys(), wrong_spk_dict.values(), color='lightgreen',
                            alpha=0.7, width=0.85)

    plt.xlabel('speaker ID')
    plt.ylabel('Error detect Frequency')
    plt.xticks(rotation=90)
    # plt.legend()
    plt.savefig(file_name)

def post_process_bigcross(valid_sp_dict, valid_sp_list, valid_label, mode):
    # print(valid_sp_dict)
    # print(valid_sp_list)
    if mode == 'm_vote':
        preds_sum = []
    
        for v_sp in valid_sp_list:
            pre_arr = np.array(valid_sp_dict[v_sp])
            predict = np.sum(pre_arr)
            preds_sum.append(predict)

        half = pre_arr.size // 2


        preds_sum = np.array(preds_sum)
        preds_new = np.zeros(preds_sum.shape)
        preds_new[preds_sum > half] = 1
        metrics = [accuracy_score(valid_label, preds_new), precision_score(valid_label, preds_new), recall_score(valid_label, preds_new), f1_score(valid_label, preds_new)]

        wrong_spk_idx = np.where(np.logical_xor(valid_label, preds_new))[0]

    return metrics, wrong_spk_idx, preds_new

def str_to_int(str):
    return list(map(int, str.strip('[').strip(']').split(' ')))


if __name__ == "__main__": 
    # Hyper parameters shared by both
    merge_way = sys.argv[1]
    model = sys.argv[3] #model name
    MODE = 'm_vote'
    valid_sp_len_list = []
    np.random.seed(42)


    #hyper parameters for not fixed epoch only
    accuracies_list = []
    combo_list = []
    all_train_df = pd.read_csv('./data/' + 'participant_all_ccc_transcript_cut.csv')
    # all_train_df = pd.read_csv('./prompt_ad_code/latest_tmp_dir' + 'train_chas_A.csv')
    # all_test_df = pd.read_csv('./prompt_ad_code/latest_tmp_dir' + 'test_chas_A.csv')
    train_label = all_train_df.ad.values
    # train_label = all_train_df.ad.values
    # test_label = all_test_df.ad.values
    # print(test_label)
    train_sp_list = all_train_df.filename.values #bert roberta cv combo
    # train_sp_list = all_train_df.id.values
    # test_sp_list = all_test_df.id.values


    if merge_way == 'rand_cv':
        ckpt_dir = sys.argv[2] #
        list_acc = []
        cls_app = 'svm'
        valid_sp_dict = {}
        valid_label_dict = {}
        for i in [1]:
            for j in range(10):
                all_result_df = pd.read_csv(os.path.join(ckpt_dir, 'checkpoints', 'test_results_last_cv{}_fold{}.csv'.format(i, j)))

                for row in range(len(all_result_df)):
                    valid_sp_dict[all_result_df['id'][row]] = all_result_df['pred_labels'][row]
                    valid_label_dict[all_result_df['id'][row]] = all_result_df['labels'][row]

            # for v_sp in valid_sp_dict:
            #     assert len(valid_sp_dict[v_sp]) == 10
            cross_out0, _, _ = post_process_bigcross(valid_sp_dict, train_sp_list, train_label, mode=MODE)
            
            f_out_str = ''
            for m in cross_out0:
                f_out_str += '{:.4f} & '.format(m)
            
            print(f_out_str)

    #eta
    #command python post_process_vote_cv.py rand_cv_emg ./output/mlm/ bert-base-uncased
    #python post_process_vote_cv.py rand_cv_emg ./output/ bert-base-uncased
    #python post_process_vote_cv.py rand_cv_emg ./output/switchprompt/ bert-base-uncased
    #python post_process_vote_cv.py rand_cv_robbertmg ./output/ptuningv2/ bert-base-uncased
    elif merge_way == 'rand_cv_emg':
        ckpt_root = sys.argv[2] #root ./output/mlm/ for baseline  ./output/  rest ./output/switchprompt/

        list_acc = []
        # template_id = [47]
        # template_id = [47, 49, 41]
        template_id = [6, 73, 49]

        # n_best_idx = [47, 49, 41]
        plot_acc_list = []
        plot_prec_list = []
        plot_rec_list = []
        plot_f1_list = []
        for seed in [0,1,2]: #

            valid_sp_dict = {s: [] for s in train_sp_list}
            #['last3rd', 'last2', 'last']
            # for ckpt_name in ['last3rd', 'last2', 'last']:

            for tem_id in template_id:
                # ckpt_dir = os.path.join(ckpt_root, 'bert-base-uncased_tempmanual{}_verbmanual_full_lr100'.format(tem_id), 'version_{}_val'.format(seed))
                #baseline
                # ckpt_dir = os.path.join(ckpt_root,
                #                         model + '_epoch10_optimadamw_pre0_bs16_joint_lr1e-05_cvTrue_{}'.format(ckpt_name),
                #                         'version_{}_val'.format(seed))
                #dapf
                #agerta
                # ckpt_dir = os.path.join(ckpt_root,
                #                         model + '_tempmanual{}_verbmanual_epoch10_optimadamw_stk0_100_domain2_bs16_prlr0.5_joint_tuneTrue_cvTrue_mFalse'.format(tem_id),
                #                         'version_{}_val'.format(seed))
                #latest roberta
                if 'roberta' in model and tem_id==6 and seed==0:
                    continue
                if 'roberta' in model and tem_id==73 and seed!=0:
                    continue
                ckpt_dir = os.path.join(ckpt_root,
                                        model + '_tempmanual{}_verbmanual_epoch10_optimadamw_stk0_100_domain2_bs16_prlr0.5_joint_cvTrue'.format(tem_id),
                                        'version_{}_val'.format(seed))

                # model_name = model + '_epoch10_optimadamw_pre0_bs16_joint_lr1e-05_cvFalse_{}'.format(ckpt_name)
                # for i in [1]:
                for idxes in [-1]:
                    for j in range(5):
                        file_path = os.path.join(ckpt_dir, 'checkpoints/epoch{}'.format(idxes), 'test_results_cv1_fold{}.csv'.format(j))
                        print(file_path)
                        if not os.path.exists(file_path):
                            print(file_path)
                            continue
                        all_result_df = pd.read_csv(file_path)
                        ids=[]
                        for id,row in all_result_df.iterrows():

                            id=str(row['id'])
                            if 'tensor' in  id:
                                temp=str(row['id'])
                                num=int(temp.split('(')[1].split(')')[0])
                                ids.append(num)
                            else:
                                break
                        if len(ids)==len(all_result_df) :
                            all_result_df=all_result_df.drop(['id'],axis=1)
                            all_result_df['id']=ids

                        # print(all_result_df['id'].head(2))
                        if 'id' not in all_result_df.columns.tolist():
                            print('no id')

                            continue
                        print('id ase %s'%(file_path))
                        for row in range(len(all_result_df)):

                            valid_sp_dict[all_result_df['id'][row]].append(all_result_df['pred_labels'][row])

            # for v_sp in valid_sp_dict:
            #     assert len(valid_sp_dict[v_sp]) == 10
            print(valid_sp_dict)
            cross_out0, _, _ = post_process_bigcross(valid_sp_dict, train_sp_list, train_label, mode=MODE)
            # print(valid_sp_dict['S083'])
            print(seed, '{:.4f}'.format(cross_out0[0]))
            # plot_acc_list.append(cross_out0[0])
            plot_acc_list.append(cross_out0[0])
            plot_prec_list.append(cross_out0[1])
            plot_rec_list.append(cross_out0[2])
            plot_f1_list.append(cross_out0[3])
        print('cv combo')
        print(template_id)
        print(sys.argv[3])
        combo_acc_arr = np.array(plot_acc_list)
        combo_acc_avg = np.mean(combo_acc_arr, axis=0)
        print('comb acc avg. {:.4f}'.format(combo_acc_avg))
        combo_acc_std = np.std(combo_acc_arr, axis=0)
        print('combo acc std {:.4f}'.format(combo_acc_std))
        combo_acc_max = np.max(combo_acc_arr, axis=0)
        print('combo acc max {:.4f}'.format(combo_acc_max))
        with open(ckpt_root + '/combo_result_cv.txt', 'a+') as run_write:
            run_write.write("combination of bert model's temp ids" + '\n')
            run_write.write(str(template_id) + '\n')
            run_write.write("acc " + str(combo_acc_avg) + " " + str(combo_acc_std) + " " + str(combo_acc_max) + '\n')

            ################################
        combo_prec_arr = np.array(plot_prec_list)
        combo_prec_avg = np.mean(combo_prec_arr, axis=0)
        print('comb prec avg. {:.4f}'.format(combo_prec_avg))
        combo_prec_std = np.std(combo_prec_arr, axis=0)
        print('combo prec std {:.4f}'.format(combo_prec_std))
        combo_prec_max = np.max(combo_prec_arr, axis=0)
        print('combo prec max {:.4f}'.format(combo_prec_max))
        with open(ckpt_root + '/combo_result_cv.txt', 'a+') as run_write:

            run_write.write(
                "prec " + str(combo_prec_avg) + " " + str(combo_prec_std) + " " + str(combo_prec_max) + '\n')
            ###########################
        combo_rec_arr = np.array(plot_rec_list)
        combo_rec_avg = np.mean(combo_rec_arr, axis=0)
        print('comb rec avg. {:.4f}'.format(combo_rec_avg))
        combo_rec_std = np.std(combo_rec_arr, axis=0)
        print('combo rec std {:.4f}'.format(combo_rec_std))
        combo_rec_max = np.max(combo_rec_arr, axis=0)
        print('combo rec max {:.4f}'.format(combo_rec_max))
        with open(ckpt_root + '/combo_result_cv.txt', 'a+') as run_write:

            run_write.write(
                "rec " + str(combo_rec_avg) + " " + str(combo_rec_std) + " " + str(combo_rec_max) + '\n')
            ###########################
        combo_f1_arr = np.array(plot_f1_list)
        combo_f1_avg = np.mean(combo_f1_arr, axis=0)
        print('comb f1 avg. {:.4f}'.format(combo_f1_avg))
        combo_f1_std = np.std(combo_f1_arr, axis=0)
        print('combo f1 std {:.4f}'.format(combo_f1_std))
        combo_f1_max = np.max(combo_f1_arr, axis=0)
        print('combo f1 max {:.4f}'.format(combo_f1_max))
        with open(ckpt_root + '/combo_result_cv.txt', 'a+') as run_write:

            run_write.write(
                "f1 " + str(combo_f1_avg) + " " + str(combo_f1_std) + " " + str(combo_f1_max) + '\n')


    elif merge_way == 'rand_cv_merge':
        ckpt_root = sys.argv[2]
        
        list_acc = []
        cls_app = 'svm'
        template_id = [1, 3]

        n_best_idx = [7, 8, 9]
        # backbone = sys.argv[3]
        plot_acc_list = []

        for seed in [1, 2, 10, 18, 26, 31, 32, 52, 61, 68, 70, 72, 85, 93, 94]: # 
            valid_sp_dict = {s: [] for s in train_sp_list}
            for tem_id in template_id:
                ckpt_dir = os.path.join(ckpt_root, 'roberta-base_tempmanual{}_verbmanual_full_100'.format(tem_id), 'version_{}_val'.format(seed))
                # bert-base-uncased roberta-base
                # test_label_dict = {}

                for idxes in n_best_idx:
                    for j in range(10):
                        file_path = os.path.join(ckpt_dir, 'checkpoints/epoch{}'.format(idxes), 'test_results_cv1_fold{}.csv'.format(j))
                        all_result_df = pd.read_csv(file_path)

                        for row in range(len(all_result_df)):
                            valid_sp_dict[all_result_df['id'][row]].append(all_result_df['pred_labels'][row])
            
            
            cross_out0, _, pre_new = post_process_bigcross(valid_sp_dict, train_sp_list, train_label, mode=MODE)
            print(seed, '{:.4f}'.format(cross_out0[0]))
            plot_acc_list.append(cross_out0[0])
        
        combo_arr = np.array(plot_acc_list)
        combo_avg = np.mean(combo_arr, axis=0)
        print(combo_avg)
        print(combo_avg * 225 * 108)
        combo_std = np.std(combo_arr, axis=0)
        print('{:.4f}'.format(combo_std))
        combo_max = np.max(combo_arr, axis=0)
        print('{:.4f}'.format(combo_max))
        

    elif merge_way == 'rand_cv_robbertmg':
        from pathlib import Path
        ckpt_root = sys.argv[2]
        
        list_acc = []
        # template_id = [47, 49,41]#bert cv
        template_id=[(47, 6), (49, 73), (41, 71)]#bert+roberta

        n_best_idx = [7, 8, 9]
        plot_acc_list = []
        plot_prec_list = []
        plot_rec_list = []
        plot_f1_list = []



        for bert_seed in [1, 2]: #
            for roberta_seed in [1, 2]: #
                valid_sp_dict = {s: [] for s in train_sp_list}
                # for tem_id in template_id:
                # for tem_id in [(47,6),(49,73),(41,71)]:
                for tem_id in [('bert-large-uncased', 'roberta-base')]:
                # for ckpt_name in ['last3rd', 'last2', 'last']:
                    if 'base' in tem_id[1]:
                        # ckpt_dir = os.path.join(ckpt_root,
                        #                         'bert-base-uncased_epoch10_optimadamw_pre16_bs16_joint_lr0.5_cvTrue',
                        #                         'version_{}_val'.format(bert_seed))
                        ckpt_dir_roberta = os.path.join(ckpt_root,
                                                        'roberta-base_epoch10_optimadamw_pre16_bs16_joint_lr0.5_cvTrue',
                                                        'version_{}_val'.format(roberta_seed))

                    if 'large' in tem_id[0]:
                        ckpt_dir = os.path.join(ckpt_root,
                                                'bert-large-uncased_epoch10_optimadamw_pre16_bs16_joint_lr0.5_cvTrue',
                                                'version_{}_val'.format(bert_seed))
                        # ckpt_dir_roberta = os.path.join(ckpt_root,
                        #                                 'roberta-large_epoch10_optimadamw_pre16_bs16_joint_lr0.5_cvTrue',
                        #                                 'version_{}_val'.format(roberta_seed))

                    #     ckpt_dir = os.path.join(ckpt_root, 'bert-base-uncased_tempmanual{}_verbmanual_epoch10_optimadamw_stk0_100_domain2_bs16_prlr0.5_joint_tuneTrue_cvTrue_mFalse'.format(tem_id[0]), 'version_{}_val'.format(bert_seed))
                #     # ckpt_dir_roberta = os.path.join(ckpt_root, 'roberta-base_tempmanual{}_verbmanual_full_100'.format(tem_id), 'version_{}_val'.format(roberta_seed))
                #     if tem_id[1] in [49,47,41]:
                #         ckpt_dir_roberta = os.path.join(ckpt_root,
                #                                         'roberta-base_tempmanual{}_verbmanual_epoch10_optimadamw_stk0_100_domain2_bs16_prlr0.5_joint_tuneTrue_cvTrue_mFalse'.format(
                #                                             tem_id[1]),
                #                                         'version_{}_val'.format(roberta_seed))
                #
                #     else:
                #         ckpt_dir_roberta = os.path.join(ckpt_root,
                #                              'roberta-base_tempmanual{}_verbmanual_epoch10_optimadamw_stk0_100_domain2_bs16_prlr0.5_joint_cvTrue'.format(
                #                                 tem_id[1]),
                #                             'version_{}_val'.format(roberta_seed))
                    #baseline
                    # ckpt_dir = os.path.join(ckpt_root,
                    #                         'bert-base-uncased' + '_epoch10_optimadamw_pre0_bs16_joint_lr1e-05_cvTrue_{}'.format(ckpt_name),
                    #                         'version_{}_val'.format(bert_seed))
                    # if 'roberta' in model and tem_id[1]==6 and roberta_seed==0:
                    #     continue
                    # if 'roberta' in model and tem_id[1]==73 and roberta_seed!=0:
                    #     continue
                    # ckpt_dir_roberta = os.path.join(ckpt_root,
                    #                         'roberta-base' + '_epoch10_optimadamw_pre0_bs16_joint_lr1e-05_cvTrue_{}'.format(ckpt_name),
                    #                         'version_{}_val'.format(roberta_seed))
                    # test_label_dict = {}
                    not_present=[0,0]
                    if not os.path.exists(ckpt_dir):
                        print(ckpt_dir)
                        not_present[0]=1
                        # continue
                    if not os.path.exists(ckpt_dir_roberta):

                        print(ckpt_dir_roberta)
                        not_present[1] = 1
                        # continue
                    if np.sum(np.array(not_present))==2:
                        print('both absent')
                        continue
                    if not_present[1] == 0:
                        print('roberta present')
                        print(bert_seed)
                        print(roberta_seed)
                    if not_present[0] == 0 and not_present[1] == 0:
                        print('both present')
                        print(tem_id)
                        print(bert_seed)
                        print(roberta_seed)
                    for idxes in [-1]:
                        for j in range(5):
                            all_result_df=None
                            all_result_df_roberta=None
                            if not_present[0]==0:
                                my_file = Path(os.path.join(ckpt_dir, 'checkpoints/epoch{}'.format(idxes), 'test_results_cv1_fold{}.csv'.format(j)))
                                print('check')
                                print(my_file.is_file())
                                if my_file.is_file():
                                    all_result_df = pd.read_csv(os.path.join(ckpt_dir, 'checkpoints/epoch{}'.format(idxes), 'test_results_cv1_fold{}.csv'.format(j)))
                                    if 'id' in all_result_df.columns.tolist():
                                        ids=[]
                                        for id, row in all_result_df.iterrows():

                                            id = str(row['id'])
                                            if 'tensor' in id:
                                                temp = str(row['id'])
                                                num = int(temp.split('(')[1].split(')')[0])
                                                ids.append(num)
                                            else:
                                                break
                                        if len(ids) == len(all_result_df):
                                            all_result_df = all_result_df.drop(['id'], axis=1)
                                            all_result_df['id'] = ids
                                    else:
                                        all_result_df=None
                            if not_present[1]==0 :


                                my_file = Path(os.path.join(ckpt_dir_roberta, 'checkpoints/epoch{}'.format(idxes), 'test_results_cv1_fold{}.csv'.format(j)))
                                if my_file.is_file():

                                    all_result_df_roberta = pd.read_csv(os.path.join(ckpt_dir_roberta, 'checkpoints/epoch{}'.format(idxes), 'test_results_cv1_fold{}.csv'.format(j)))
                                    if 'id' in all_result_df_roberta.columns.tolist():

                                        ids=[]
                                        for id, row in all_result_df_roberta.iterrows():

                                            id = str(row['id'])
                                            if 'tensor' in id:
                                                temp = str(row['id'])
                                                num = int(temp.split('(')[1].split(')')[0])
                                                ids.append(num)
                                            else:
                                                break
                                        if len(ids) == len(all_result_df_roberta):
                                            all_result_df_roberta = all_result_df_roberta.drop(['id'], axis=1)
                                            all_result_df_roberta['id'] = ids
                                    else:
                                        all_result_df_roberta=None
                            df=None
                            if not_present[0] == 0 and all_result_df is not None:
                                df=all_result_df
                            elif not_present[1] == 0 and all_result_df_roberta is not None:
                                df=all_result_df_roberta
                            if df is None:
                                continue
                            for row in range(len(df)):
                                if not_present[0] == 0 and all_result_df is not None:
                                    valid_sp_dict[all_result_df['id'][row]].append(all_result_df['pred_labels'][row])
                                # if 'roberta' in model and tem_id[1]==6 and roberta_seed==0:
                                #     continue
                                # if 'roberta' in model and tem_id[1]==73 and roberta_seed!=0:
                                #     continue
                                if not_present[1] == 0 and all_result_df_roberta is not None:
                                    # valid_sp_dict[all_result_df_roberta['id'][row]].append(all_result_df_roberta['pred_labels'][row])

                                    if not(tem_id[1]==6 and roberta_seed==0) :
                                        print('bad tem_id[1]==6 roberta_seed==0')
                                    elif not(tem_id[1]==73 and roberta_seed!=0):
                                        print('tem_id[1]==73 and roberta_seed!=0')
                                    else:
                                        valid_sp_dict[all_result_df_roberta['id'][row]].append(all_result_df_roberta['pred_labels'][row])
                                # for row in range(len(all_result_df_roberta)):
                # print(bert_seed)
                # print(roberta_seed)
                # print(valid_sp_dict)
                cross_out0, _, pre_new = post_process_bigcross(valid_sp_dict, train_sp_list, train_label, mode=MODE)
                # print(seed, '{:.4f}'.format(cross_out0[0]))
                plot_acc_list.append(cross_out0[0])
                plot_prec_list.append(cross_out0[1])
                plot_rec_list.append(cross_out0[2])
                plot_f1_list.append(cross_out0[3])
        print('cv combo'+"bert+roberta")
        print(template_id)
        print(sys.argv[3])
        combo_acc_arr = np.array(plot_acc_list)
        combo_acc_avg = np.mean(combo_acc_arr, axis=0)
        print('comb acc avg. {:.4f}'.format(combo_acc_avg))
        combo_acc_std = np.std(combo_acc_arr, axis=0)
        print('combo acc std {:.4f}'.format(combo_acc_std))
        combo_acc_max = np.max(combo_acc_arr, axis=0)
        print('combo acc max {:.4f}'.format(combo_acc_max))
        with open(ckpt_root + '/combo_result_cv.txt', 'a+') as run_write:
            run_write.write("combination of bert model's temp ids" + '\n')
            run_write.write(str(template_id) + '\n')
            run_write.write(
                "acc " + str(combo_acc_avg) + " " + str(combo_acc_std) + " " + str(combo_acc_max) + '\n')

            ################################
        combo_prec_arr = np.array(plot_prec_list)
        combo_prec_avg = np.mean(combo_prec_arr, axis=0)
        print('comb prec avg. {:.4f}'.format(combo_prec_avg))
        combo_prec_std = np.std(combo_prec_arr, axis=0)
        print('combo prec std {:.4f}'.format(combo_prec_std))
        combo_prec_max = np.max(combo_prec_arr, axis=0)
        print('combo prec max {:.4f}'.format(combo_prec_max))
        with open(ckpt_root + '/combo_result_cv.txt', 'a+') as run_write:

            run_write.write(
                "prec " + str(combo_prec_avg) + " " + str(combo_prec_std) + " " + str(combo_prec_max) + '\n')
            ###########################
        combo_rec_arr = np.array(plot_rec_list)
        combo_rec_avg = np.mean(combo_rec_arr, axis=0)
        print('comb rec avg. {:.4f}'.format(combo_rec_avg))
        combo_rec_std = np.std(combo_rec_arr, axis=0)
        print('combo rec std {:.4f}'.format(combo_rec_std))
        combo_rec_max = np.max(combo_rec_arr, axis=0)
        print('combo rec max {:.4f}'.format(combo_rec_max))
        with open(ckpt_root + '/combo_result_cv.txt', 'a+') as run_write:

            run_write.write(
                "rec " + str(combo_rec_avg) + " " + str(combo_rec_std) + " " + str(combo_rec_max) + '\n')
            ###########################
        combo_f1_arr = np.array(plot_f1_list)
        combo_f1_avg = np.mean(combo_f1_arr, axis=0)
        print('comb f1 avg. {:.4f}'.format(combo_f1_avg))
        combo_f1_std = np.std(combo_f1_arr, axis=0)
        print('combo f1 std {:.4f}'.format(combo_f1_std))
        combo_f1_max = np.max(combo_f1_arr, axis=0)
        print('combo f1 max {:.4f}'.format(combo_f1_max))
        with open(ckpt_root + '/combo_result_cv.txt', 'a+') as run_write:

            run_write.write(
                "f1 " + str(combo_f1_avg) + " " + str(combo_f1_std) + " " + str(combo_f1_max) + '\n')

        print(len(plot_acc_list))
        combo_arr = np.array(plot_acc_list)
        combo_avg = np.mean(combo_arr, axis=0)
        print('{:.4f}'.format(combo_avg))
        # print(combo_avg)
        # print(combo_avg * 225 * 108)
        combo_std = np.std(combo_arr, axis=0)
        print('{:.4f}'.format(combo_std))
        combo_max = np.max(combo_arr, axis=0)
        print('{:.4f}'.format(combo_max))
        


    else:
        NotImplemented