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

def plot_wrong_spk(wrong_spk_dict, file_name):
    
    plt.figure()
    plt.bar(wrong_spk_dict.keys(), wrong_spk_dict.values(), color='lightgreen',
                            alpha=0.7, width=0.85)

    plt.xlabel('speaker ID')
    plt.ylabel('Error detect Frequency')
    plt.xticks(rotation=90)
    # plt.legend()
    plt.savefig(file_name)

def post_process_bigcross(valid_sp_dict, valid_sp_list, valid_label, mode): #valid_sp_list r valid_label er order same

    if mode == 'm_vote':
        preds_sum = []
    
        for v_sp in valid_sp_list:
            pre_arr = np.array(valid_sp_dict[v_sp])
            predict = np.sum(pre_arr)
            preds_sum.append(predict)

        half = pre_arr.size // 2

        try :

            preds_sum = np.array(preds_sum)
            preds_new = np.zeros(preds_sum.shape)

            preds_new[preds_sum > half] = 1
            metrics = [accuracy_score(valid_label, preds_new), precision_score(valid_label, preds_new), recall_score(valid_label, preds_new), f1_score(valid_label, preds_new)]

            wrong_spk_idx = np.where(np.logical_xor(valid_label, preds_new))[0]
        except Exception as e:
            print('problem ekhane')
            print(e)

    return metrics, wrong_spk_idx, preds_new

def str_to_int(str):
    return list(map(int, str.strip('[').strip(']').split(' ')))


if __name__ == "__main__": 
    # Hyper parameters shared by both
    merge_way = sys.argv[1]
    model = sys.argv[3]
    MODE = 'm_vote'
    valid_sp_len_list = []
    np.random.seed(42)


    #hyper parameters for not fixed epoch only
    accuracies_list = []
    combo_list = []

    # all_train_df = pd.read_csv('./prompt_ad_code/latest_tmp_dir' + 'train_chas_A.csv')
    # all_test_df = pd.read_csv('./prompt_ad_code/latest_tmp_dir' + 'test_chas_A.csv')
    all_test_df = pd.read_csv('data/adress-test_all.csv')

    # train_label = all_train_df.ad.values
    # test_label = all_test_df.ad.values
    # # print(test_label)
    #emni temp merge er shomoy roberta id, bert roberta merge filename_old for adress test
    test_label = all_test_df.ad.values
    # train_sp_list = all_train_df.id.values
    # test_sp_list = all_test_df.id.values
    #bert  roberta merge
    # test_sp_list = all_test_df.filename_old.values #template base ta chalanor shomor adres test e, ccc te shobshomoy
    test_sp_list=list(range(len(all_test_df)))#baseline adress


    if merge_way == 'rand_test':
        ckpt_root = sys.argv[2]
        assert 'val' not in ckpt_root
        list_acc = []
        cls_app = 'svm'
        all_wrong_speakerlist = []
        for seed in [1, 2, 10, 18, 26, 31, 32, 52, 61, 68, 70, 72, 85, 93, 94]: # 
            
            ckpt_dir = ckpt_root.rstrip('/') + '/version_{}'.format(seed)
            test_sp_dict = {}
            test_label_dict = {}

            all_result_df = pd.read_csv(os.path.join(ckpt_dir, 'checkpoints', 'test_results_last.csv'))

            for row in range(len(all_result_df)):
                test_sp_dict[all_result_df['id'][row]] = all_result_df['pred_labels'][row]
                test_label_dict[all_result_df['id'][row]] = all_result_df['labels'][row]

            cross_out0, wrong_spk_idx, _ = post_process_bigcross(test_sp_dict, test_sp_list, test_label, mode=MODE)

            print(seed, '{:.4f}'.format(cross_out0[0]))
            all_wrong_speakerlist.extend(test_sp_list[wrong_spk_idx])
        
        # count_wrong_spk = Counter(all_wrong_speakerlist)
        # with open(os.path.join(ckpt_root, 'wrong_spk.json'), 'w+') as json_w:
        #     json.dump(dict(count_wrong_spk), json_w)
        
    elif merge_way == 'rand_test_emg': #  epoch merge
        ckpt_root = sys.argv[2]
        
        list_acc = []
        template_id = [3, 1]
        with open(os.path.join('./prompt_ad_code/latest_tmp_dir', 'test_all_spk.json'), 'r') as j_read:
            bert_list_speakers = json.load(j_read)
        
        n_best_idx = [7, 8, 9]

        for tem_id in template_id:
            print('tem_id', tem_id)
            for seed in [1, 2, 10, 18, 26, 31, 32, 52, 61, 68, 70, 72, 85, 93, 94]: # 
                test_sp_dict = {s: [] for s in test_sp_list}
                ckpt_dir = os.path.join(ckpt_root, 'bert-base-uncased_tempmanual{}_verbmanual_full_100'.format(tem_id), 'version_{}'.format(seed))
                
                # test_label_dict = {}
                
                for idxes in n_best_idx:
                    all_result_df = pd.read_csv(os.path.join(ckpt_dir, 'checkpoints', 'test_results_epoch{}.csv'.format(idxes)))

                    for row in range(len(all_result_df)):
                        test_sp_dict[all_result_df['id'][row]].append(all_result_df['pred_labels'][row])
                        # test_label_dict[all_result_df['id'][row]] = all_result_df['labels'][row]
                
                latest_cross_out0, _, pre_new = post_process_bigcross(test_sp_dict, test_sp_list, test_label, mode=MODE)

                print(seed, '{:.4f}'.format(latest_cross_out0[0]))
    #eta use kortec
    # python post_process_vote.py rand_test_merge ./output/ roberta-base
    elif merge_way == 'rand_test_merge': #merging tempid resusults of diffrent version of a particular model like bert or roberta
                # -uncased
                # test_label_dict = {}

        ckpt_root = sys.argv[2] #./output/mlm/ for baseline, ./output/ for rest
        
        list_acc = []
        cls_app = 'svm'
        template_id = [4,74,6]#for adress roberta
        # template_id = [47,49,41]
        # template_id = [4, 7, 40]
        # with open(os.path.join('./prompt_ad_code/latest_tmp_dir', 'test_all_spk.json'), 'r') as j_read:
        #     bert_list_speakers = json.load(j_read)
        bert_list_speakers=test_sp_list
        n_best_idx = [-1]
        plot_acc_list = []
        plot_prec_list = []
        plot_rec_list = []
        plot_f1_list = []
        right_tie_speakerlist = []
        wrong_tie_speakerlist = []
        # for seed in [1, 2, 10, 18, 26, 31, 32, 52, 61, 68, 70, 72, 85, 93, 94]: #
        #baseline MLM voting adress test, command: python post_process_vote.py rand_test_merge ./output/mlm/ bert-base-uncased,
         #python post_process_vote.py rand_test_merge ./output/ bert-base-uncased
                #python post_process_vote.py rand_test_merge ./output/ roberta-base
        print(test_sp_list)
        crossvalidation=False
        for seed in [0,1,2]:
            test_sp_dict = {s: [] for s in test_sp_list}
            for tem_id in template_id:
            # for ckpt_name in ['last3rd','last2','last']:
                #     domain_id=2
                # else:
                #     domain_id=-1
                # 'bert-base-uncased_tempmanual7_verbmanual_epoch10_optimadamw_stk0_100_domain2_bs16_prlr0.5_joint'
                # if 'roberta' in model:
                #     model_name = model + '_tempmanual{}_verbmanual_epoch10_optimadamw_stk0_100_domain{}_bs16_prlr0.5_joint_tuneTrue'.format(tem_id, domain_id)
                #     # model_name = model+'_epoch10_optimadamw_pre0_bs16_joint_lr1e-05_cvFalse_{}'.format(ckpt_name)
                #
                # else:
                #     model_name=model+'_tempmanual{}_verbmanual_epoch10_optimadamw_stk0_100_domain{}_bs16_prlr0.5_joint'.format(tem_id,domain_id)
                # model_name= model+'_tempmanual{}_verbmanual_epoch10_optimadamw_stk2_2_domain2_bs16_prlr0.5_joint_tuneTrue_cvFalse_mFalse'.format(tem_id)
                if tem_id==6:
                    model_name=model+'_tempmanual{}_verbmanual_epoch10_optimadamw_stk0_100_domain2_bs16_prlr0.5_joint'.format(tem_id)
                elif tem_id==4:
                    model_name = model + '_tempmanual{}_verbmanual_epoch15_optimadamw_stk0_100_domain2_bs16_prlr0.5_joint'.format(
                        tem_id)
                else:
                    model_name= model+'_tempmanual{}_verbmanual_epoch10_optimadamw_stk0_100_domain2_bs16_prlr0.5_joint_cvFalse'.format(tem_id)



                print(model_name)
                # model_name = model+'_epoch10_optimadamw_pre0_bs16_joint_lr1e-05_cvFalse_{}'.format(ckpt_name)
                # _tempmanual3_verbmanual_epoch10_optimadamw_stk0_100_domain2_bs16_prlr0
                # .5
                # _joint_tuneTrue_cvFalse
                # _mFalse

                ckpt_dir = os.path.join(ckpt_root, model_name, 'version_{}'.format(seed)) #merging tempid resusults of diffrent version of a particular model like bert or roberta
                # -uncased
                # test_label_dict = {}

                for idxes in n_best_idx:
                    all_result_df = pd.read_csv(os.path.join(ckpt_dir, 'checkpoints', 'epoch{}'.format(idxes), 'test_results.csv'))
                    if 'id' not in all_result_df.columns.tolist():

                        all_result_df['id']=list(range(len(all_result_df)))
                    elif 'id' in all_result_df.columns.tolist():
                        ids = []
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
                        all_result_df = None
                    for row in range(len(all_result_df)):
                        try:
                            test_sp_dict[all_result_df['id'][row]].append(all_result_df['pred_labels'][row])
                        except:
                            filename_old = all_test_df.loc[all_test_df['filename_old'] == all_result_df['id'][row]][
                                'id'].values.tolist()[0] #ekhane filename er crresponding id int ber kortec, given speaker list int id
                            # print(filename_old)
                            test_sp_dict[filename_old].append(all_result_df['pred_labels'][row])
                        # test_label_dict[all_result_df['id'][row]] = all_result_df['labels'][row]
            
            latest_cross_out0, _, pre_new = post_process_bigcross(test_sp_dict, test_sp_list, test_label, mode=MODE)
            print(seed, '{:.4f}'.format(latest_cross_out0[0]))
            plot_acc_list.append(latest_cross_out0[0])
            plot_prec_list.append(latest_cross_out0[1])
            plot_rec_list.append(latest_cross_out0[2])
            plot_f1_list.append(latest_cross_out0[3])



            for k, t_sp in enumerate(bert_list_speakers):
                if test_sp_dict[t_sp].count(1) == len(test_sp_dict[t_sp]) // 2:
                    if pre_new[k] == int(test_label[k]):
                        right_tie_speakerlist.append(t_sp)
                    else:
                        wrong_tie_speakerlist.append(t_sp)

        with open(ckpt_root+'/combo_result.txt', 'a+') as run_write:
            run_write.write(model_name + '\n')
        # print(len(test_sp_dict['S191']))
        # print("combination of bert model's temp ids")
        print(template_id)
        print('bert  adress test n=32')
        print(sys.argv[3])
        combo_acc_arr = np.array(plot_acc_list)
        combo_acc_avg = np.mean(combo_acc_arr, axis=0)
        print('comb acc avg. {:.4f}'.format(combo_acc_avg))
        combo_acc_std = np.std(combo_acc_arr, axis=0)
        print('combo acc std {:.4f}'.format(combo_acc_std))
        combo_acc_max = np.max(combo_acc_arr, axis=0)
        print('combo acc max {:.4f}'.format(combo_acc_max))
        with open(ckpt_root + '/combo_result.txt', 'a+') as run_write:
            run_write.write("combination of bert model's temp ids" + '\n')
            run_write.write(str(template_id) + '\n')
            run_write.write("acc "+str(combo_acc_avg) +" "+str(combo_acc_std)+" "+str(combo_acc_max) +'\n')


                ################################
        combo_prec_arr = np.array(plot_prec_list)
        combo_prec_avg = np.mean(combo_prec_arr, axis=0)
        print('comb prec avg. {:.4f}'.format(combo_prec_avg))
        combo_prec_std = np.std(combo_prec_arr, axis=0)
        print('combo prec std {:.4f}'.format(combo_prec_std))
        combo_prec_max = np.max(combo_prec_arr, axis=0)
        print('combo prec max {:.4f}'.format(combo_prec_max))
        with open(ckpt_root + '/combo_result.txt', 'a+') as run_write:

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
        with open(ckpt_root + '/combo_result.txt', 'a+') as run_write:

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
        with open(ckpt_root + '/combo_result.txt', 'a+') as run_write:

            run_write.write(
                "f1 " + str(combo_f1_avg) + " " + str(combo_f1_std) + " " + str(combo_f1_max) + '\n')
    # python post_process_vote.py rand_test_robbertmg ./output/ roberta-base
    # python post_process_vote.py rand_test_robbertmg ./output/ptuningv2/ roberta-base
    elif merge_way == 'rand_test_robbertmg': #merging bert and roberta results
        ckpt_root = sys.argv[2]
        
        list_acc = []
        template_id = [1, 3]
        # with open(os.path.join('./prompt_ad_code/latest_tmp_dir', 'test_all_spk.json'), 'r') as j_read:
        #     bert_list_speakers = json.load(j_read)
        bert_list_speakers = test_sp_list
        n_best_idx = [7, 8, 9]
        plot_acc_list = []
        plot_prec_list = []
        plot_rec_list = []
        plot_f1_list = []
        right_tie_speakerlist = []
        wrong_tie_speakerlist = []
        for bert_seed in [1,2]: #[0,1, 2]
            for roberta_seed in [1,2]: #[0,1, 2]

                test_sp_dict = {s: [] for s in test_sp_list}
                # for tem_id in [(7,49),(4,47),(40,41)]:
                # for tem_id in [(7, 4), (4, 74),(40,6)]: #new roberta result 4,74,6
                for tem_id in [('bert-large-uncased', 'roberta-base')]:
                    # print(tem_id[0])
                    # print(tem_id[1])
                # for ckpt_name in ['last3rd','last2','last']:

                    if 'base' in tem_id[1]:
                        # ckpt_dir = os.path.join(ckpt_root,
                        #                         'bert-base-uncased_epoch10_optimadamw_pre64_bs32_joint_lr0.5_cvFalse',
                        #                         'version_{}'.format(bert_seed))
                        ckpt_dir_roberta = os.path.join(ckpt_root,
                                                        'roberta-base_epoch10_optimadamw_pre40_bs16_joint_lr0.5_cvFalse', 'version_{}'.format(roberta_seed))

                    if 'large' in tem_id[0]:
                            ckpt_dir = os.path.join(ckpt_root,
                                                    'bert-large-uncased_epoch10_optimadamw_pre32_bs32_joint_lr0.5_cvFalse', 'version_{}'.format(bert_seed))
                            # ckpt_dir_roberta = os.path.join(ckpt_root,
                            #                                 'roberta-large_epoch10_optimadamw_pre16_bs32_joint_lr0.5_cvFalse', 'version_{}'.format(roberta_seed))

                    # #     ckpt_dir = os.path.join(ckpt_root, 'bert-base-uncased_tempmanual{}_verbmanual_epoch10_optimadamw_stk0_100_domain2_bs16_prlr0.5_joint'.format(tem_id[0]), 'version_{}'.format(bert_seed))
                    # # ckpt_dir_roberta = os.path.join(ckpt_root, 'roberta-base_tempmanual{}_verbmanual_epoch10_optimadamw_stk0_100_domain2_bs16_prlr0.5_joint_tuneTrue_cvFalse_mFalse'.format(tem_id[1]), 'version_{}'.format(roberta_seed))
                    # if tem_id[1] == 6:
                    #     ckpt_dir_roberta=os.path.join(ckpt_root,
                    #                  'roberta-base_tempmanual{}_verbmanual_epoch10_optimadamw_stk0_100_domain2_bs16_prlr0.5_joint'.format(
                    #                      tem_id[1]), 'version_{}'.format(roberta_seed))
                    #
                    # elif tem_id[1] == 4:
                    #     ckpt_dir_roberta = os.path.join(ckpt_root,'roberta-base_tempmanual{}_verbmanual_epoch15_optimadamw_stk0_100_domain2_bs16_prlr0.5_joint'.format(
                    #                                         tem_id[1]), 'version_{}'.format(roberta_seed))
                    #
                    # elif tem_id[1] == 74:
                    #     ckpt_dir_roberta = os.path.join(ckpt_root,'roberta-base_tempmanual{}_verbmanual_epoch10_optimadamw_stk0_100_domain2_bs16_prlr0.5_joint_cvFalse'.format(
                    #                                         tem_id[1]), 'version_{}'.format(roberta_seed))
                    #

                    for idxes in [-1]:
                        all_result_df = pd.read_csv(os.path.join(ckpt_dir, 'checkpoints', 'epoch{}'.format(idxes), 'test_results.csv'))
                        all_result_df_roberta = pd.read_csv(os.path.join(ckpt_dir_roberta, 'checkpoints', 'epoch{}'.format(idxes), 'test_results.csv'))
                        ids = []
                        #tensor
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
                        # print(all_result_df['id'])
                        # print(test_sp_dict)
                        for row in range(len(all_result_df)):
                            test_sp_dict[all_result_df['id'][row]].append(all_result_df['pred_labels'][row])
                            try:
                                test_sp_dict[all_result_df_roberta['id'][row]].append(all_result_df_roberta['pred_labels'][row])
                            except:
                                # print('mismatch in id')

                                filename_old=all_test_df.loc[all_test_df['id']==all_result_df_roberta['id'][row]]['filename_old'].values.tolist()[0]
                                # print(filename_old)
                                test_sp_dict[filename_old].append(all_result_df_roberta['pred_labels'][row])

                            # test_label_dict[all_result_df['id'][row]] = all_result_df['labels'][row]

                
                latest_cross_out0, _, pre_new = post_process_bigcross(test_sp_dict, test_sp_list, test_label, mode=MODE)
                # print(seed, '{:.4f}'.format(latest_cross_out0[0]))
                print('bert+roberta temp mix')
                print(list(pre_new))
                print(np.array(pre_new))
                plot_acc_list.append(latest_cross_out0[0])
                plot_prec_list.append(latest_cross_out0[1])
                plot_rec_list.append(latest_cross_out0[2])
                plot_f1_list.append(latest_cross_out0[3])
                for k, t_sp in enumerate(bert_list_speakers):
                    if test_sp_dict[t_sp].count(1) == len(test_sp_dict[t_sp]) // 2:
                        if pre_new[k] == int(test_label[k]):
                            right_tie_speakerlist.append(t_sp)
                        else:
                            wrong_tie_speakerlist.append(t_sp)
        
        # print(len(test_sp_dict['S191']))
        # print(len(plot_acc_list))
        print(template_id)
        print('baseline adress test')
        print(sys.argv[3])
        combo_acc_arr = np.array(plot_acc_list)
        combo_acc_avg = np.mean(combo_acc_arr, axis=0)
        print('comb acc avg. {:.4f}'.format(combo_acc_avg))
        combo_acc_std = np.std(combo_acc_arr, axis=0)
        print('combo acc std {:.4f}'.format(combo_acc_std))
        combo_acc_max = np.max(combo_acc_arr, axis=0)
        print('combo acc max {:.4f}'.format(combo_acc_max))
        with open(ckpt_root + '/combo_result.txt', 'a+') as run_write:
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
        with open(ckpt_root + '/combo_result.txt', 'a+') as run_write:

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
        with open(ckpt_root + '/combo_result.txt', 'a+') as run_write:

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
        combo_arr = np.array(plot_acc_list)
        combo_avg = np.mean(combo_arr, axis=0)
        print('acc {:.4f}'.format(combo_avg))
        combo_std = np.std(combo_arr, axis=0)
        print('acc{:.4f}'.format(combo_std))
        combo_max = np.max(combo_arr, axis=0)
        print('{:.4f}'.format(combo_max))


    else:
        NotImplemented