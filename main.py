import json
import os
import sys
import pandas as pd
import torch
from annoy import AnnoyIndex
from tqdm import tqdm
from transformers import MODEL_FOR_MASKED_LM_MAPPING, HfArgumentParser
from Arguments.argument_setting import ModelArguments, DataTrainingArguments, TrainingArguments
from Arguments.program_set import program_setting
from models.model_test import test_impute_model, test_impute_model_distinct
from models.model_train import train_impute_model, only_train_prompt_model, only_train_impute_model, \
    train_impute_model_distinct
from models.model_util import get_model
from models.promptModel import PromptTuning
from uti.data_loader import get_dataloader, get_test_dataloader, get_data_set
from uti.util import get_miss_data, get_observe_embedding, build_annoy, categorical_to_code, Data_convert, \
    get_cols_by_name, get_nan_code, reconvert_data, errorLoss, get_data_index, get_data_info, printInfo, get_embedding
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def main():
    model_args, data_args, training_args, device = program_setting()
    # 原始训练集数据，缺失数据和data_m
    ori_data, nan_data, data_m = get_miss_data(data_args)
    data_number = len(ori_data)
    # 划分训练集，验证集分别为 0.8,0.2
    train_ori_data,train_nan_data,train_data_m,valid_ori_data,valid_nan_data,valid_data_m = ori_data.iloc[:int(0.8*data_number),:], nan_data.iloc[:int(0.8*data_number),:], data_m.iloc[:int(0.8*data_number),:],\
                                                                                            ori_data.iloc[int(0.8*data_number):,:], nan_data.iloc[int(0.8*data_number):,:], data_m.iloc[int(0.8*data_number):,:]
    # true_data_row = data_m[data_m.all(axis=1)].index
    # other_row_indices = data_m.index.difference(true_data_row)
    # # 带缺失的数据，即需要填充的数据
    # # ground_truth, test_data, test_m = ori_data.loc[other_row_indices], nan_data.loc[other_row_indices], data_m.loc[other_row_indices]
    # valid_use_index = [i for i in other_row_indices if i in valid_ori_data.index]
    # # valid_ground_truth, valid_data, valid_m = valid_ori_data.loc[valid_use_index], valid_nan_data.loc[valid_use_index], valid_data_m.loc[valid_use_index]
    # 获取missing_column中每列对应的维度
    decoder_column, value_cat, continuous_cols, fields, fields_dict, enc, nan_feed_data, M = get_data_info(data_args, nan_data, ori_data, data_m, device)


    IsUsePromptTune = False
    annoy_k = 5
    for i in range(2, 3):
        if i == 0:
            setting = {'use_annoy': False, 'finetune_all': False}
        elif i == 1:
            setting = {'use_annoy': True, 'finetune_all': False}
        elif i == 2:
            setting = {'use_annoy': False, 'finetune_all': True}
        elif i == 3:
            setting = {'use_annoy': True, 'finetune_all': True}
        # 设置是否使用Prompt——tune
        # 把没有缺失数据进行embedding到Annoy中， model为Bert-model
        tokenizers, config, bert_encoder, bert_impute = get_model(model_path='bertt', IsUsePromptTune = IsUsePromptTune, fields=fields, setting=setting, attr_num=ori_data.shape[1],annoy_k = annoy_k,
                                                                  device=device)
        # 构建Annoy
        if setting['use_annoy']:
            # annoy_path, dim, meta, dict_path = 'annoy/buy_observed.ann', 768, 'angular','annoy/buy_dict.json'
            # with open(dict_path, 'r') as handle:
            #     index_annoy_dict = json.load(handle)
            # index_annoy_dict = {int(k): v for k, v in index_annoy_dict.items()}
            # annoy = AnnoyIndex(dim, meta) # 768, angular,observed.ann
            # annoy.load(annoy_path)
            # observe_embedding, index_annoy_dict, tokenizer_data = get_observe_embedding(bert_encoder, tokenizers, data_args.max_seq_length, nan_data, data_m)
            a = data_m.copy()
            # a[:] = 1
            observe_embedding, index_annoy_dict, tokenizer_data = get_observe_embedding(bert_encoder, tokenizers, data_args.max_seq_length, ori_data, a)
            annoy, annoy_path, dim, meta = build_annoy(observe_embedding, num_tree=20)
        else:
            index_annoy_dict = {}
            tokenizer_data = []
            annoy = None

        train_loader, num_train, valid_loader, num_valid, test_loader, num_test = get_data_set(tokenizers, training_args, data_args,
                                                                                               train_nan_data, train_ori_data,
                                                                                               valid_nan_data, valid_ori_data,
                                                                                               IsUsePromptTune)

        printInfo(num_train, training_args)

        best_impute_model, bert_impute = train_impute_model(bert_impute, training_args, train_loader, valid_loader,
                       continuous_cols, value_cat,  decoder_column,
                       nan_feed_data, ori_data, M,
                       valid_nan_data, valid_ori_data, valid_data_m,
                       train_nan_data, train_ori_data, train_data_m,
                       fields, fields_dict, enc,
                    annoy, index_annoy_dict, device, setting['finetune_all'])

        # test
        path = 'best_impute_model.pth'
        RMSE, ACC  = test_impute_model(best_impute_model, num_test, test_loader, path,
                       continuous_cols, value_cat,  decoder_column,
                       nan_feed_data, ori_data,
                       train_nan_data, train_ori_data,
                      train_data_m, M, fields,  fields_dict, enc,
                       annoy, index_annoy_dict)

        print("数据集为：{}, 缺失率为：{}, RMSE为：{:.4f}，Acc为：{:.2f}, i为:{}".format(data_args.dataset_name, data_args.missing_rate, RMSE, ACC, i))



if __name__ == '__main__':
    # df = pd.read_excel('datasets/data_imputation/movie/movie_rag.xlsx')
    # df.to_excel('datasets/data_imputation/movie/movie_rag_id.xlsx')
    # df = df.dropna(subset=['type'])
    # all = []
    # for index, row in df.iterrows():
    #     if '(' in row['type']:
    #         row['type'] = row['type'].split(' ')[0]
    #     elif 'or ' or 'and ' in row['type']:
    #         row['type'] = row['type'].split(' ')[-1]
    #     elif '/' in row['type']:
    #         row['type'] = row['type'].split('/')[0]
    #     if '/' in row['type']:
    #         row['type'] = row['type'].split('/')[0]
    #     all.append(row)
    # all = pd.DataFrame(all)
    # all.to_excel('datasets/data_imputation/Restaurant/restaurant.xlsx', index=False)
    main()
