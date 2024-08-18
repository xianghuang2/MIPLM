import json
import os
import sys

import torch
from transformers import HfArgumentParser

from Arguments.argument_setting import ModelArguments, DataTrainingArguments, TrainingArguments
from uti.util import get_data_index


def program_setting():
    print(torch.cuda.is_available())
    print(torch.__version__)
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    # 在argument设置中指定
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments,
         TrainingArguments))
    # 是否提供json文件
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # 设置数据地址
    data_args.dataset_name = 'movie_RAG'
    with open("param.json") as f:
        params = json.load(f)
    data_index = get_data_index(data_args.dataset_name, params)
    if data_index == -1:
        pass
    param = params[data_index]
    data_args.file = param['data_path']
    # data_args.ex_file = param['ex_data_path'] if param['ex_data_path'] else None
    data_args.miss_column = param['miss_column']
    data_args.cat_column = param['cat_column']
    data_args.num_column = param['num_column']
    data_args.detail_column = param['detail_column']
    data_args.decoder_column = param['miss_column']
    data_args.missing_rate = 0.2
    training_args.num_train_epochs = param['epochs']
    training_args.use_annoy = True
    training_args.finetune_all = False
    training_args.annoy_k = 3
    return model_args, data_args, training_args, device