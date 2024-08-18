import random
from typing import *

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

class ReadCsv:
    def __init__(self, nan_data):
        df = nan_data
        df = df.fillna('')
        # 将数据转换为self.input_tuples格式
        self.input_tuples = [
            {
                'attributes': df.columns.tolist(),
                'values': row.values.tolist(),
                'record_relation': index
            }
            for index, row in df.iterrows()
        ]
    def get_item(self, index):
        line = self.input_tuples[index]
        return line


class TrainDataSet(Dataset):
    def __init__(self, tokenizer, data, true_data, type_path, data_args, max_examples=-1,
                 max_input_len=200, max_output_len=200, IsUsePromptTune = False):
        """
        max_examples: if > 0 then will load only max_examples into the dataset; -1 means use all
        max_src and max_tgt len refer to number of tokens in the input sequences
        # Note: these are not randomized. If they were we might need to collate.
        """
        valid_type_paths = ["test", "train", "val"]
        assert type_path in valid_type_paths, f"Type path must be one of {valid_type_paths}"
        self.file_path = data
        self.true_data = true_data
        self.max_examples = max_examples
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len  # max num of tokens in tokenize()
        self.max_output_len = max_output_len  # max num of tokens in tokenize()
        self.input_tuples = []  # list of dict
        self.shuffle = False
        self.data_args = data_args
        self.IsUsePromptTune = IsUsePromptTune
        self.column_indices = [data.columns.get_loc(column) for column in data_args.decoder_column]
        self._build()  # fill inputs, targets, max_lens
    def __len__(self):
        return len(self.input_tuples)
    # 对于一个元组
    def __getitem__(self, index):
        line = self.input_tuples[index]
        true_values = self.true_input_tuples[index]['values']
        attributes = [attr for attr in line['attributes']]
        values = line['values']
        record_relation = int(line['record_relation'])
        missing_column = self.data_args.miss_column
        mask_list = [0] * len(line['attributes'])
        # 对于一个元组选择其中可以mask掉的列
        non_empty_indices = [i for i, value in enumerate(values) if value != '' and attributes[i] in missing_column]
        if len(non_empty_indices) == 0:
            mask_id_list = []
        else:
            num_to_select = random.randint(1, len(non_empty_indices))
            mask_id_list = random.sample(non_empty_indices, num_to_select)
        block_flag_begin = [1] * self.max_input_len # 哪些位置以后需要替换为begin
        block_flag_end = [1] * self.max_input_len# 哪些位置以后需要替换为end

        block_flag = [1] * self.max_input_len
        attention_mask = [1] * self.max_input_len # 哪些位置是mask的
        impute_place = [1] * self.max_input_len # 哪些位置需要被填充
        input_ids = [torch.tensor([101], dtype=torch.long).unsqueeze(0)]  # 句子的所有embed
        source_input = " "
        target_output = " "
        begin = " attribute "
        begin_ids = self.tokenizer.encode(begin, add_special_tokens = False, return_tensors = "pt")
        end = " value "
        end_ids = self.tokenizer.encode(end, add_special_tokens=False, return_tensors = "pt")
        cur_len = 1 # 当前遍历的长度，去除CLS
        # 假设Prompt位置先由begin和end代替
        for i in range(len(attributes)):
            for j in range(cur_len, cur_len + begin_ids.shape[1]):
                block_flag_begin[j] = 0
                block_flag[j] = 0
            input_ids.append(begin_ids)

            att_i_ids = self.tokenizer.encode(attributes[i], add_special_tokens=False, return_tensors = "pt")
            input_ids.append(att_i_ids)
            cur_len += begin_ids.shape[1] + att_i_ids.shape[1]

            source_input += begin + (attributes[i]) + " "
            for j in range(cur_len, cur_len + end_ids.shape[1]):
                block_flag_end[j] = 0
                block_flag[j] = 0
            input_ids.append(end_ids)

            cur_len += end_ids.shape[1]
            if i in self.column_indices:
                impute_place[cur_len] = 0

            if i in mask_id_list:
                mask_list[i] = 1
                source_input += end + " Mask "
                val_i_ids =torch.tensor([self.tokenizer.mask_token_id], dtype = torch.long).unsqueeze(0)
                attention_mask[cur_len] = 0
            else:
                source_input += end + str(values[i])
                val_i_ids = self.tokenizer.encode(str(values[i]), add_special_tokens=False, return_tensors="pt")
            input_ids.append(val_i_ids)
            cur_len += val_i_ids.shape[1]

            target_output += ' attribute ' + attributes[i] + ' value ' + str(true_values[i])
        source_input += ' .'
        target_output += ' .'
        mask_list = np.array(mask_list)
        # 获取需要扩充的维度
        input_ids.append(torch.tensor([102], dtype=torch.long).unsqueeze(0))
        cur_len += 1
        padding_length = self.max_input_len - cur_len
        padding_tensor = torch.full((1, padding_length), self.tokenizer.pad_token_id, dtype=torch.long)
        input_ids.append(padding_tensor)
        input_ids = torch.cat(input_ids, dim = 1).squeeze(0).long()

        block_flag_begin = torch.tensor(block_flag_begin).long()
        block_flag_end = torch.tensor(block_flag_end).long()
        block_flag = torch.tensor(block_flag).long()

        attention_mask[cur_len:] = [0] * padding_length
        attention_mask = torch.tensor(attention_mask).long()
        token_type_ids = torch.tensor([0] * self.max_input_len).long()
        impute_place = torch.tensor(impute_place).long()

        return {
            "input_ids": input_ids,
            "block_flag_begin": block_flag_begin,
            "block_flag_end": block_flag_end,
            "block_flag": block_flag,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "impute_place": impute_place,
            "use_len": cur_len,
            "source_input": source_input,
            "target_input": target_output,
            "attributes": '#,#'.join(attributes),
            "relation_info": record_relation,  # 索引
            "mask_list": mask_list,
            "begin": begin,
            "end": end,
        }


    def _build(self):
        R = ReadCsv(self.file_path)

        self.input_tuples = [R.get_item(i) for i in range(len(R.input_tuples))]
        R_true = ReadCsv(self.true_data)
        self.true_input_tuples = [R_true.get_item(i) for i in range(len(R_true.input_tuples))]
        print("train_data_bulid...")


class TestDataSet(Dataset):
    def __init__(self, tokenizer, data, true_data, type_path, data_args, max_examples=-1,
                 max_input_len=200, max_output_len=200, IsUsePromptTune = False):
        valid_type_paths = ["test", "train", "val"]
        assert type_path in valid_type_paths, f"Type path must be one of {valid_type_paths}"
        self.data = data
        self.true_data = true_data
        self.max_examples = max_examples
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len  # max num of tokens in tokenize()
        self.max_output_len = max_output_len  # max num of tokens in tokenize()
        self.input_tuples = []  # list of dict
        self.true_input_tuples = []
        self.data_args = data_args
        self._build()  # fill inputs, targets, max_lens
        self.shuffle = False
        self.IsUsePromptTune = IsUsePromptTune
        self.column_indices = [data.columns.get_loc(column) for column in data_args.decoder_column]
    def __len__(self):
        return len(self.input_tuples)
    def __getitem__(self, index):
        line = self.input_tuples[index]
        line_true_data = self.true_input_tuples[index]
        attributes = [attr for attr in line['attributes']]
        values = line['values']
        true_values = line_true_data['values']
        record_relation = int(line['record_relation'])
        mask_id_list = []

        for index in range(len(attributes)):
            if values[index] == '':
                mask_id_list.append(index)
        mask_list = [0] * len(line['attributes'])

        block_flag_begin = [1] * self.max_input_len  # 哪些位置以后需要替换为begin
        block_flag_end = [1] * self.max_input_len  # 哪些位置以后需要替换为end
        block_flag = [1] * self.max_input_len
        attention_mask = [1] * self.max_input_len  # 哪些位置是mask的
        impute_place = [1] * self.max_input_len

        input_ids = [torch.tensor([101], dtype=torch.long).unsqueeze(0)]  # 句子的所有embed
        source_input = " "
        target_output = " "
        begin = " attribute "
        begin_ids = self.tokenizer.encode(begin, add_special_tokens=False, return_tensors="pt")
        end = " value "
        end_ids = self.tokenizer.encode(end, add_special_tokens=False, return_tensors="pt")
        cur_len = 1  # 当前遍历的长度，去除CLS
        # 假设Prompt位置先由begin和end代替
        for i in range(len(attributes)):
            for j in range(cur_len, cur_len + begin_ids.shape[1]):
                block_flag_begin[j] = 0
                block_flag[j] = 0

            input_ids.append(begin_ids)
            att_i_ids = self.tokenizer.encode(attributes[i], add_special_tokens=False, return_tensors="pt")
            input_ids.append(att_i_ids)
            cur_len += begin_ids.shape[1] + att_i_ids.shape[1]
            source_input += begin + (attributes[i]) + " "

            for j in range(cur_len, cur_len + end_ids.shape[1]):
                block_flag_end[j] = 0
                block_flag[j] = 0
            input_ids.append(end_ids)
            cur_len += end_ids.shape[1]

            if i in self.column_indices:
                impute_place[cur_len] = 0
            if i in mask_id_list:
                mask_list[i] = 1
                source_input += end + " Mask "
                val_i_ids = torch.tensor([self.tokenizer.mask_token_id], dtype=torch.long).unsqueeze(0)
                attention_mask[cur_len] = 0
            else:
                source_input += end + str(values[i])
                val_i_ids = self.tokenizer.encode(str(values[i]), add_special_tokens=False, return_tensors="pt")
            input_ids.append(val_i_ids)
            cur_len += val_i_ids.shape[1]

            target_output += ' attribute ' + attributes[i] + ' value ' + str(true_values[i])
        source_input += ' .'
        target_output += ' .'
        mask_list = np.array(mask_list)
        input_ids.append(torch.tensor([102], dtype=torch.long).unsqueeze(0))
        cur_len += 1

        padding_length = self.max_input_len - cur_len
        padding_tensor = torch.full((1, padding_length), self.tokenizer.pad_token_id, dtype=torch.long)
        input_ids.append(padding_tensor)

        input_ids = torch.cat(input_ids, dim=1).squeeze(0).long()
        block_flag_begin = torch.tensor(block_flag_begin).long()
        block_flag_end = torch.tensor(block_flag_end).long()
        block_flag = torch.tensor(block_flag).long()

        attention_mask[cur_len:] = [0] * padding_length
        attention_mask = torch.tensor(attention_mask).long()
        token_type_ids = torch.tensor([0] * self.max_input_len).long()
        impute_place = torch.tensor(impute_place).long()
        return {
            "input_ids":input_ids,
            "block_flag_begin":block_flag_begin,
            "block_flag_end":block_flag_end,
            "block_flag": block_flag,
            "attention_mask":attention_mask,
            "impute_place":impute_place,
            "token_type_ids":token_type_ids,
            "use_len": cur_len,
            "source_input": source_input,
            "target_input": target_output,
            "attributes": '#,#'.join(attributes),
            "relation_info": record_relation,  # 索引
            "mask_list": mask_list, # 缺失的地方是1
            "begin": begin,
            "end": end,
        }
    def _build(self):
        R = ReadCsv(self.data)
        self.input_tuples = [R.get_item(i) for i in range(len(R.input_tuples))]
        R_true = ReadCsv(self.true_data)
        self.true_input_tuples = [R_true.get_item(i) for i in range(len(R_true.input_tuples))]
        print("test_data_bulid...")

def get_dataloader(tokenizer, batch_size, type_path, num_examples, data, true_data, num_workers, data_args, train_args, IsUsePromptTune, shuffle_=False) -> Tuple[
    DataLoader, DataLoader]:
    data_set = TrainDataSet(tokenizer, type_path=type_path, data=data, true_data = true_data, data_args=data_args, max_examples=num_examples,
                            max_input_len=train_args.max_input_dim, max_output_len=train_args.max_out_dim, IsUsePromptTune = IsUsePromptTune)
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle_, num_workers=num_workers)
    return data_loader


def get_test_dataloader(tokenizer, batch_size, type_path, num_examples, data, true_data, num_workers, data_args, IsUsePromptTune, shuffle_=False) -> Tuple[
    DataLoader, DataLoader]:
    data_set = TestDataSet(tokenizer, type_path=type_path, data=data, true_data = true_data, data_args=data_args, max_examples=num_examples,
                           max_input_len=200, max_output_len=200, IsUsePromptTune = IsUsePromptTune)
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle_, num_workers=0)
    return data_loader


def get_data_set(tokenizers, training_args, data_args, train_nan_data, train_ori_data, valid_nan_data, valid_ori_data, IsUsePromptTune):
    # 加载训练集
    train_loader = get_dataloader(tokenizers, batch_size=training_args.per_device_train_batch_size, type_path="train",
                                  num_examples=-1, data=train_nan_data, true_data=train_ori_data,
                                  num_workers=data_args.preprocessing_num_workers, data_args=data_args,
                                  train_args=training_args, IsUsePromptTune = IsUsePromptTune, shuffle_=True)

    num_train = len(train_loader.dataset)

    # 加载验证集
    valid_loader = get_test_dataloader(tokenizers, batch_size=training_args.per_device_eval_batch_size,
                                       type_path="val",
                                       num_examples=-1, data=valid_nan_data, true_data=valid_ori_data,
                                       num_workers=data_args.preprocessing_num_workers, data_args=data_args,
                                       IsUsePromptTune = IsUsePromptTune,
                                       shuffle_=True)

    num_valid = len(valid_loader.dataset)

    # 加载测试集,减少负担
    test_loader = get_test_dataloader(tokenizers, batch_size=training_args.per_device_eval_batch_size, type_path="test",
                                      num_examples=-1, data=train_nan_data, true_data=train_ori_data,
                                      num_workers=data_args.preprocessing_num_workers, data_args=data_args, IsUsePromptTune = IsUsePromptTune,shuffle_=True)

    num_test = len(test_loader.dataset)
    return  train_loader, num_train, valid_loader, num_valid, test_loader, num_test