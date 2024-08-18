import random
from typing import *

import numpy as np
import pandas as pd
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
    def __init__(self, tokenizer, data, type_path, data_args, max_examples=-1,
                 max_input_len=200, max_output_len=200):
        """
        max_examples: if > 0 then will load only max_examples into the dataset; -1 means use all
        max_src and max_tgt len refer to number of tokens in the input sequences
        # Note: these are not randomized. If they were we might need to collate.
        """
        valid_type_paths = ["test", "train", "val"]
        assert type_path in valid_type_paths, f"Type path must be one of {valid_type_paths}"
        self.file_path = data
        self.max_examples = max_examples
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len  # max num of tokens in tokenize()
        self.max_output_len = max_output_len  # max num of tokens in tokenize()
        self.input_tuples = []  # list of dict
        self.shuffle = False
        self.data_args = data_args
        self._build()  # fill inputs, targets, max_lens
    def __len__(self):
        return len(self.input_tuples)
    # 对于一个元组
    def __getitem__(self, index):
        line = self.input_tuples[index]
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
        mask_id_list = []
        source_input = ''
        target_output = ''
        target_values = values
        positions = [i for i in range(len(attributes))]
        for i in positions:
            if i in mask_id_list:
                mask_list[i] = 1
                source_input += ' attribute ' + attributes[i] + ' value <mask>'
            else:
                source_input += ' attribute ' + attributes[i] + ' value ' + str(values[i])
            target_output += ' attribute ' + attributes[i] + ' value ' + str(values[i])
        source_input += ' .'

        target_output += ' .'
        tokenized_inputs = self.tokenizer(
            [source_input], max_length=self.max_input_len, padding="max_length", return_tensors="pt", truncation=True
        )
        inputs = tokenized_inputs['input_ids'].squeeze()
        mask_inputs = tokenized_inputs['attention_mask'].squeeze()
        mask_list = np.array(mask_list)
        return {"inputs": inputs, # 词汇表的id
                "attributes": '#,#'.join(attributes),
                "mask_inputs": mask_inputs,
                "relation_info": record_relation, # 索引
                "mask_list": mask_list,
                }
    def _build(self):
        R = ReadCsv(self.file_path)
        self.input_tuples = [R.get_item(i) for i in range(len(R.input_tuples))]

class TestDataSet(Dataset):
    def __init__(self, tokenizer, data, type_path, data_args, max_examples=-1,
                 max_input_len=200, max_output_len=200):
        valid_type_paths = ["test", "train", "val"]
        assert type_path in valid_type_paths, f"Type path must be one of {valid_type_paths}"
        self.file_path = data
        self.max_examples = max_examples
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len  # max num of tokens in tokenize()
        self.max_output_len = max_output_len  # max num of tokens in tokenize()
        self.input_tuples = []  # list of dict
        self.data_args = data_args
        self._build()  # fill inputs, targets, max_lens
        self.shuffle = False
    def __len__(self):
        return len(self.input_tuples)
    def __getitem__(self, index):
        line = self.input_tuples[index]
        attributes = [attr for attr in line['attributes']]
        values = line['values']
        positions = [i for i in range(len(attributes))]
        if self.shuffle:
            random.shuffle(positions)
        record_relation = int(line['record_relation'])
        mask_id_list = []
        mask_list = [0] * len(line['attributes'])
        for index in range(len(attributes)):
            if values[index] == '':
                mask_id_list.append(index)
        source_input = ''
        positions = [i for i in range(len(attributes))]
        for i in positions:
            if i in mask_id_list:
                mask_list[i] = 1
                source_input += ' attribute ' + attributes[i] + ' value <mask>'
            else:
                source_input += ' attribute ' + attributes[i] + ' value ' + str(values[i])
        source_input += ' .'

        tokenized_inputs = self.tokenizer(
            [source_input], max_length=self.max_input_len, padding="max_length", return_tensors="pt", truncation=True
        )
        inputs = tokenized_inputs['input_ids'].squeeze()
        mask_inputs = tokenized_inputs['attention_mask'].squeeze()
        mask_list = np.array(mask_list)
        return {"inputs": inputs,
                "attributes": '#,#'.join(attributes),
                "mask_inputs": mask_inputs,
                "relation_info": record_relation,
                "mask_list": mask_list,
                }
    def _build(self):
        R = ReadCsv(self.file_path)
        self.input_tuples = [R.get_item(i) for i in range(len(R.input_tuples))]
        print("test_data_bulid...")

def get_dataloader(tokenizer, batch_size, type_path, num_examples, data, num_workers, data_args, train_args, shuffle_=False) -> Tuple[
    DataLoader, DataLoader]:
    data_set = TrainDataSet(tokenizer, type_path=type_path, data=data, data_args=data_args, max_examples=num_examples,
                            max_input_len=train_args.max_input_dim, max_output_len=train_args.max_out_dim)
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle_, num_workers=num_workers)
    return data_loader

def get_testdataloader(tokenizer, batch_size, type_path, num_examples, data, num_workers, data_args, shuffle_=False) -> Tuple[
    DataLoader, DataLoader]:
    data_set = TestDataSet(tokenizer, type_path=type_path, data=data, data_args=data_args, max_examples=num_examples,
                           max_input_len=200, max_output_len=200)
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle_, num_workers=0)
    return data_loader