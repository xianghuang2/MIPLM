import random

import torch
from transformers import BertTokenizer, BertConfig, BertModel

from models.model import Bert_fill_model, Bert_fill_model_distinct
from uti.util import seed_everything


def get_model(model_path,IsUsePromptTune, fields, setting, attr_num, annoy_k, device):
    seed_everything()
    tokenizer = BertTokenizer.from_pretrained(model_path)
    config = BertConfig.from_pretrained(model_path)
    bert_model = BertModel.from_pretrained(model_path)
    only_encoder_bert = BertModel.from_pretrained(model_path)
    hidden_dim = random.choice([32,64,128,256,512])
    out_dim = 0
    for field in fields:
        if field.data_type == 'Categorical Data':
            out_dim += field.dim()
        else:
            out_dim += 1
    impute_model = Bert_fill_model(bert_model, only_encoder_bert, IsUsePromptTune,tokenizer, config.hidden_size, hidden_dim, out_dim, fields, setting, attr_num, annoy_k, device)
    # impute_model = Bert_fill_model_distinct(bert_model, only_encoder_bert, IsUsePromptTune, tokenizer, config.hidden_size, fields, setting, attr_num, device)
    return tokenizer, config, bert_model, impute_model