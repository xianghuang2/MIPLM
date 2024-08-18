import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from models.model import find_nearest_neighbors


def get_res_loss(true_data, out_put, mask_id_list, fields, fields_dict):
    cur_dim = 0
    all_loss = 0
    for index, field in enumerate(fields):
        col_id = fields_dict[index]
        mask_id = mask_id_list[:,col_id]
        # 只对缺失mask的数据计算损失
        res_row = [x for x in range(mask_id.shape[0]) if mask_id[x]==1]
        if len(res_row) == 0: continue
        if field.data_type == 'Categorical Data':
            dim = field.dim()
            cur_true_data = true_data[res_row,cur_dim:cur_dim+dim]
            cur_pro_data = out_put[res_row, cur_dim:cur_dim+dim]
            targets = torch.argmax(cur_true_data, dim=1)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(cur_pro_data, targets) + 0.1
            cur_dim += dim
        else:
            cur_true_data = true_data[res_row,cur_dim:cur_dim+1]
            cur_pro_data = out_put[res_row, cur_dim:cur_dim+1]
            criterion = nn.MSELoss()
            loss = criterion(cur_pro_data, cur_true_data)
            cur_dim += 1
        all_loss += loss
    return all_loss



def forward_loss(bert_impute, batch, nan_feed_data, fields, fields_dict, annoy, index_annoy_dict, ori_data):

    out_put, nn_indices = bert_impute(batch, annoy, index_annoy_dict, ori_data)
    # 获取原数据索引
    data_index = []
    for index in nn_indices:
        data_index.append(index_annoy_dict[index])

    nn_data = ori_data.loc[data_index,:]
    # 比较和真实值是否相近
    tup_ids = batch['relation_info']
    miss_data = ori_data.loc[tup_ids,:]

    # 计算损失
    true_data = nan_feed_data[tup_ids, :]
    res_loss = get_res_loss(true_data, out_put, batch['mask_list'], fields, fields_dict)
    # if isinstance(res_loss, int):
    #     res_loss = torch.tensor(0)
    return res_loss, out_put

def forward_loss_distinct(bert_impute, batch, nan_feed_data, fields, fields_dict, annoy, k, index_annoy_dict,tokenizer_data, ori_data, nan_data, device, annoy_labels, enc):
    # 把输出的索引去annoy中找近邻
    out_put, nn_indices = bert_impute(batch, annoy, k, index_annoy_dict, ori_data, tokenizer_data)
    # 获取最近邻的数据与out_put
    neighbors_tensor, neighbors_index = find_nearest_neighbors(out_put, 3, annoy_labels, device)
    dim = fields[0].dim()
    # 获取最近邻的数据
    selected_elements = [neighbors_index[i] for i in range(len(neighbors_index)) if i%3 == 0]
    # 转换为62维的one-hot向量
    out = np.zeros((len(selected_elements), dim))  # 创建一个全0的矩阵
    for i, element in enumerate(selected_elements):
        if element < dim:  # 确保索引不会超出62维的范围
            out[i, element] = 1  # 在相应位置设置为1
    out = torch.tensor(out).to(device)
    # 定义其损失
    useful_index = batch['relation_info']
    #
    # 1. 获取数据真实标签的tensor
    tup_ids = list(useful_index.numpy())
    batch_data = nan_data.iloc[tup_ids, 2].to_frame()
    label_index = enc.transform(batch_data).flatten().astype(int).tolist()
    true_vectors = torch.tensor([annoy_labels.get_item_vector(i) for i in label_index]).to(device)

    col_id = fields_dict[0]
    mask_id = batch['mask_list'][:, col_id]
    # 只对缺失mask的数据计算损失
    res_row = [x for x in range(mask_id.shape[0]) if mask_id[x] == 1]
    if len(res_row) == 0:
        return 0, out
    out_put = out_put[res_row, :]
    true_vectors = true_vectors[res_row, :]
    cosine_similarity = F.cosine_similarity(out_put, true_vectors, dim=1)
    # 加其他损失
    other_loss = 0
    for i in range(3):
        selected_indices = neighbors_tensor[:, i , :].squeeze()
        other_loss += F.cosine_similarity(out_put, selected_indices, dim=1)
    all_loss = other_loss.mean() - cosine_similarity.mean() * 5 + 5
    return all_loss, out