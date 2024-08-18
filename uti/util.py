import json
import random
import sys
import torch
from annoy import AnnoyIndex
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertConfig, BertModel
from sklearn.preprocessing import OrdinalEncoder
from fields import NumericalField, CategoricalField
from models.model import Bert_fill_model
from torch import nn

def get_data_info(data_args, nan_data, ori_data, data_m, device):
    decoder_column = data_args.miss_column
    cat_column = data_args.cat_column
    num_column = data_args.num_column
    value_cat = [x for x in decoder_column if x in cat_column]
    value_num = [x for x in decoder_column if x in num_column]
    decoder_cols, continuous_cols, categorical_cols = get_cols_by_name(nan_data, decoder_column), get_cols_by_name(nan_data,
                                                                                                                   value_num), get_cols_by_name(
        nan_data, value_cat)
    cat_to_code_data, enc = categorical_to_code(ori_data.copy(), value_cat, enc=None)
    cat_to_code_data.columns = [x for x in range(cat_to_code_data.shape[1])]
    # 根据missing_columns排，fields为可处理数据后的处理过程，fields_dict表示fields中的某个处理过程对应于第几列
    fields, feed_data, fields_dict = Data_convert(cat_to_code_data, data_args.num_process_model, continuous_cols,
                                                  categorical_cols)
    feed_data_code = torch.tensor(feed_data.values)
    # 对nan_data进行编码,即nan_data的编码
    nan_feed_data, M = get_nan_code(fields, feed_data_code, data_m, fields_dict, device)
    return decoder_column,value_cat, continuous_cols, fields, fields_dict, enc, nan_feed_data, M

def printInfo(num_train, training_args):
    # 打印相关信息
    total_steps = ((num_train // training_args.per_device_train_batch_size) * training_args.num_train_epochs)
    total_train = num_train * training_args.num_train_epochs
    print(f'total_steps: {total_steps}\t Training Epochs: {training_args.num_train_epochs}\n'
          f'\t total_train: {total_train}')
    print(f'save model in {training_args.output_dir}')
    
def get_miss_data(data_args):
    np.random.seed(42)
    seed_everything()
    data_path = data_args.file
    missing_column = data_args.miss_column
    missing_rate = data_args.missing_rate
    # data = pd.read_csv(data_path)
    data = pd.read_excel(data_path)
    # data = data.iloc[:100,:]
    ori_data = data.copy()
    dirty_data = data.copy()
    for column in missing_column:
        if column in dirty_data.columns:
            num_missing = int(missing_rate * len(dirty_data))
            missing_indices = np.random.choice(dirty_data.index, num_missing, replace=False) # replace=false一个元素被选中后不会选择第二次
            dirty_data.loc[missing_indices, column] = np.nan
    data_m = dirty_data.notna().astype(int)
    # mode_value = dirty_data['manufacturer'].mode()[0]
    # 用众数填充第二列中的NaN值
    # dirty_data['manufacturer'].fillna(mode_value, inplace=True)
    return ori_data, dirty_data, data_m

# 类别转数值，传入pd，类别的列名，enc-（可选None）
def categorical_to_code(miss_data_x, value_cat, enc):
    if len(value_cat) == 0:
        return miss_data_x, None
    if enc is None:
        # 将类别进行编码为数字
        enc = OrdinalEncoder()
        enc.fit(miss_data_x[value_cat])
    attr_list_map = {}
    for col_name in miss_data_x.columns:
        attr_list_map[col_name] = miss_data_x[col_name].value_counts()
    miss_data_x[value_cat] = enc.transform(miss_data_x[value_cat])
    sim_data_x = pd.DataFrame(miss_data_x)
    return sim_data_x, enc

def resver_value(data, value_cat, enc):
    data[value_cat] = enc.inverse_transform(data[value_cat])
    return data

def labelCode(data, value_cat, enc):
    data[value_cat] = enc.transform(data[value_cat])
    return data

def concatValue(current_data, miss_data, m):
    current_data = current_data.values
    miss_data = miss_data.values
    new_data = miss_data * m + current_data * (1 - m)
    return pd.DataFrame(new_data)



def reconvert_data(out_code, fields, value_cat, decoder_column, enc):
    current_data = []
    current_ind = 0
    for i in range(len(fields)):
        dim = fields[i].dim()
        data_transept = out_code[:, current_ind:current_ind + dim].cpu().detach().numpy()
        if dim == 1:
            current_data.append(pd.DataFrame(fields[i].reverse(data_transept)))
        else:
            rever_data = pd.DataFrame(fields[i].reverse(data_transept))
            current_data.append(rever_data)
        current_ind = current_ind + dim
    current_data = pd.concat(current_data, axis=1)
    current_data.columns = decoder_column
    current_data[value_cat] = enc.inverse_transform(current_data[value_cat])
    return current_data

def get_cols_by_name(data, name_cols):
    return [i for i in range(data.shape[1]) if data.columns[i] in name_cols]


def get_nan_code(fields, feed_data, data_m, fields_dict, device):
    data_m = data_m.copy().values
    feed_M = []
    M = []
    index = 0
    for index, field in enumerate(fields):
        i = int(fields_dict[index])
        m = data_m[:, i].reshape(-1,1)
        if fields[index].data_type == 'Categorical Data':
            dim = fields[index].dim()
            m = np.repeat(m, dim, axis=1)
            m = torch.tensor(m)
            cur_feed_data = feed_data[:, index:index+dim]
            feed_m = m * cur_feed_data
            feed_M.append(feed_m)
            M.append(m)
            index = index + dim
        else:
            m = torch.tensor(m)
            cur_feed_data = feed_data[:, index:index+1]
            feed_m = m * cur_feed_data
            feed_M.append(feed_m)
            M.append(m)
            index = index + 1
    feed_M = torch.cat(feed_M, dim=1)
    M = torch.cat(M, dim=1)
    return feed_M.to(device), M.to(device)


def Data_convert(data, model_name, continuous_cols, categorical_cols):
    fields = []
    feed_data = []
    fields_dict = {}
    index = 0
    for i, col in enumerate(list(data)):
        # data[i]第i列所有数据
        if i in continuous_cols:
            col2 = NumericalField(model=model_name)
            # 把data传进去
            col2.get_data(data[i])
            fields.append(col2)
            # 获取mean等
            col2.learn()
            # 对第i列数据使用数据标准化方式进行编码,数据-均值/方差
            feed_data.append(col2.convert(np.asarray(data[i])))
            fields_dict[index] = i
            index += 1
        elif i in categorical_cols:
            col1 = CategoricalField("one-hot", noise=None)
            fields.append(col1)
            col1.get_data(data[i])
            col1.learn()
            # 将类别数据使用one-hot向量进行编码
            features = col1.convert(np.asarray(data[i]))
            cols = features.shape[1]
            rows = features.shape[0]
            for j in range(cols):
                feed_data.append(features.T[j])
            fields_dict[index] = i
            index += 1
    feed_data = pd.DataFrame(feed_data).T
    return fields, feed_data, fields_dict


# 传入ori_data、missing_column,返回missing_column中被处理后的数据

# 根据vector去Annoy中寻找k个最近的vector

def build_annoy(embeddings, num_tree):
    f = embeddings[0].shape[1]
    meta = 'angular'
    annoy = AnnoyIndex(f, meta)# euclidean,angular
    for i, v in enumerate(embeddings):
        vector = v.numpy().ravel()
        annoy.add_item(i,vector)
    annoy.build(num_tree)
    annoy_path = 'annoy/buy_observed.ann'
    annoy.save(annoy_path)
    return annoy, annoy_path, f, meta

# 把observed的元组转化为texts
def data_to_text(data):
    texts = []
    attributes = data.columns.tolist()
    for index, row in data.iterrows():
        text_input = ''
        values = row.values.tolist()
        positions = [i for i in range(len(attributes))]
        for i in positions:
            text_input += ' attribute ' + attributes[i] + ' value ' + str(values[i])
        texts.append(text_input)
    return texts

# 传入数据获取所有干净数据的embedding
def get_observe_embedding(bert_encoder,tokenizer,max_input_len, nan_data,data_m):
    embeddings = []
    tokenized_data = []
    observe_index = data_m.index[data_m.all(axis=1)].tolist()
    # annay中的index对应于数据表中的数据索引
    index_annay_dict = {}
    for key in range(len(observe_index)):
        index_annay_dict[key] = observe_index[key]

    observe_data = nan_data.iloc[observe_index,:]
    observe_texts = data_to_text(observe_data)

    for text in observe_texts:
        tokenized_inputs = tokenizer(
            [text], max_length=max_input_len, padding="max_length", return_tensors="pt", truncation=True
        )
        with torch.no_grad():
            encoded_vector = bert_encoder(**tokenized_inputs)

        encoded_vector = encoded_vector.last_hidden_state
        cls_representation = encoded_vector[:, 0, :] # (1, 768)的embedding

        embeddings.append(cls_representation)
        tokenized_data.append(tokenized_inputs)
    return embeddings, index_annay_dict, tokenized_data

def get_embedding(enc, bert_encoder, tokenizers):
    labels = enc.categories_[0] # 所有的标签数据
    embeddings = []
    for label in labels:
        tokenized_inputs = tokenizers(
            [label], return_tensors="pt"
        )
        with torch.no_grad():
            encoder_label = bert_encoder(**tokenized_inputs)
        encoder_label = encoder_label.last_hidden_state[:, 0, :]
        embeddings.append(encoder_label)
    return embeddings


# input: (batch_size, data_dim)
#---------------------------------------------------#
#   设置种子
#---------------------------------------------------#
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False





def errorLoss(filled_pd, ground_truth_pd, test_m_pd, value_cat, continuous_cols, enc):
    M = test_m_pd.values
    copy_ori_data = ground_truth_pd.copy()
    copy_imputed_data = filled_pd.copy()
    no, dim = copy_imputed_data.shape
    H = np.ones((no, dim))
    # H在数值类型数据全设为0
    for i in continuous_cols:
        H[:, i] = 0
    # data_h类别类型上数据保持为缺失状态，数值上数据全为1
    data_h = 1 - (1 - M) * H
    # data_m数值上的保持为缺失状态，类别上数据全为1
    data_m = 1 - (1 - M) * (1 - H)

    if len(value_cat) != 0:
        copy_imputed_data[value_cat] = enc.transform(copy_imputed_data[value_cat])
        copy_ori_data[value_cat] = enc.transform(copy_ori_data[value_cat])

    imputed_data = copy_imputed_data.values
    ori_data = copy_ori_data.values

    imputed_data = imputed_data.astype(float)
    data_m = data_m.astype(float)
    ori_data = ori_data.astype(float)

    # cate_imputed_data表示类别上的数据
    cate_imputed_data = imputed_data * (1 - data_h)
    cate_ori_data = ori_data * (1 - data_h)
    # 不等于0表示不计算不缺失的位置
    Z = (cate_imputed_data == cate_ori_data) & (cate_imputed_data != 0)
    C = (cate_imputed_data != cate_ori_data).astype('int')
    c = np.where(C == 1)[0]
    Z = Z.astype('int')
    # print("错误的索引id：")
    # print(c)
    # print("\n")
    # print("正确的数据是：")
    # print(ground_truth_pd.iloc[c])
    # print("\n")
    # print("填充的数据是：")
    # print(filled_pd.iloc[c])
    cat_miss_num = np.sum(1-data_h)
    Acc = np.sum(Z) * 100.00 / cat_miss_num

    # 对数值数据进行归一化为[-1,1]
    ori_data, norm_parameters = normalization(ori_data)
    imputed_data, _ = normalization(imputed_data, norm_parameters)
    # data_m = M.astype(float)
    cur_num = (1 - data_m) * ori_data - (1 - data_m) * imputed_data
    CORR = cur_num ** 2
    if np.sum(1-data_m)==0:
        RMSE = 0
    else:
        RMSE = np.sqrt(np.sum(CORR) / np.sum(1 - data_m) + 1e-5)

    print("RMSE为：{}  ACC为：{}".format(RMSE, Acc))
    # # 遍历每一列求其RMSE或者MAE
    # ARMSE = 0
    # AMAE = 0
    # miss_dim = 0
    # for i in range(dim):
    #     ori_get_data = ori_data[:, i]
    #     imputed_get_data = imputed_data[:, i]
    #     if i in continuous_cols:
    #         data_i_m = data_m[:, i]
    #         if np.sum((1 - data_i_m)) == 0:
    #             continue
    #         AR = np.sqrt(np.sum((1 - data_i_m) * ((ori_get_data - imputed_get_data) ** 2)) / np.sum(1 - data_i_m))
    #         ARMSE = ARMSE + AR
    #         MAR = np.sum((1 - data_i_m) * np.abs(ori_get_data - imputed_get_data)) / np.sum(1 - data_i_m)
    #         AMAE = AMAE + MAR
    #         miss_dim = miss_dim + 1
    #     else:
    #         data_i_h = data_h[:, i]
    #         if np.sum((1 - data_i_h)) == 0:
    #             continue
    #         equal = (ori_get_data != imputed_get_data).astype('int')
    #         AR = (np.sum((1 - data_i_h) * equal) / np.sum(1 - data_i_h))
    #         MAR = np.sum((1 - data_i_h) * equal) / np.sum(1 - data_i_h)
    #         ARMSE = ARMSE + AR
    #         AMAE = AMAE + MAR
    #         miss_dim = miss_dim + 1
    #     # if AR > 1:
    #     #     print(1)
    # ARMSE = ARMSE / dim
    # AMAE = AMAE / dim
    return RMSE, Acc

def normalization(data, parameters=None):
    # Parameters
    _, dim = data.shape
    norm_data = data.copy()
    if parameters is None:
        # MixMax normalization
        min_val = np.zeros(dim)
        max_val = np.zeros(dim)
        # For each dimension
        for i in range(dim):
            # if i == 7 :
            #     print(1)
            min_val[i] = np.nanmin(norm_data[:, i])
            norm_data[:, i] = norm_data[:, i] - np.nanmin(norm_data[:, i])
            max_val[i] = np.nanmax(norm_data[:, i])
            norm_data[:, i] = norm_data[:, i] / (np.nanmax(norm_data[:, i])+ 1e-6)
            # Return norm_parameters for renormalization
        norm_parameters = {'min_val': min_val,
                           'max_val': max_val}

    else:
        min_val = parameters['min_val']
        max_val = parameters['max_val']

        # For each dimension
        for i in range(dim):
            # if i == 7:
            #     print(1)
            norm_data[:, i] = norm_data[:, i] - min_val[i]
            norm_data[:, i] = norm_data[:, i] / (max_val[i] + 1e-6)
        norm_parameters = parameters
    return norm_data, norm_parameters


def get_data_index(data_name, params):
    for index,dict in enumerate(params):
        if dict['dataset_name'] == data_name:
            return index
    return -1



