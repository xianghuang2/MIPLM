import torch
from transformers import BertModel, BertTokenizer, MODEL_FOR_MASKED_LM_MAPPING, HfArgumentParser, BertConfig, \
    BertForMaskedLM, AdamW
from torch import nn
import numpy as np

# 返回：batch_size,k,embedding
def find_nearest_neighbors(vector, k, annoy, device):
    nearest_neighbors = []
    final_indices = []
    vector_list = torch.split(vector, 1, dim=0)
    # 对每个表征获取其最近邻
    for index, cur_vector in enumerate(vector_list):
        v = cur_vector.detach().cpu().numpy().ravel()
        nn_indices = annoy.get_nns_by_vector(v, k)  # 查询k个最近邻
        nn_vectors = [annoy.get_item_vector(i) for i in nn_indices]
        tensors = []
        for indices in nn_indices:
            final_indices.append(indices)
        for nn_vector in nn_vectors:
            tensors.append(torch.tensor(np.array(nn_vector).reshape(1,-1)).squeeze(0).to(device).to(torch.float32))
        nearest_neighbors.append(tensors)
    stacked_list = [torch.stack(sublist) for sublist in nearest_neighbors]
    final_tensor = torch.stack(stacked_list)
    return final_tensor, final_indices

def get_neighbors_tensor(encoder, neighbors_tokenizers,k, device):
    nearest_neighbors = []
    for tokenizer in neighbors_tokenizers:
        input_ids = tokenizer['input_ids'].to(device)
        attention_mask = tokenizer['attention_mask'].to(device)
        encoder_code = encoder(input_ids=input_ids, attention_mask=attention_mask)
        encoder_code = encoder_code.last_hidden_state
        cls_representation = encoder_code[:, 0, :]
        nearest_neighbors.append(cls_representation)
    split_lists = [nearest_neighbors[i:i + k] for i in range(0, len(nearest_neighbors), k)]
    tensors = [torch.stack(sublist) for sublist in split_lists]
    final_tensor = torch.stack(tensors).squeeze()
    return  final_tensor


# 模型：Bert-Encoder、Vector_embedding（可选）、重采样(可选)、Decoder
# input: Encoder_model、Annoy、hidden_dim、output_dim、fields(输出中哪些维度进行对应的映射)
# input is tokenized_inputs


def update_input(data, attributes, result_list, tokenizer):
    new_data_ids, new_data_block, new_data_mask = data['input_ids'], data['block_flag'], data['attention_mask']
    candidate_prompt = " candidate "
    candidate_prompt_id = tokenizer.encode(candidate_prompt, add_special_tokens=False, return_tensors="pt")
    mask_list = data['mask_list']
    for i in range(new_data_ids.shape[0]):
        nn_data = result_list[i]
        # 获取前k项数据对应属性的id
        mask_index = np.array(mask_list[i]).tolist()
        for m in range(len(mask_index)):
            candidate_value_ids = []
            if mask_index[m] == 1:
                attribute = attributes[m]
                for j in range(nn_data.shape[0]):
                    value = nn_data.iloc[j,m]
                    value_id = tokenizer.encode(value, add_special_tokens=False, return_tensors="pt")
                    candidate_value_ids.append(value_id)
                end_indices = int(torch.where(new_data_ids[i] == 102)[0])
                begin_end = end_indices
                # 从end_indices开始修改
                for k_indices, candidate_value_id in enumerate(candidate_value_ids):
                    prompt_length = candidate_prompt_id.shape[1]
                    k_value_length = candidate_value_id.shape[1]

                    new_data_ids[i, end_indices:end_indices+prompt_length] = candidate_prompt_id
                    new_data_block[i, end_indices:end_indices+prompt_length] = 0
                    end_indices += prompt_length

                    new_data_ids[i, end_indices:end_indices+k_value_length] = candidate_value_id
                    end_indices += k_value_length
                new_data_ids[i, end_indices] = 102
                end_indices += 1
                new_data_mask[i, begin_end:end_indices] = 1
    return new_data_ids, new_data_block, new_data_mask


class Bert_fill_model(nn.Module):
    def __init__(self, bert_encoder, only_encoder_BERT, IsUsePromptTune, tokenizer, encoder_dim, hidden_dim, output_dim, fields, setting, attr_num, annoy_k, device):
        super(Bert_fill_model, self).__init__()
        self.encoder = bert_encoder
        self.only_encoder_BERT = only_encoder_BERT
        self.setting = setting
        self.max_input_len = 200
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(encoder_dim, encoder_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(encoder_dim, encoder_dim))
        self.decoder = nn.ModuleList(
            [
                nn.Linear(encoder_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, output_dim)
            ]
        )
        self.prompt_length = attr_num * 2 + annoy_k
        self.prompt_embeddings = torch.nn.Embedding(self.prompt_length, encoder_dim)
        self.query_linear = nn.Linear(encoder_dim, encoder_dim)
        self.key_linear = nn.Linear(encoder_dim, encoder_dim)
        self.value_linear = nn.Linear(encoder_dim, encoder_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.fields = fields
        self.device = device
        self.IsUsePromptTune = IsUsePromptTune
        self.tokenizer = tokenizer
        self.annoy_k = annoy_k

    def forward(self, data, annoy, index_annoy_dict, ori_data):
        # 获取target数据的表征
        with torch.no_grad():
            target_representation = []
            for batch_target_input in data['target_input']:
                tokenized_inputs = self.tokenizer(
                    [batch_target_input], max_length=self.max_input_len, padding="max_length", return_tensors="pt",
                    truncation=True
                ).to(self.device)
                target_code = self.only_encoder_BERT(**tokenized_inputs)
                target_code = target_code.last_hidden_state
                target_code = target_code[:, 0, :].squeeze(0)  # 1 , 1, 768
                target_representation.append(target_code)
            target_representation = torch.stack(target_representation, dim=0)  # batch, 768

        if self.setting['use_annoy']:
            neighbors_tensor, neighbors_index = find_nearest_neighbors(target_representation, self.annoy_k, annoy, self.device)
            # 根据index获取数据
            data_index = []
            for index in neighbors_index:
                data_index.append(index_annoy_dict[index])
            # 最近邻的数据
            nn_data = ori_data.loc[data_index, :]

            # 处理数据为每个list代表每条数据的k
            result_list = []  # 存储结果的列表
            for i in range(len(data['mask_list'])):
                start_idx = i * self.annoy_k  # 计算每组的起始索引
                end_idx = start_idx + self.annoy_k  # 计算每组的结束索引
                selected_data = nn_data.iloc[start_idx:end_idx, :].copy()  # 取出当前组的数据
                result_list.append(selected_data)  # 添加到结果列表
            attributes = ori_data.columns.tolist()
            # 修改data['input_ids']和data['block_flag']
            data['input_ids'], data['block_flag'], data['attention_mask'] = update_input(data, attributes, result_list, self.tokenizer)
        else:
            neighbors_index = []

        with torch.no_grad():
            # (batch_size, max_length, embeds)
            input_embeddings = self.only_encoder_BERT.embeddings.word_embeddings(data['input_ids'].to(self.device))
        replace_embeds = self.prompt_embeddings(
            torch.LongTensor(list(range(self.prompt_length))).to(self.device))
        replace_index = 0
        for i in range(data['block_flag'].size(0)):
            # 遍历每一行的每个元素，直到倒数第三个
            for j in range(data['use_len'][i]):
                # 检查是否存在连续三个0
                if data['block_flag'][i, j] == 0:
                    # 如果存在，则在input_embeddings中对应位置替换为全0的tensor
                    input_embeddings[i, j, :] = replace_embeds[replace_index,:]
                    replace_index += 1
            replace_index = 0
        # 把embedding变为cls_representation
        # outputs = self.encoder(input_ids=data['input_ids'].to(self.device), attention_mask = data['attention_mask'].to(self.device), token_type_ids = data['token_type_ids'].to(self.device))
        # outputs = self.encoder(input_ids=data['input_ids'].to(self.device), attention_mask = data['attention_mask'].to(self.device))
        outputs = self.encoder(inputs_embeds=input_embeddings.to(self.device), attention_mask = data['attention_mask'].to(self.device), token_type_ids = data['token_type_ids'].to(self.device))
        cls_representation = outputs.last_hidden_state
        decoder_code = cls_representation[:,0,:]
        # (batch_size, length, dim)
        for decoder in self.decoder:
            decoder_code = decoder(decoder_code)
        # 根据out_dim 获取最后的输出
        output = []
        current_index = 0
        for i in range(len(self.fields)):
            if self.fields[i].data_type == "Categorical Data":
                dim = self.fields[i].dim()
                # data = self.softmax(decoder_code[:, current_index:current_index + dim])
                data = decoder_code[:, current_index:current_index + dim]
                output.append(data)
                current_index = current_index + dim
            else:
                # output.append(self.sigmod(decoder_code[:, current_index:current_index + 1]))
                output.append(decoder_code[:, current_index:current_index + 1])
                current_index = current_index + 1
        output = torch.cat(output, dim=1)
        return output, neighbors_index
# 最后通过向量的距离获取结果
class Bert_fill_model_distinct(nn.Module):
    def __init__(self, bert_encoder, BERT, IsUsePromptTune, tokenizer, encoder_dim, fields, setting, attr_num, device):
        super(Bert_fill_model_distinct, self).__init__()
        self.encoder = bert_encoder
        self.bert = BERT
        self.setting = setting
        self.max_input_len = 200
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(encoder_dim, encoder_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(encoder_dim, encoder_dim))
        self.prompt_length = attr_num * 2
        self.prompt_embeddings = torch.nn.Embedding(attr_num * 2, encoder_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.fields = fields
        self.device = device
        self.IsUsePromptTune = IsUsePromptTune
        self.tokenizer = tokenizer

    def forward(self, data, annoy, k, index_annoy_dict, ori_data, tokenizer_data):
        neighbors_index = []
        with torch.no_grad():
            # (batch_size, max_length, embeds)
            input_embeddings = self.bert.embeddings.word_embeddings(data['input_ids'].to(self.device))

        replace_embeds = self.prompt_embeddings(
            torch.LongTensor(list(range(self.prompt_length))).to(self.device))
        replace_index = 0

        for i in range(data['block_flag'].size(0)):
            cur_batch_place = []
            # 遍历每一行的每个元素，直到倒数第三个
            for j in range(data['use_len'][i]):
                # 检查是否存在连续三个0
                if data['block_flag'][i, j] == 0:
                    # 如果存在，则在input_embeddings中对应位置替换为全0的tensor
                    input_embeddings[i, j, :] = replace_embeds[replace_index,:]
                    replace_index += 1
            replace_index = 0

        # 把embedding变为cls_representation
        outputs = self.encoder(inputs_embeds=input_embeddings.to(self.device), attention_mask = data['attention_mask'].to(self.device), token_type_ids = data['token_type_ids'].to(self.device))
        cls_representation = outputs.last_hidden_state[:,0,:]
        output = cls_representation
        # 需要获取缺失的位置
        # batch_index, col_index = torch.where(data['impute_place']==0)
        # output = cls_representation[batch_index, col_index, :]
        # # 获取mask中为0的位置的cls表征
        # masked_cls_list = []
        # for i in range(cls_representation.size(0)):  # 遍历每个样本
        #     masked_cls = cls_representation[i][data['impute_place'][i] == 0]  # 选择mask为0的位置
        #     masked_cls_list.append(masked_cls)
        # output = torch.cat(masked_cls_list, dim = 0) # batch,768
        return output, neighbors_index


