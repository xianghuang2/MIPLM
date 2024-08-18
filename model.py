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
        for nn_indice in nn_indices:
            final_indices.append(nn_indice)
        for nn_vector in nn_vectors:
            tensors.append(torch.tensor(np.array(nn_vector).reshape(1,-1)).squeeze(0).to(device).to(torch.float32))
        nearest_neighbors.append(tensors)
    stacked_list = [torch.stack(sublist) for sublist in nearest_neighbors]
    final_tensor = torch.stack(stacked_list)
    return final_tensor, final_indices



# 模型：Bert-Encoder、Vector_embedding（可选）、重采样(可选)、Decoder
# input: Encoder_model、Annoy、hidden_dim、output_dim、fields(输出中哪些维度进行对应的映射)
# input is tokenized_inputs
class Bert_fill_model(nn.Module):
    def __init__(self, bert_encoder, encoder_dim, hidden_dim, output_dim, fields, setting, device):
        super(Bert_fill_model, self).__init__()
        self.encoder = bert_encoder
        self.setting = setting
        self.decoder = nn.ModuleList(
            [
                nn.Linear(encoder_dim, encoder_dim),
                nn.ReLU(),
                nn.Dropout(0.2),

                nn.Linear(encoder_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),

                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),

                nn.Linear(hidden_dim, output_dim)
            ]
        )
        self.query_linear = nn.Linear(encoder_dim, encoder_dim)
        self.key_linear = nn.Linear(encoder_dim, encoder_dim)
        self.value_linear = nn.Linear(encoder_dim, encoder_dim)
        self.relu = nn.ReLU()
        self.sigmod = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.fields = fields
        self.device = device



    def forward(self, input_ids,input_mask, annoy, k):
        if self.setting['finetune_all']:
            encoder_code = self.encoder(input_ids=input_ids, attention_mask=input_mask)# **inputs意思是inputs是个字典，将其全部转为参数进行传递
        else:
            with torch.no_grad():
                encoder_code = self.encoder(input_ids=input_ids, attention_mask=input_mask)
        encoder_code = encoder_code.last_hidden_state
        # 获取CLS的输出
        cls_representation = encoder_code[:, 0, :]
        if self.setting['use_annoy']:
            neighbors_tensor, neighbors_index = find_nearest_neighbors(cls_representation, k, annoy, self.device)# 返回batch的list，其中每项包含对于k个向量

            cls_representation = cls_representation.unsqueeze(1)
            all_tensor = torch.cat((cls_representation, neighbors_tensor), dim=1)
            Q = self.relu(self.query_linear(cls_representation))  # b,1,768
            K = self.relu(self.key_linear(all_tensor))  # b,n,768
            V = self.relu(self.value_linear(all_tensor))  # b,n,768
            attention_score = torch.softmax(torch.bmm(Q, K.transpose(1, 2)) / torch.sqrt(torch.tensor(Q.shape[-1])), dim=-1)  # b,1,n
            attention_representation = self.relu(torch.bmm(attention_score, V))
            cls_representation = attention_representation.to(torch.float32)
        else:
            neighbors_index = []
        # 考虑是否使用VAE?
        decoder_code = cls_representation.squeeze(1)
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



