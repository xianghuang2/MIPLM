import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForMaskedLM
import torch.nn as nn
import torch.optim as optim

import torch
from transformers import BertTokenizer, BertModel
import random

# Return a list of tokens
class PromptTuning(nn.Module):
    def __init__(self, bert_model, tokenizer, device, token_num=3):
        super(PromptTuning, self).__init__()
        self.bert = bert_model
        self.token_num = token_num
        prompt_indices_begin = "the attribute is"
        prompt_begin = tokenizer([prompt_indices_begin], return_tensors="pt")
        a = prompt_begin['input_ids']
        with torch.no_grad():
            embedding = bert_model(input_ids=prompt_begin['input_ids'])
            encoded_vector_begin = embedding.last_hidden_state
        token_embeddings_begin = encoded_vector_begin[0, 1 : 1 + token_num, :]


        prompt_indices_end = "the value is"
        prompt_end = tokenizer([prompt_indices_end], return_tensors="pt")
        with torch.no_grad():
            embedding = bert_model(input_ids=prompt_end['input_ids'])
            encoded_vector_end = embedding.last_hidden_state
        token_embeddings_end = encoded_vector_end[0, 1 : 1 + token_num, :]

        print(f'Initial context: "{prompt_indices_begin}+{prompt_indices_end}"')
        print(f"Number of context words (tokens): {token_num}")

        self.begin_embedding = nn.Parameter(token_embeddings_begin)
        self.end_embedding = nn.Parameter(token_embeddings_end)

        self.device = device
        self.vocab_size = len(tokenizer.vocab)
        self.tokenizer = tokenizer
        self.prompt_set = nn.ModuleList(
            [
                nn.Linear(self.vocab_size // 5, self.vocab_size),
                nn.ReLU(),
                nn.Dropout(0.2),
            ]
        )

    def forward(self):
        # begin_code = self.prompt_indices_begin
        # prompt_embeddings_begin = nn.Parameter(torch.argmax(begin_code, dim=1),requires_grad=False)
        # begin_prompt = self.tokenizer.decode(prompt_embeddings_begin)
        #
        # end_code = self.prompt_indices_end
        # for prompt_s in self.prompt_set:
        #     end_code = prompt_s(end_code)
        # prompt_embeddings_end = nn.Parameter(torch.argmax(end_code, dim=1),requires_grad=False)
        # end_prompt = self.tokenizer.decode(prompt_embeddings_end)
        return self.begin_embedding, self.end_embedding


# 希望返回可学习得prompt即可
# class PromptLearner(nn.Module):
#     def __init__(self, bert_model, tokenizer, device, token_num=3):
#         super().__init__()
#
#         ctx_init = cfg.TRAINER.COCOOP.CTX_INIT
#         dtype = clip_model.dtype
#         ctx_dim = clip_model.ln_final.weight.shape[0]
#
#         if ctx_init:
#             # use given words to initialize context vectors
#             ctx_init = ctx_init.replace("_", " ")
#             n_ctx = len(ctx_init.split(" "))
#             prompt = clip.tokenize(ctx_init)
#             with torch.no_grad():
#                 embedding = clip_model.token_embedding(prompt).type(dtype)
#             ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
#             prompt_prefix = ctx_init
#         else:
#             # random initialization
#             ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
#             nn.init.normal_(ctx_vectors, std=0.02)
#             prompt_prefix = " ".join(["X"] * n_ctx)
#
#         print(f'Initial context: "{prompt_prefix}"')
#         print(f"Number of context words (tokens): {n_ctx}")
#
#         self.ctx = nn.Parameter(ctx_vectors)
#
#         self.meta_net = nn.Sequential(OrderedDict([
#             ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
#             ("relu", nn.ReLU(inplace=True)),
#             ("linear2", nn.Linear(vis_dim // 16, ctx_dim))
#         ]))
#
#         if cfg.TRAINER.COCOOP.PREC == "fp16":
#             self.meta_net.half()
#
#         classnames = [name.replace("_", " ") for name in classnames]
#         name_lens = [len(_tokenizer.encode(name)) for name in classnames]
#         prompts = [prompt_prefix + " " + name + "." for name in classnames]
#
#         tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
#         with torch.no_grad():
#             embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
#
#         # These token vectors will be saved when in save_model(),
#         # but they should be ignored in load_model() as we want to use
#         # those computed using the current class names
#         self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
#         self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS
#
#         self.n_cls = n_cls
#         self.n_ctx = n_ctx
#         self.tokenized_prompts = tokenized_prompts  # torch.Tensor
#         self.name_lens = name_lens
#
#     def construct_prompts(self, ctx, prefix, suffix, label=None):
#         # dim0 is either batch_size (during training) or n_cls (during testing)
#         # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
#         # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
#         # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)
#
#         if label is not None:
#             prefix = prefix[label]
#             suffix = suffix[label]
#
#         prompts = torch.cat(
#             [
#                 prefix,  # (dim0, 1, dim)
#                 ctx,  # (dim0, n_ctx, dim)
#                 suffix,  # (dim0, *, dim)
#             ],
#             dim=1,
#         )
#
#         return prompts
#
#     def forward(self, im_features):
#         prefix = self.token_prefix
#         suffix = self.token_suffix
#         ctx = self.ctx  # (n_ctx, ctx_dim)
#         bias = self.meta_net(im_features)  # (batch, ctx_dim)
#         bias = bias.unsqueeze(1)  # (batch, 1, ctx_dim)
#         ctx = ctx.unsqueeze(0)  # (1, n_ctx, ctx_dim)
#         ctx_shifted = ctx + bias  # (batch, n_ctx, ctx_dim)
#
#         # Use instance-conditioned context tokens for all classes
#         prompts = []
#         for ctx_shifted_i in ctx_shifted:
#             ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
#             pts_i = self.construct_prompts(ctx_i, prefix, suffix)  # (n_cls, n_tkn, ctx_dim)
#             prompts.append(pts_i)
#         prompts = torch.stack(prompts)
#
#         return prompts

# tokenizer = BertTokenizer.from_pretrained("../bertt")
# bert_model = BertModel.from_pretrained("../bertt")
# device = "cuda:0"
# t_model = PromptTuning(bert_model, tokenizer, device, 3)
# test_prompt = t_model()
# print(test_prompt)

