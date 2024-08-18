import torch
from tqdm import tqdm

from models.loss_function import forward_loss, forward_loss_distinct
from uti.util import reconvert_data, categorical_to_code, errorLoss


def train_impute_model(model, training_args, train_loader, valid_loader,
                       continuous_cols, value_cat,  decoder_column,
                       nan_feed_data, ori_data, M,
                       valid_nan_data, valid_ori_data, valid_data_m,
                       train_nan_data, train_ori_data, train_data_m,
                       fields, fields_dict, enc,
                       annoy, index_annoy_dict, device, use_P_tunning):
    bert_impute = model
    bert_impute.to(device)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_impute.encoder.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in bert_impute.encoder.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0},
        {'params': bert_impute.decoder.parameters()}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=training_args.learning_rate,
                                  eps=training_args.adam_epsilon)
    embedding_parameters = [
        {'params': [p for p in bert_impute.mlp.parameters()]},
        {'params': [p for p in bert_impute.prompt_embeddings.parameters()]}
    ]
    embedding_optimizer = torch.optim.AdamW(embedding_parameters, lr=training_args.learning_rate * 10, eps=training_args.adam_epsilon)
    epoch = 0  # number of times we have passed through entire set of training examples
    step = 0  # number of total examples we have done (will be epoch * len(data_set) at end of each epoch)

    best_impute_model, best_ACC, best_epoch = bert_impute, 0, 0

    bert_impute.train()
    with tqdm(total=training_args.num_train_epochs) as progress_bar:
        for epoch in range(1, training_args.num_train_epochs + 1):
            training_loss = 0
            train_total = 0
            with torch.enable_grad():
                for batch_num, batch in enumerate(train_loader):
                    batch_size = len(batch["source_input"])
                    train_total += batch_size
                    optimizer.zero_grad()
                    if use_P_tunning:
                        embedding_optimizer.zero_grad()
                    loss, out = forward_loss(bert_impute, batch, nan_feed_data, fields, fields_dict, annoy,
                                             index_annoy_dict, ori_data)
                    if isinstance(loss, int):
                        loss_val = loss
                    else:
                        loss_val = loss.item()
                    training_loss += loss_val
                    if not isinstance(loss, int):
                        loss.backward()
                        optimizer.step()
                    if use_P_tunning:
                        embedding_optimizer.step()
                    step += batch_size
            progress_bar.update(1)
            progress_bar.set_postfix(epoch=epoch, loss=training_loss / train_total)

            if epoch % 10 == 0:
                bert_impute.eval()
                filled_train_pd = train_nan_data.copy()[decoder_column]
                with torch.no_grad():
                    valid_total = 0
                    for batch_num, batch in enumerate(train_loader):
                        batch_size = len(batch["source_input"])
                        valid_total += batch_size
                        loss, out = forward_loss(bert_impute, batch, nan_feed_data, fields, fields_dict, annoy,
                                                 index_annoy_dict, ori_data)
                        tup_ids = batch['relation_info']
                        # 将code进行拼接
                        nan_code = nan_feed_data[tup_ids, :]
                        M_code = M[tup_ids, :]
                        out_code = M_code * nan_code + (1 - M_code) * out
                        reverse_data = reconvert_data(out_code, fields, value_cat, decoder_column, enc)
                        filled_train_pd.loc[tup_ids, :] = reverse_data.values

                    # 根据filled_pd和ground_truth计算损失
                    ground_truth_train_pd = train_ori_data.copy()[decoder_column]
                    cat_to_code_data, enc = categorical_to_code(ground_truth_train_pd.copy(), value_cat, enc)
                    cat_to_code_data.columns = [x for x in range(cat_to_code_data.shape[1])]
                    train_m_pd = train_data_m.copy()[decoder_column]
                    RMSE, Acc = errorLoss(filled_train_pd, ground_truth_train_pd, train_m_pd, value_cat, continuous_cols, enc)
                    if Acc >= best_ACC:
                        best_impute_model = bert_impute
                        best_ACC = Acc
                        best_epoch = epoch
                bert_impute.train()

    return best_impute_model, bert_impute


def train_impute_model_distinct(model, training_args, train_loader, valid_loader,
                       continuous_cols, value_cat,  decoder_column,
                       nan_feed_data, ori_data, nan_data, M,
                       valid_nan_data, valid_ori_data, valid_data_m,
                       fields, fields_dict, enc,
                       annoy, k, index_annoy_dict, tokenizer_data, device, annoy_label):
    bert_impute = model
    bert_impute.to(device)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_impute.encoder.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in bert_impute.encoder.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0},
        # {'params': bert_impute.decoder.parameters()}
        # {'params': [p for p in bert_impute.mlp.parameters()]},
        # {'params': [p for p in bert_impute.prompt_embeddings.parameters()]}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=training_args.learning_rate,
                                  eps=training_args.adam_epsilon)
    embedding_parameters = [
        {'params': [p for p in bert_impute.mlp.parameters()]},
        {'params': [p for p in bert_impute.prompt_embeddings.parameters()]}
    ]
    embedding_optimizer = torch.optim.AdamW(embedding_parameters, lr=training_args.learning_rate * 10, eps=training_args.adam_epsilon)
    epoch = 0  # number of times we have passed through entire set of training examples
    step = 0  # number of total examples we have done (will be epoch * len(data_set) at end of each epoch)

    best_impute_model, best_ACC, best_epoch = bert_impute, 0, 0

    bert_impute.train()
    with tqdm(total=training_args.num_train_epochs) as progress_bar:
        for epoch in range(1, training_args.num_train_epochs + 1):
            training_loss = 0
            train_total = 0
            with torch.enable_grad():
                for batch_num, batch in enumerate(train_loader):
                    batch_size = len(batch["source_input"])
                    train_total += batch_size
                    optimizer.zero_grad()
                    embedding_optimizer.zero_grad()
                    loss, out = forward_loss_distinct(bert_impute, batch, nan_feed_data, fields, fields_dict, annoy, k,
                                             index_annoy_dict, tokenizer_data, ori_data, nan_data, device, annoy_label, enc)
                    loss_val = loss.item()
                    training_loss += loss_val
                    loss.backward()
                    optimizer.step()
                    embedding_optimizer.step()
                    step += batch_size
            progress_bar.update(1)
            progress_bar.set_postfix(epoch=epoch, loss=training_loss / train_total)
            if epoch % 10 == 0:
                bert_impute.eval()
                filled_valid_pd = valid_nan_data.copy()[decoder_column]
                with torch.no_grad():
                    valid_total = 0
                    for batch_num, batch in enumerate(valid_loader):
                        batch_size = len(batch["source_input"])
                        valid_total += batch_size
                        loss, out = forward_loss_distinct(bert_impute, batch, nan_feed_data, fields, fields_dict, annoy, k,
                                                 index_annoy_dict, tokenizer_data, ori_data, nan_data, device, annoy_label, enc)
                        tup_ids = batch['relation_info']
                        # 将code进行拼接
                        nan_code = nan_feed_data[tup_ids, :]
                        M_code = M[tup_ids, :]
                        out_code = M_code * nan_code + (1 - M_code) * out
                        reverse_data = reconvert_data(out_code, fields, value_cat, decoder_column, enc)
                        filled_valid_pd.loc[tup_ids, :] = reverse_data.values

                    # 根据filled_pd和ground_truth计算损失
                    ground_truth_valid_pd = valid_ori_data.copy()[decoder_column]
                    cat_to_code_data, enc = categorical_to_code(ground_truth_valid_pd.copy(), value_cat, enc)
                    cat_to_code_data.columns = [x for x in range(cat_to_code_data.shape[1])]
                    valid_m_pd = valid_data_m.copy()[decoder_column]
                    RMSE, Acc = errorLoss(filled_valid_pd, ground_truth_valid_pd, valid_m_pd, value_cat, continuous_cols, enc)
                    if Acc >= best_ACC:
                        best_impute_model = bert_impute
                        best_ACC = Acc
                        best_epoch = epoch

                bert_impute.train()

    return best_impute_model, bert_impute


def only_train_prompt_model(model, prompt_model, training_args, train_loader, valid_loader,
                       continuous_cols, value_cat,  decoder_column,
                       nan_feed_data, ori_data, M,
                       valid_nan_data, valid_ori_data, valid_data_m,
                       fields, fields_dict, enc,
                       annoy, k, index_annoy_dict, tokenizer_data, device):
    bert_impute = model
    bert_impute.to(device)
    if prompt_model is not None:
        prompt_model.to(device)
        optimizer_prompt = torch.optim.SGD(prompt_model.parameters(), lr=training_args.learning_rate)
        best_prompt_model = prompt_model
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_impute.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in bert_impute.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=training_args.learning_rate,
                                  eps=training_args.adam_epsilon)
    epoch = 0  # number of times we have passed through entire set of training examples
    step = 0  # number of total examples we have done (will be epoch * len(data_set) at end of each epoch)

    best_impute_model, best_ACC, best_epoch = bert_impute, 0, 0

    bert_impute.eval()
    with tqdm(total=training_args.num_train_epochs) as progress_bar:
        for epoch in range(1, training_args.num_train_epochs + 1):
            training_loss = 0
            train_total = 0
            with torch.enable_grad():
                for batch_num, batch in enumerate(train_loader):
                    batch_size = len(batch["inputs"])
                    train_total += batch_size
                    # optimizer.zero_grad()
                    if prompt_model is not None:
                        optimizer_prompt.zero_grad()
                    loss, out = forward_loss(bert_impute, batch, nan_feed_data, fields, fields_dict, annoy, k,
                                             index_annoy_dict, tokenizer_data, ori_data, device)
                    loss_val = loss.item()
                    training_loss += loss_val
                    loss.backward()
                    # optimizer.step()
                    if prompt_model is not None:
                        optimizer_prompt.step()
                    step += batch_size
            progress_bar.update(1)
            progress_bar.set_postfix(epoch=epoch, loss=training_loss / train_total)
            if epoch % 10 == 0:
                bert_impute.eval()
                if prompt_model is not None:
                    prompt_model.eval()
                filled_valid_pd = valid_nan_data.copy()[decoder_column]
                with torch.no_grad():
                    valid_total = 0
                    for batch_num, batch in enumerate(valid_loader):
                        batch_size = len(batch["inputs"])
                        if prompt_model is not None:
                            begin = batch["begin"][0]
                            end = batch["end"][0]
                            if batch_num == 0:
                                print("\n" + begin + "\n" + end)
                        valid_total += batch_size
                        out, nn_indices = bert_impute(batch['inputs'].to(device), batch['mask_inputs'].to(device),
                                                      batch['target_inputs'].to(device),
                                                      batch['target_mask_inputs'].to(device),
                                                      annoy, k, tokenizer_data)
                        tup_ids = batch['relation_info']
                        # 将code进行拼接
                        nan_code = nan_feed_data[tup_ids, :]
                        M_code = M[tup_ids, :]
                        out_code = M_code * nan_code + (1 - M_code) * out
                        reverse_data = reconvert_data(out_code, fields, value_cat, decoder_column, enc)
                        filled_valid_pd.loc[tup_ids, :] = reverse_data.values

                    # 根据filled_pd和ground_truth计算损失
                    ground_truth_valid_pd = valid_ori_data.copy()[decoder_column]
                    cat_to_code_data, enc = categorical_to_code(ground_truth_valid_pd.copy(), value_cat, enc)
                    cat_to_code_data.columns = [x for x in range(cat_to_code_data.shape[1])]
                    valid_m_pd = valid_data_m.copy()[decoder_column]
                    RMSE, Acc = errorLoss(filled_valid_pd, ground_truth_valid_pd, valid_m_pd, value_cat,
                                          continuous_cols,
                                          enc)
                    if Acc > best_ACC:
                        best_impute_model = bert_impute
                        best_ACC = Acc
                        best_epoch = epoch
                        best_prompt_model = prompt_model
                bert_impute.train()
                if prompt_model is not None:
                    prompt_model.train()
    return best_impute_model, best_prompt_model


def only_train_impute_model(model, prompt_model, training_args, train_loader, valid_loader,
                       continuous_cols, value_cat,  decoder_column,
                       nan_feed_data, ori_data, M,
                       valid_nan_data, valid_ori_data, valid_data_m,
                       fields, fields_dict, enc,
                       annoy, k, index_annoy_dict, tokenizer_data, device):
    bert_impute = model
    bert_impute.to(device)
    if prompt_model is not None:
        prompt_model.to(device)
        optimizer_prompt = torch.optim.SGD(prompt_model.parameters(), lr=training_args.learning_rate)
        best_prompt_model = prompt_model
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_impute.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in bert_impute.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=training_args.learning_rate,
                                  eps=training_args.adam_epsilon)
    epoch = 0  # number of times we have passed through entire set of training examples
    step = 0  # number of total examples we have done (will be epoch * len(data_set) at end of each epoch)

    best_impute_model, best_ACC, best_epoch = bert_impute, 0, 0

    bert_impute.train()
    with tqdm(total=training_args.num_train_epochs) as progress_bar:
        for epoch in range(1, training_args.num_train_epochs + 1):
            training_loss = 0
            train_total = 0
            with torch.enable_grad():
                for batch_num, batch in enumerate(train_loader):
                    batch_size = len(batch["inputs"])
                    train_total += batch_size
                    optimizer.zero_grad()
                    # if prompt_model is not None:
                    #     optimizer_prompt.zero_grad()
                    loss, out = forward_loss(bert_impute, batch, nan_feed_data, fields, fields_dict, annoy, k,
                                             index_annoy_dict, tokenizer_data, ori_data, device)
                    loss_val = loss.item()
                    training_loss += loss_val
                    loss.backward()
                    optimizer.step()
                    # if prompt_model is not None:
                    #     optimizer_prompt.step()
                    step += batch_size
            progress_bar.update(1)
            progress_bar.set_postfix(epoch=epoch, loss=training_loss / train_total)
            if epoch % 10 == 0:
                bert_impute.eval()
                if prompt_model is not None:
                    prompt_model.eval()
                filled_valid_pd = valid_nan_data.copy()[decoder_column]
                with torch.no_grad():
                    valid_total = 0
                    for batch_num, batch in enumerate(valid_loader):
                        batch_size = len(batch["inputs"])
                        if prompt_model is not None:
                            begin = batch["begin"][0]
                            end = batch["end"][0]
                            if batch_num == 0:
                                print("\n" + begin + "\n" + end)
                        valid_total += batch_size
                        out, nn_indices = bert_impute(batch['inputs'].to(device), batch['mask_inputs'].to(device),
                                                      batch['target_inputs'].to(device),
                                                      batch['target_mask_inputs'].to(device),
                                                      annoy, k, tokenizer_data)
                        tup_ids = batch['relation_info']
                        # 将code进行拼接
                        nan_code = nan_feed_data[tup_ids, :]
                        M_code = M[tup_ids, :]
                        out_code = M_code * nan_code + (1 - M_code) * out
                        reverse_data = reconvert_data(out_code, fields, value_cat, decoder_column, enc)
                        filled_valid_pd.loc[tup_ids, :] = reverse_data.values

                    # 根据filled_pd和ground_truth计算损失
                    ground_truth_valid_pd = valid_ori_data.copy()[decoder_column]
                    cat_to_code_data, enc = categorical_to_code(ground_truth_valid_pd.copy(), value_cat, enc)
                    cat_to_code_data.columns = [x for x in range(cat_to_code_data.shape[1])]
                    valid_m_pd = valid_data_m.copy()[decoder_column]
                    RMSE, Acc = errorLoss(filled_valid_pd, ground_truth_valid_pd, valid_m_pd, value_cat,
                                          continuous_cols,
                                          enc)
                    if Acc > best_ACC:
                        best_impute_model = bert_impute
                        best_ACC = Acc
                        best_epoch = epoch
                        best_prompt_model = prompt_model
                bert_impute.train()
                if prompt_model is not None:
                    prompt_model.train()
    return best_impute_model, best_prompt_model