import torch
from tqdm import tqdm

from models.loss_function import forward_loss, forward_loss_distinct
from uti.util import reconvert_data, categorical_to_code, errorLoss


def test_impute_model(model, num_test, test_loader, path,
                       continuous_cols, value_cat,  decoder_column,
                       nan_feed_data, ori_data,
                      train_nan_data, train_ori_data,
                      train_data_m, M, fields, fields_dict, enc,
                       annoy, index_annoy_dict):
    best_impute_model = model
    best_impute_model.eval()
    torch.save(best_impute_model.state_dict(), path)  # 保存模型权重
    filled_pd = train_nan_data.copy()[decoder_column]
    with torch.no_grad(), tqdm(total=num_test) as progress_bar_test:
        test_total = 0
        for batch_num, batch in enumerate(test_loader):
            batch_size = len(batch["source_input"])
            test_total += batch_size
            loss, out = forward_loss(best_impute_model, batch, nan_feed_data, fields, fields_dict, annoy,
                                     index_annoy_dict, ori_data)
            tup_ids = batch['relation_info']
            # 将code进行拼接
            nan_code = nan_feed_data[tup_ids, :]
            M_code = M[tup_ids, :]
            out_code = M_code * nan_code + (1 - M_code) * out
            reverse_data = reconvert_data(out_code, fields, value_cat, decoder_column, enc)
            filled_pd.loc[tup_ids, :] = reverse_data.values
            progress_bar_test.update(batch_size)
            progress_bar_test.set_postfix(batch=test_total / num_test)
    # 根据filled_pd和ground_truth计算损失
    ground_truth_pd = train_ori_data.copy()[decoder_column]
    cat_to_code_data, enc = categorical_to_code(ground_truth_pd.copy(), value_cat, enc)
    cat_to_code_data.columns = [x for x in range(cat_to_code_data.shape[1])]
    test_m_pd = train_data_m.copy()[decoder_column]
    RMSE, Acc = errorLoss(filled_pd, ground_truth_pd, test_m_pd, value_cat, continuous_cols, enc)
    return RMSE, Acc





def test_impute_model_distinct(model, num_test, test_loader, path,
                       continuous_cols, value_cat,  decoder_column,
                       nan_feed_data, ori_data,
                      train_nan_data, train_ori_data,
                      train_data_m, M, fields, fields_dict, enc,
                       annoy, k, tokenizer_data, index_annoy_dict, nan_data, device, annoy_label):
    best_impute_model = model
    best_impute_model.eval()
    torch.save(best_impute_model.state_dict(), path)  # 保存模型权重
    # model.load_state_dict(torch.load('model_path.pth'))
    filled_pd = train_nan_data.copy()[decoder_column]
    with torch.no_grad(), tqdm(total=num_test) as progress_bar_test:
        test_total = 0
        for batch_num, batch in enumerate(test_loader):
            batch_size = len(batch["source_input"])
            test_total += batch_size
            # out, nn_indices = best_impute_model(batch['source_input'], batch['target_input'], batch['mask_list'], annoy, k, index_annoy_dict, nan_data, tokenizer_data)
            loss, out = forward_loss_distinct(best_impute_model, batch, nan_feed_data, fields, fields_dict, annoy, k,
                                     index_annoy_dict, tokenizer_data, ori_data, nan_data, device,annoy_label,enc)
            tup_ids = batch['relation_info']
            # 将code进行拼接
            nan_code = nan_feed_data[tup_ids, :]
            M_code = M[tup_ids, :]
            out_code = M_code * nan_code + (1 - M_code) * out
            reverse_data = reconvert_data(out_code, fields, value_cat, decoder_column, enc)
            filled_pd.loc[tup_ids, :] = reverse_data.values
            progress_bar_test.update(batch_size)
            progress_bar_test.set_postfix(batch=test_total / num_test)

    # 根据filled_pd和ground_truth计算损失
    ground_truth_pd = train_ori_data.copy()[decoder_column]
    cat_to_code_data, enc = categorical_to_code(ground_truth_pd.copy(), value_cat, enc)
    cat_to_code_data.columns = [x for x in range(cat_to_code_data.shape[1])]
    test_m_pd = train_data_m.copy()[decoder_column]
    RMSE, Acc = errorLoss(filled_pd, ground_truth_pd, test_m_pd, value_cat, continuous_cols, enc)
    return RMSE, Acc