import torch
import torch.nn as nn
import pdb
from sklearn import metrics

def eval(args, model, img_encoder,loader, metric, device):
    model.eval()

    if args.crf_on or not args.sc_only:
        targets_all, outputs_all = [], []
    else:
        n_test_correct, n_test_total = 0, 0
        targets_all, outputs_all = None, None

    for i, batch in enumerate(loader):
        # Forward pass
        aesc_infos = batch['AESC']
        #with torch.no_grad():
            #imgs_f = [x.numpy().tolist() for x in batch['image_features']]
            #imgs_f = torch.tensor(imgs_f).to(device)
            #imgs_f, img_mean, img_att = img_encoder(imgs_f, 7)
            #img_att = img_att.view(-1, 2048, args.img_num).permute(0, 2, 1)
            #img_att = img_att[:, :args.img_num,:]

        predict = model.forward(
            input_ids=batch['input_ids'].to(device),
            #image_features=list(map(lambda x: x.to(device), img_att)),
            image_features=None,
            sentiment_value=batch['sentiment_value'].to(device) if batch['sentiment_value'] is not None else None,
            noun_mask=batch['noun_mask'].to(device),
            attention_mask=batch['attention_mask'].to(device),
            syn_dep_adj_matrix=batch['syn_dep_matrix'].to(device),
            syn_dis_adj_matrix=batch['syn_dis_matrix'].to(device),
            aesc_infos=aesc_infos)

        targets = aesc_infos.to(device)

        targets_all.append(targets)
        outputs_all.extend(predict)

        print("Eval {}/{}".format(i, len(loader)))
        # break

    res = {}
    targets_all = pad_token(targets_all).flatten()
    outputs_all = pad_token(outputs_all).flatten()

    recall = metrics.recall_score(targets_all.cpu(), outputs_all.cpu(), average='macro')
    precision = metrics.precision_score(targets_all.cpu(), outputs_all.cpu(), average='macro')
    f1 = metrics.f1_score(targets_all.cpu(), outputs_all.cpu(), average='macro')
    model.train()

    res['aesc_rec'] = round(recall * 100, 2)
    res['aesc_pre'] = round(precision * 100, 2)
    res['aesc_f'] = round(f1 * 100, 2)
    return res


def pad_token(tokens):
    if type(tokens[0]) == list:
        max_len = max(len(token) for token in tokens)
        pad_result = torch.full((len(tokens), max_len), 0)
        for i, x in enumerate(tokens):
            pad_result[i, :len(x)] = torch.tensor(tokens[i], dtype=torch.long)
        return pad_result
    max_len = max(tensor.shape[1] for tensor in tokens)
    padded_tensors = [torch.nn.functional.pad(tensor, (0, max_len - tensor.shape[1])) for tensor in tokens]
    return torch.cat(padded_tensors, dim=0)