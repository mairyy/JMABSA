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
        if args.aesc_enabled == False:
            aesc_infos = batch['SC']
        else:
            if args.task == 'twitter_ae':
                aesc_infos = {
                    key: value
                    for key, value in batch['TWITTER_AE'].items()
                }
            elif args.task == 'twitter_sc':
                aesc_infos = {
                    key: value
                    for key, value in batch['TWITTER_SC'].items()
                }
            else:
                aesc_infos = {key: value for key, value in batch['AESC'].items()}
        #with torch.no_grad():
            #imgs_f = [x.numpy().tolist() for x in batch['image_features']]
            #imgs_f = torch.tensor(imgs_f).to(device)
            #imgs_f, img_mean, img_att = img_encoder(imgs_f, 7)
            #img_att = img_att.view(-1, 2048, args.img_num).permute(0, 2, 1)
            #img_att = img_att[:, :args.img_num,:]

        if args.aesc_enabled:
            predict = model.predict(
                input_ids=batch['input_ids'].to(device),
                #image_features=list(map(lambda x: x.to(device), img_att)),
                image_features=None,
                sentiment_value=batch['sentiment_value'].to(device) if batch['sentiment_value'] is not None else None,
                noun_mask=batch['noun_mask'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                syn_dep_adj_matrix=batch['syn_dep_matrix'].to(device),
                syn_dis_adj_matrix=batch['syn_dis_matrix'].to(device),
                aesc_infos=aesc_infos)
            metric.evaluate(aesc_infos['spans'], predict,
                            aesc_infos['labels'].to(device))
        else:
            predict = model.forward(
                input_ids=batch['input_ids'].to(device),
                #image_features=list(map(lambda x: x.to(device), img_att)),
                image_features=None,
                sentiment_value=batch['sentiment_value'].to(device) if batch['sentiment_value'] is not None else None,
                noun_mask=batch['noun_mask'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                syn_dep_adj_matrix=batch['syn_dep_matrix'].to(device),
                syn_dis_adj_matrix=batch['syn_dis_matrix'].to(device),
                aesc_infos=aesc_infos,
                aspect_mask=batch['aspect_mask'].to(device),
                )
            targets = aesc_infos.to(device)
            if not args.crf_on:
                if not args.sc_only:
                    targets_all.append(targets)
                    outputs_all.extend(predict)
                    # predict = torch.argmax(predict, -1)
                    # print(predict[0], targets[0])
                    # targets_all.append(targets)
                    # outputs_all.append(predict)
                else:
                    n_test_correct += (torch.argmax(predict, -1) == targets).sum().item()
                    n_test_total += len(predict)
                    targets_all = torch.cat((targets_all, targets), dim=0) if targets_all is not None else targets
                    outputs_all = torch.cat((outputs_all, predict), dim=0) if outputs_all is not None else predict
            else:
                targets_all.append(targets)
                outputs_all.extend(predict)

        print("Eval {}/{}".format(i, len(loader)))
        # break

    if args.aesc_enabled == False:
        res = {}
        if args.crf_on or not args.sc_only:
            targets_all = pad_token(targets_all).flatten()
            outputs_all = pad_token(outputs_all).flatten()
            #print(targets_all, outputs_all, targets_all.shape, outputs_all.shape)
            #acc = metrics.accuracy_score(targets_all.cpu(), outputs_all.cpu())
            recall = metrics.recall_score(targets_all.cpu(), outputs_all.cpu(), average='macro')
            precision = metrics.precision_score(targets_all.cpu(), outputs_all.cpu(), average='macro')
            f1 = metrics.f1_score(targets_all.cpu(), outputs_all.cpu(), average='macro')
            model.train()

            res['aesc_rec'] = round(recall * 100, 2)
            res['aesc_pre'] = round(precision * 100, 2)
            res['aesc_f'] = round(f1 * 100, 2)
            return res
        else:
            test_acc = n_test_correct / n_test_total
            f1 = metrics.f1_score(targets_all.cpu(), torch.argmax(outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')
            model.train()

            res['sc_acc'] = round(test_acc * 100, 2)
            res['sc_f'] = round(f1 * 100, 2)
            return res
    
    res = metric.get_metric()
    model.train()
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