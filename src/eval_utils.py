import torch
import torch.nn as nn
import pdb
from sklearn import metrics
from src.model.metrics import _compute_f_pre_rec

def eval(args, model, img_encoder,loader, metric, device):
    model.eval()

    if args.crf_on or not args.sc_only:
        targets_all, outputs_all = [], []
    if args.task == 'SC':
        n_test_correct, n_test_total = 0, 0
        targets_all, outputs_all = None, None
        tp, fp, fn = {0: 0, 1: 0, 2: 0}, {0: 0, 1: 0, 2: 0}, {0: 0, 1: 0, 2: 0}

    for i, batch in enumerate(loader):
        # Forward pass
        aesc_infos = batch[args.task]
        # aesc_infos = {key: value for key, value in batch['AESC'].items()}
        with torch.no_grad():
            imgs_f = [x.numpy().tolist() for x in batch['image_features']]
            imgs_f = torch.tensor(imgs_f).to(device)
            imgs_f, img_mean, img_att = img_encoder(imgs_f, 7)
            img_att = img_att.view(-1, 2048, args.img_num).permute(0, 2, 1)
            # img_att = img_att[:, :args.img_num,:]

        if args.task == 'AESC':
            # predict = model.forward(
            predict = model.predict(
                input_ids=batch['input_ids'].to(device),
                #image_features=list(map(lambda x: x.to(device), img_att)),
                image_features=img_att.to(device),
                # image_features=None,
                sentiment_value=batch['sentiment_value'].to(device) if batch['sentiment_value'] is not None else None,
                noun_mask=batch['noun_mask'],
                attention_mask=batch['attention_mask'].to(device),
                syn_dep_adj_matrix=batch['syn_dep_matrix'].to(device),
                syn_dis_adj_matrix=batch['syn_dis_matrix'].to(device),
                aesc_infos=aesc_infos)
            print(aesc_infos['spans'].shape, aesc_infos['spans'], aesc_infos['labels'].shape, aesc_infos['labels'])
            # metric.evaluate(aesc_infos.to(device), predict)

            metric.evaluate(aesc_infos['spans'], predict,
                        aesc_infos['labels'].to(device))

        elif args.task == 'SC':
            predict = model.forward(
            # predict = model.predict(
                input_ids=batch['input_ids'].to(device),
                image_features=list(map(lambda x: x.to(device), img_att)),
                #image_features=img_att.to(device),
                # image_features=None,
                sentiment_value=batch['sentiment_value'].to(device) if batch['sentiment_value'] is not None else None,
                noun_mask=batch['noun_mask'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                syn_dep_adj_matrix=batch['syn_dep_matrix'].to(device),
                syn_dis_adj_matrix=batch['syn_dis_matrix'].to(device),
                aspect_mask=batch['aspect_mask'].to(device),
                aesc_infos=aesc_infos)
            
            targets = aesc_infos.to(device)
            n_test_correct += (torch.argmax(predict, -1) == targets).sum().item()
            n_test_total += len(predict)
            predict_ = torch.argmax(predict, -1)
            for i, p in enumerate(predict_):
                if p == targets[i]:
                    tp[p.item()] += 1
                else:
                    fp[p.item()] += 1
                    fn[targets[i].item()] += 1
            #targets_all = torch.cat((targets_all, targets), dim=0) if targets_all is not None else targets
            #outputs_all = torch.cat((outputs_all, predict), dim=0) if outputs_all is not None else predict

        # metric.evaluate(aesc_infos['spans'], predict,
        #                 aesc_infos['labels'].to(device))
        
        # targets = aesc_infos.to(device)

        # targets_all.append(targets)
        # outputs_all.extend(predict)

        # print("Eval {}/{}".format(i, len(loader)))
        # break

    # res = {}
    # targets_all = pad_token(targets_all).flatten()
    # outputs_all = pad_token(outputs_all).flatten()

    # recall = metrics.recall_score(targets_all.cpu(), outputs_all.cpu(), average='macro')
    # precision = metrics.precision_score(targets_all.cpu(), outputs_all.cpu(), average='macro')
    # f1 = metrics.f1_score(targets_all.cpu(), outputs_all.cpu(), average='macro')
    # model.train()

    # res['aesc_rec'] = round(recall * 100, 2)
    # res['aesc_pre'] = round(precision * 100, 2)
    # res['aesc_f'] = round(f1 * 100, 2)

    res = {}
    f_sum, pre_sum, rec_sum = 0, 0, 0
    if args.task == 'SC':
        test_acc = n_test_correct / n_test_total
        #f1_macro = metrics.f1_score(targets_all.cpu(), torch.argmax(outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')
        #f1_micro = metrics.f1_score(targets_all.cpu(), torch.argmax(outputs_all, -1).cpu(), labels=[0, 1, 2], average='micro')
        for tag in tp.keys():
            #assert tag not in (0, 1, self.conflict_id), (tag, self.conflict_id)
            tp_ = tp[tag]
            fn_ = fn[tag]
            fp_ = fp[tag]
            f, pre, rec = _compute_f_pre_rec(1, tp_, fn_, fp_)
            f_sum += f
            pre_sum += pre
            rec_sum += rec

        rec_sum /= (len(args.label_dict.values()) + 1e-12)
        pre_sum /= (len(args.label_dict.values()) + 1e-12)
        f_sum /= (len(args.label_dict.values()) + 1e-12)
        res['sc_f_ma'] = round(f_sum *100, 2)
        res['sc_f_mi'] = round(
            2 * pre_sum * rec_sum / (pre_sum + rec_sum + 1e-12) * 100, 2)
        res['sc_rec'] = round(rec_sum * 100, 2)
        res['sc_pre'] = round(pre_sum * 100, 2)
        
        model.train()

        res['sc_acc'] = round(test_acc * 100, 2)
        #res['sc_f_mi'] = round(f1_micro * 100, 2)
        #res['sc_f_ma'] = round(f1_macro * 100, 2)
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