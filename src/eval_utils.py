
import torch
import torch.nn as nn
import pdb
from sklearn import metrics

def eval(args, model, img_encoder,loader, metric, device):
    model.eval()

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

        with torch.no_grad():
            imgs_f = [x.numpy().tolist() for x in batch['image_features']]
            imgs_f = torch.tensor(imgs_f).to(device)
            imgs_f, img_mean, img_att = img_encoder(imgs_f)
            img_att = img_att.view(-1, 2048, 49).permute(0, 2, 1)
        
        if args.aesc_enabled:
            predict = model.predict(
                input_ids=batch['input_ids'].to(device),
                image_features=list(map(lambda x: x.to(device), img_att)),
                sentiment_value=batch['sentiment_value'].to(device) if batch['sentiment_value'] is not None else None,
                noun_mask=batch['noun_mask'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                dependency_matrix=batch['dependency_matrix'].to(device),
                aesc_infos=aesc_infos)    
            metric.evaluate(aesc_infos['spans'], predict, aesc_infos['labels'].to(device))
        else:
            predict = model.forward(
                input_ids=batch['input_ids'].to(device),
                image_features=list(map(lambda x: x.to(device), img_att)),
                sentiment_value=batch['sentiment_value'].to(device) if batch['sentiment_value'] is not None else None,
                noun_mask=batch['noun_mask'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                dependency_matrix=batch['dependency_matrix'].to(device),
                aesc_infos=aesc_infos,
                aspect_mask=batch['aspect_mask'].to(device),
                short_mask=batch['short_mask'].to(device))    
            targets = aesc_infos.to(device)
            n_test_correct += (torch.argmax(predict, -1) == targets).sum().item()
            n_test_total += len(predict)
            targets_all = torch.cat((targets_all, targets), dim=0) if targets_all is not None else targets
            outputs_all = torch.cat((outputs_all, predict), dim=0) if outputs_all is not None else predict

    if args.aesc_enabled == False:
        test_acc = n_test_correct / n_test_total
        f1 = metrics.f1_score(targets_all.cpu(), torch.argmax(outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')
        model.train()

        res['sc_acc'] = round(test_acc * 100, 2)
        res['sc_f'] = round(f1 * 100, 2)
        return res
        # break

    res = metric.get_metric()
    model.train()
    return res