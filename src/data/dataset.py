import torch
import numpy as np
import json
import csv
import os
import json
import torch.utils.data as data
from PIL import Image
from transformers import AutoTokenizer
from torchvision import transforms
import pdb
import pickle

class Twitter_Dataset(data.Dataset):
    def __init__(self, args, img_path,infos, split):
        self.path_img = img_path
        self.infos = json.load(open(infos, 'r'))

        if split == 'train':
            self.data_set = json.load(
                open(self.infos['data_dir'] + '/train_preprocessed.json', 'r'))
            #self.img_path = self.infos['data_dir'] + '/train_img_feats.pkl'
        elif split == 'dev':
            self.data_set = json.load(
                open(self.infos['data_dir'] + '/dev_preprocessed.json', 'r'))
            #self.img_path = self.infos['data_dir'] + '/dev_img_feats.pkl'
        elif split == 'test':
            self.data_set = json.load(
                open(self.infos['data_dir'] + '/test_preprocessed.json', 'r'))
            #self.img_path = self.infos['data_dir'] + '/test_img_feats.pkl'
        else:
            raise RuntimeError("split type is not exist!!!")
        
        if args.task == 'SC':
            self.data_set = self.get_dataset(self.data_set)

        crop_size = 224
        self.transform = transforms.Compose([
            transforms.Resize((crop_size, crop_size)),  # args.crop_size, by default it is set to be 224
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])

        self.count_img_error=0
        self.args = args

    def __len__(self):
        return len(self.data_set)

    def get_dataset(self, data):
        new_dataset = []
        for d in data:
            for aspect in d['aspects']:
                words = d['words']
                image_id = d['image_id']
                noun = d['noun']
                new_dataset.append({'words': words, 'image_id': image_id, 'aspects': aspect, 'noun': noun, 'head': d['head'], 'short': d['short']\
                                    , 'deprel': d['deprel'], 'syn_dep_adj': d['syn_dep_adj'], 'syn_dis_adj': d['syn_dis_adj']})
        return new_dataset
    
    def get_img_feature(self,id):
        image_path = os.path.join(self.path_img, id)
        if not os.path.exists(image_path):
            print(image_path)
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
        except:
            self.count_img_error += 1
            # print('image has problem!')
            image_path_fail = os.path.join(self.path_img, '17_06_4705.jpg')
            image = Image.open(image_path_fail).convert('RGB')
            image = self.transform(image)
        return image

    #def get_img_feature(self, id):
    #    with open(self.img_path, 'rb') as outfile:
    #        img_features = pickle.load(outfile)
    #    return img_features.get(id)

    def get_aesc_spans(self, dic):
        aesc_spans = []
        for x in dic:
            aesc_spans.append((x['from'], x['to'], x['polarity']))
        return aesc_spans

    def get_gt_aspect_senti(self, dic):
        gt = []
        for x in dic:
            gt.append((' '.join(x['term']), x['polarity']))
        return gt
    
    def get_aspect(self, dic):
        aspects = []
        for x in dic:
            aspects.append(x['term'])
        return aspects
    
    def get_polarity(self, dic):
        pol = []
        for x in dic:
            pol.append(x['polarity'])
        return pol
    
    def get_label(self, dic):
        dic_ = {}
        for x in dic:
            aspect = x['term']
            for a in aspect:
                dic_[a] = x['polarity']
        return dic_

    def __getitem__(self, index):
        output = {}
        data = self.data_set[index]
        img_id = data['image_id']
        img_feature = self.get_img_feature(img_id)
        output['img_feat'] = img_feature

        output['sentence'] = ' '.join(data['words'])
        # output['sentence'] = 'The sentence "' + ' '.join(data['words']) + '" has the aspect "' + ' '.join(data['aspects']['term']) + '"'
        # add
        # print(output['sentence'])
        output['noun']=data['noun']
        output['image_id'] = img_id

        if not self.args.aesc_enabled:
            if self.args.task == 'AESC':
                output['labels'] = self.get_label(data['aspects'])
            elif self.args.task == 'SC':
                output['labels'] = self.args.label_dict[data['aspects']['polarity']]
                output['aspects'] = data['aspects']['term']
        else:
            aesc_spans = self.get_aesc_spans(data['aspects'])
            output['aesc_spans'] = aesc_spans
            gt = self.get_gt_aspect_senti(data['aspects'])
            output['gt'] = gt
        output['syn_dep_adj'] = [(d[0], d[1], d[2]) for d in data['syn_dep_adj']]
        output['syn_dis_adj'] = [(d[0], d[1], d[2]) for d in data['syn_dis_adj']]

        # aesc_spans = self.get_aesc_spans(data['aspects'])
        # output['aesc_spans'] = aesc_spans
        # gt = self.get_gt_aspect_senti(data['aspects'])
        # output['gt'] = gt
        
        return output


class TRC_Dataset(data.Dataset):
    def __init__(self,infos):
        self.infos = json.load(open(infos, 'r'))
        self.path_img = self.infos["img_path"]
        self.data_set = json.load(open(self.infos['data_dir'] + '/trc.json', 'r'))

        crop_size = 224
        self.transform = transforms.Compose([
            transforms.Resize((crop_size, crop_size)),  # args.crop_size, by default it is set to be 224
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])
        self.count_img_error=0

    def __len__(self):
        return len(self.data_set)


    def get_img_feature(self,id):
        image_path = os.path.join(self.path_img, id)
        if not os.path.exists(image_path):
            print(image_path)
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        # try:
        #     image = Image.open(image_path).convert('RGB')
        #     image = self.transform(image)
        # except:
        #     with open('/home/zhouru/ABSA3/image_not_found.txt','a') as file:
        #         file.write(id+'\r')
        #     self.count_img_error += 1
        #     # print('image has problem!')
        #     image_path_fail = os.path.join('/home/zhouru/IJCAI2019_data/twitter2015_images', '17_06_4705.jpg')
        #     image = Image.open(image_path_fail).convert('RGB')
        #     image = self.transform(image)
            # print("img_not_found={}".format(self.count_img_error))
        return image


    def __getitem__(self, index):
        output = {}
        data = self.data_set[index]
        img_id = data['image_id']
        img_feature = self.get_img_feature(img_id)
        output['img_feat'] = img_feature
        output['sentence'] = ' '.join(data['words'])
        output['image_id'] = img_id
        output['ifpairs'] = data['ifpairs'][1]
        return output