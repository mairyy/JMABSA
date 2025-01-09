import torch
from torchvision import transforms
import json
import os
import src.resnet.resnet as resnet
from src.resnet.resnet_utils import myResnet
from PIL import Image
import pickle

device = torch.device('cpu')
net = getattr(resnet, 'resnet152')()
net.load_state_dict(torch.load('src/resnet/resnet152-b121ed2d.pth'))
img_encoder = myResnet(net, True, device)
img_encoder.to(device)
count_img_error = 0

crop_size = 224
transform = transforms.Compose([
    transforms.Resize((crop_size, crop_size)),  # args.crop_size, by default it is set to be 224
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()])

def get_img_feature(path='./src/data/twitter2015', mode='train'):
    dataset = json.load(open(os.path.join(path, mode + '.json')))
    path_img = './src/data/IJCAI2019_data/twitter2015_images'
    features = {}

    for d in dataset:
        id = d['image_id']
        image_path = os.path.join(path_img, id)
        if not os.path.exists(image_path):
            print(image_path)
        try:
            image = Image.open(image_path).convert('RGB')
            image = transform(image)
        except:
            count_img_error += 1
            # print('image has problem!')
            image_path_fail = os.path.join(path_img, '17_06_4705.jpg')
            image = Image.open(image_path_fail).convert('RGB')
            image = transform(image)

        with torch.no_grad():
            imgs_f = torch.tensor([image.numpy().tolist()]).to(device)
            imgs_f, img_mean, img_att = img_encoder(imgs_f, 3)
            #print(img_att.shape)
            img_att = img_att.view(-1, 2048, 9).permute(0, 2, 1)
        #print(img_att[0], img_att, img_att[0].shape)
        #features.append({id: img_att[0]})
        features[id] = img_att[0]
    
    with open(os.path.join(path, mode + '_img_feats.pkl'), 'wb') as outfile:
        pickle.dump(features, outfile)

get_img_feature()
get_img_feature(mode='dev')
get_img_feature(mode='test')
with open('/Users/admin/Documents/Projects/JMABSA/src/data/twitter2015/train_img_feats.pkl', 'rb') as file:
    data = pickle.load(file)
print(data.get("1860693.jpg"))
#for d in data:
#    print(d)
#move to root folder to run