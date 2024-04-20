from __future__ import print_function

from multiprocessing import freeze_support

import torch
import torch.backends.cudnn as cudnn
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import os
from Dataloader.dataloader_casmeII import Dataload
from torch.autograd import Variable
from Model.VIT_SR import  MaskedAutoencoderViT_SR
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd
from functools import partial
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
use_cuda = True
torch.backends.cudnn.benchmark = True
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambd=lambda x: x / 255),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])

])

model_dir ="./best_model/casmeII_SR"
all_sub_list = ['sub17', 'sub02', 'sub05', 'sub09', 'sub10', 'sub11', 'sub12', 'sub19', 'sub20', 'sub23', 'sub26',
                'sub01', 'sub03', 'sub04', 'sub06', 'sub07', 'sub08', 'sub13', 'sub14', 'sub15', 'sub16', 'sub18',
                'sub21', 'sub22', 'sub24', 'sub25']
subnumber=len(all_sub_list)

if __name__=='__main__':
    correct = 0
    total = 0
    accuracysub=0
    y_true = []
    y_pre = []
    for leave_out_sub in all_sub_list:
        print('==> Leaving out ' + leave_out_sub)
        net = MaskedAutoencoderViT_SR(
            patch_size=16, embed_dim=768, depth=3, num_heads=16,
            decoder_embed_dim=512, decoder_depth=3, decoder_num_heads=16,
            mlp_ratio=2, norm_layer=partial(nn.LayerNorm, eps=1e-6))
        model_path = os.path.join(model_dir, f'casmeII_SR_{leave_out_sub}.pt')
        pretrained_dict = torch.load( model_path)
        model_dict = net.state_dict()
        keys = []
        for k, v in model_dict.items():
            if k.startswith('memory'):
                continue
            keys.append(k)
        pretrained_dict = {k: v for k, v in  pretrained_dict.items() if k in model_dict}
        # 参数更新
        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)
        net.eval()
        with torch.no_grad():
            if use_cuda:
                net.cuda()
            test_set = Dataload(img_root="./dataset/", split='Testing', mode='F', leave_out=leave_out_sub,
                                 transform=transform_test)
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=30, shuffle=False,
                                                     num_workers=0)
            for i, (image_one, image_two, label) in enumerate(test_loader):
                print(image_one.shape)
                print(label.shape)
                if use_cuda:
                    image_one, image_two, y = image_one.cuda(), image_two.cuda(), label.cuda()  # convert input data to GPU Variable
                with torch.no_grad():
                     image_one, image_two, y = Variable(image_one), Variable(image_two), Variable(y)# convert input data to GPU Variable
                x_cls,y_pred= net.forward_class(image_one, mask_ratio=0)
                _, predicted = torch.max(y_pred.data, 1)
                total += y.size(0)
                correct += predicted.eq(y.data).cpu().sum()
            accuracy=correct/total
            accuracysub=accuracysub+accuracy
                
            
    # calculate accuracy
    Test_acc=accuracysub/subnumber
    print('Test accuracy: %0.6f' % Test_acc)





