from __future__ import print_function
from PIL import Image
import numpy as np
import torch.utils.data as data
import torch
import xlrd
import os
import pandas as pd
from glob import glob
workbook_path="./data_apex.xlsx"
workbook = xlrd.open_workbook(workbook_path)
Data_sheet = workbook.sheet_by_index(0)
rowNum = Data_sheet.nrows
colNum = Data_sheet.ncols
class Dataload(data.Dataset):
    def __init__(self, img_root, split, mode,leave_out, transform=None):
        global train_labels, test_labels, train_label, test_label, img_path
        self.transform = transform
        self.split = split
        self.img_root=img_root
        self.mode=mode
        self.train_labels = []
        self.test_labels = []

        if self.split == 'Training':
            self.flow = []
            self.apex=[]
            for i in range(1, rowNum):
                rows = Data_sheet.row_valbel=str(rows[3]).split('.')[0]
                subject=str(rows[1]).split('.')[0]
                clip=str(rows[2])
                catego=str(rows[10])
                if catego!=leave_out:
                    if catego==leave_out:
                        print('数据泄露')
                    pg = rows[8]
                    pg= str(pg.split('/')[-1]).split('.')[0]
                    if 'casme' in catego:
                        img_path = f"{self.img_root}/casme2/{subject}/{clip}"
                    elif 'samm' in catego:
                        subject = subject.rjust(3, '0')
                        img_path = f"{self.img_root}/samm/{subject}/{clip}"
                    elif 'smic' in catego:
                        subject = subject.rjust(2, '0')
                        img_path = f"{self.img_root}/smic/HS_long/SMIC_HS_E/s{subject}/{clip}"
                    frames = glob(os.path.join(img_path, f'*_{pg}.npz'))
                    for i, npz_path in enumerate(frames[0:]):
                        preprocess_image = np.load(npz_path)
                        flow = torch.FloatTensor(preprocess_image['flow'])
                        apex_frame = torch.FloatTensor(preprocess_image['apex_frame'])
                        onset_frame = torch.FloatTensor(preprocess_image['onset_frame'])
                        flow = flow.permute(2, 0, 1)
                        apex_frame = apex_frame.permute(2, 0, 1)
                        onset_frame = onset_frame.permute(2, 0, 1)
                        stream_one = torch.FloatTensor(preprocess_image[ Dataload.build_mode(self.mode)])
                        stream_one = stream_one.unsqueeze(0)
                        image_one = torch.cat([flow, stream_one], dim=0)
                        image_two = apex_frame
                        self.flow.append(image_one)
                        self.apex.append(image_two)
                        self.train_labels.append(train_label)

        if self.split == 'Testing':
            self.flow = []
            self.apex = []
            for i in range(1, rowNum):
                rows = Data_sheet.row_values(i)
                test_label = str(rows[3]).split('.')[0]
                subject = str(rows[1]).split('.')[0]
                clip = str(rows[2])
                catego = str(rows[10])

                if catego == leave_out:
                    pg = rows[8]
                    pg = str(pg.split('/')[-1]).split('.')[0]
                    if 'casme' in catego:
                        img_path = f"{self.img_root}/casme2/{subject}/{clip}"
                    elif 'samm' in catego:
                        subject = subject.rjust(3, '0')
                        img_path = f"{self.img_root}/samm/{subject}/{clip}"
                    elif 'smic' in catego:
                        subject = subject.rjust(2, '0')
                        img_path = f"{self.img_root}/smic/HS_long/SMIC_HS_E/s{subject}/{clip}"
                    npz_path = f"{img_path}/{subject}_{clip}_{pg}.npz"

                    preprocess_image = np.load(npz_path)
                    flow = torch.FloatTensor(preprocess_image['flow'])
                    apex_frame = torch.FloatTensor(preprocess_image['apex_frame'])
                    onset_frame = torch.FloatTensor(preprocess_image['onset_frame'])
                    flow = flow.permute(2, 0, 1)
                    apex_frame = apex_frame.permute(2, 0, 1)
                    onset_frame = onset_frame.permute(2, 0, 1)
                    stream_one = torch.FloatTensor(preprocess_image[Dataload.build_mode(self.mode)])
                    stream_one = stream_one.unsqueeze(0)
                    image_one = torch.cat([flow, stream_one], dim=0)
                    image_two = apex_frame
                    self.flow.append(image_one)
                    self.apex.append(image_two)
                    self.test_labels.append(test_label)
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        global target, image_one, image_two, label
        if self.split == 'Training':
            image_one,image_two, label = self.flow[index], self.apex[index],self.train_labels[index]
        elif self.split == 'Testing':
            image_one, image_two, label = self.flow[index], self.apex[index], self.test_labels[index]

        image_one= self.transform(image_one)
        image_two = self.transform(image_two)
        label=int(label)
        return image_one,image_two, label


    def __len__(self):
        if self.split == 'Training':
            return len(self.train_labels)
        elif self.split == 'Testing':

            return len(self.test_labels)



