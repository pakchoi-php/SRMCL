from __future__ import print_function
from PIL import Image
import numpy as np
import torch.utils.data as data
import torch
import xlrd
import os

from glob import glob

workbook_path="./dataset/smic_3.xlsx"

workbook = xlrd.open_workbook(workbook_path)
Data_sheet = workbook.sheet_by_index(0)
rowNum = Data_sheet.nrows
colNum = Data_sheet.ncols


class Dataload(data.Dataset):
    def __init__(self, img_root, split, mode,leave_out, transform=None):
        global train_labels, test_labels, train_label, test_label
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
                rows = Data_sheet.row_values(i)
                rows[1] = str(rows[1]).split('.')[0]
                if 's' + rows[1] != leave_out:
                    rows[1]= rows[1].zfill(2)
                    img_path = f"{self.img_root}/smic/HS_long/SMIC_HS_E/s{rows[1]}/{rows[2]}"
                    # pg=str(rows[5]).split('.')[0]
                    pg = str(rows[8].split('/')[-1]).split('.')[0]

                    frames = glob(os.path.join(img_path, f'*_{pg}.npz'))
                    # if len(frames)<=1:
                    #     print(img_path)


                    for i, npz_path in enumerate(frames[0:]):
                        preprocess_image = np.load(npz_path)
                        # Load in the flow
                        # print("##",npz_path)
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
                        self.train_labels.append(rows[3])
                        ##############################################经过相同的transform

        if self.split == 'Testing':
            self.flow = []
            self.apex = []


            for i in range(1, rowNum):
                rows = Data_sheet.row_values(i)

                rows[1] = str(rows[1]).split('.')[0]
                if 's' + rows[1] == leave_out:
                    rows[1] = rows[1].zfill(2)

                    img_path = f"{self.img_root}/smic/HS_long/SMIC_HS_E/s{rows[1]}/{rows[2]}"
                    pg = str(rows[8].split('/')[-1]).split('.')[0]

                    npz_path = f"{img_path}/{rows[1]}_{rows[2]}_{pg}.npz"

                    preprocess_image = np.load(npz_path)

                    flow = torch.FloatTensor(preprocess_image["flow"])
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
                    self.test_labels.append(rows[3])

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
            label=int(label)
        elif self.split == 'Testing':
            image_one, image_two, label = self.flow[index], self.apex[index], self.test_labels[index]
            label = int(label)
        image_one= self.transform(image_one)
        image_two = self.transform(image_two)

        return image_one,image_two, label


    def __len__(self):
        if self.split == 'Training':
            return len(self.train_labels)
        elif self.split == 'Testing':
            return len(self.test_labels)

    @staticmethod
    def build_mode(mode):
        if mode == "F":
            return "mag"
        elif mode == "G":
            return "gray"
        elif mode == "S":
            return "strain"

