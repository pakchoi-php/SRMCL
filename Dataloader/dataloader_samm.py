from __future__ import print_function
from PIL import Image
import numpy as np
import torch.utils.data as data
import torch
import xlrd
import os

from glob import glob
workbook_path="./dataset/SAMM_Micro_FACS_Codes_v2.xlsx"

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
            for i in range(14, rowNum):
                rows = Data_sheet.row_values(i)

                if rows[9] == 'Anger' or rows[9] == 'Contempt' or rows[9] == 'Happiness' or rows[9] == 'Surprise' or \
                        rows[9] == 'Other':
                    if rows[0] != leave_out:
                        if rows[9] == 'Anger':
                            train_label = 0
                        elif rows[9] == 'Contempt':
                            train_label = 1
                        elif rows[9] == 'Happiness':
                            train_label = 2
                        elif rows[9] == 'Surprise':
                            train_label = 3
                        elif rows[9] == 'Other':
                            train_label = 4
                        else:
                            print('A wrong sample has been selected.')
                        img_path = f"{self.img_root}/samm/{rows[0]}/{rows[1]}"
                        pg=str(rows[4]).split('.')[0]
                        frames = glob(os.path.join(img_path,f'{rows[0]}_{rows[1]}*.npz'))
                        for i, npz_path in enumerate(frames[0:]):
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
                            self.train_labels.append(train_label)

        if self.split == 'Testing':
            self.flow = []
            self.apex = []


            for i in range(1, rowNum):
                rows = Data_sheet.row_values(i)
                if rows[9] == 'Anger' or rows[9] == 'Contempt' or rows[9] == 'Happiness' or rows[9] == 'Surprise' or \
                        rows[9] == 'Other':
                    if rows[0] == leave_out:

                        if rows[9] == 'Anger':
                            test_label = 0
                        elif rows[9] == 'Contempt':
                            test_label= 1
                        elif rows[9] == 'Happiness':
                            test_label= 2
                        elif rows[9] == 'Surprise':
                            test_label = 3
                        elif rows[9] == 'Other':
                            test_label = 4
                        else:
                            print('A wrong sample has been selected.')
                        img_path = f"{self.img_root}/samm/{rows[0]}/{rows[1]}/"
                        pg = str(rows[4]).split('.')[0]
                        npz_path = glob(f"{img_path}{rows[0]}_{rows[1]}_{rows[0]}_*{pg}.npz")
                        preprocess_image = np.load(npz_path[0])
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

