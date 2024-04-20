from __future__ import print_function
from PIL import Image
import numpy as np
import torch.utils.data as data
import torch
import xlrd
import os

from glob import glob

workbook_path="./dataset/CASME2-coding-20140508.xlsx"

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
                if rows[8] == 'disgust' or rows[8] == 'repression' or rows[8] == 'happiness' or rows[8] == 'surprise' or rows[8] == 'others':
                    if 'sub' + rows[0] != leave_out:

                        # label
                        if rows[8] == 'happiness':
                            train_label = 0
                        elif rows[8] == 'disgust':
                            train_label = 1
                        elif rows[8] == 'repression':
                            train_label = 2
                        elif rows[8] == 'surprise':
                            train_label = 3
                        elif rows[8] == 'others':
                            train_label = 4
                        else:
                            print('A wrong sample has been selected.')
                        img_path = f"{self.img_root}/casme2/sub{rows[0]}/{rows[1]}"
                        pg=str(rows[4]).split('.')[0]

                        frames = glob(os.path.join(img_path, f'*_img{pg}.npz'))
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
                if rows[8] == 'disgust' or rows[8] == 'repression' or rows[8] == 'happiness' or rows[8] == 'surprise' or rows[8] == 'others':
                    if 'sub' + rows[0] == leave_out:
                        if rows[8] == 'happiness':
                            test_label = 0
                        elif rows[8] == 'disgust':
                            test_label = 1
                        elif rows[8] == 'repression':
                            test_label = 2
                        elif rows[8] == 'surprise':
                            test_label = 3
                        elif rows[8] == 'others':
                            test_label = 4
                        else:
                            print('A wrong sample has been selected.')
                        img_path = f"{self.img_root}/casme2/sub{rows[0]}/{rows[1]}"
                        pg = str(rows[4]).split('.')[0]
                        npz_path = f"{img_path}/sub{rows[0]}_{rows[1]}_img{pg}.npz"

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

