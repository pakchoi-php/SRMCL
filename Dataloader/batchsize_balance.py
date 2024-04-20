import torch.utils.data

import numpy as np
##获取每类样本对应的索引，用字典保存，并将每类的索引打乱，因为此采样器返回的就是索引列表，
# 用于在Dataset中获取样本的，相当于其中的index

from torch.utils.data.sampler import BatchSampler
class BalancedBatchSampler(BatchSampler):

    def __init__(self,dataset, n_classes, n_samples):



        self.labels =np.array(dataset.train_labels)  #所有样本的labels

        #print("000", self.labels)
        self.labels_set = list(set(self.labels)) #0～199,如果是1～200会报错的
        self.label_to_indices = {label: np.where(self.labels == label)[0] #返回每类中的样本对应的index，字典
                                 for label in self.labels_set}
        #print("00",self.label_to_indices)
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l]) #将每类对应的索引打乱

        self.used_label_indices_count = {label: 0 for label in self.labels_set}  ##每类样本使用过的数量
        self.count = 0 #用过的图片数量，用于统计看够不够下一个batch用

        self.n_classes = n_classes
        self.n_samples = n_samples
        self.dataset = dataset

        self.batch_size = self.n_samples * self.n_classes#此处保存一个batch样本的数量

    def __iter__(self):
        self.count = 0 #只用关心样本使用一次的事情，也就是一个epoch后就归零了

        while self.count + self.batch_size < len(self.dataset): #也就是使用过的图片数量再加上一个batch仍然小于总数，那么可以继续提供一个batch的图片
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            #print("classes",classes)#选类别，不放回的抽，也就是抽出来的不能有重复
            indices = []
            for class_ in classes: # 1 ， 3 ，4

                indices.extend(self.label_to_indices[class_][ #label_to_indices是一个字典，{class:[index]}
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples]) #顺序获取每类中的n个样本
                self.used_label_indices_count[class_] += self.n_samples #每类样本使用过的数量增加此次使用的样本数量
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]): #使用过的加上下次的样本沟用不够，大于表示不够下次使用了
                    np.random.shuffle(self.label_to_indices[class_])#不够下次使用，那就将其重新打乱，并将每类的使用数量归零
                    self.used_label_indices_count[class_] = 0

            yield indices #每类都获取到了后，就送出去，送出去的是样本的索引
            self.count += self.n_classes * self.n_samples #增加本批样本数量

    def __len__(self):
        return len(self.dataset) // self.batch_size

