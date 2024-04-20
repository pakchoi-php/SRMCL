import torch
import numpy as np
import torch.nn as nn
from sklearn.metrics import recall_score, f1_score
from sklearn.metrics import accuracy_score


class Meter(object):

    def __init__(self):
        """
        To record the measure the performance.
        """
        self.Y_true = np.array(

          []

        , dtype=np.int)
        self.Y_pred = np.array(


            []

            , dtype=np.int)

    def add(self, y_true, y_pred, verbose=False):
        if len(self.Y_true.shape) != len(y_true.shape):
            print('shape self.Y_true', self.Y_true.shape)
            print('y_true', y_true.shape)

        self.Y_true = np.concatenate((self.Y_true, y_true))
        self.Y_pred = np.concatenate((self.Y_pred, y_pred))

    def reset(self):
        self.Y_true = np.array([], dtype=np.int)
        self.Y_pred = np.array([], dtype=np.int)

    def value(self,num_class):

        eye = np.eye(num_class, dtype=np.int)
        Y_true = eye[self.Y_true]
        Y_pred = eye[self.Y_pred]
        acc = accuracy_score(Y_true, Y_pred)
        uar = recall_score(Y_true, Y_pred, average=None)
        uf1 = f1_score(Y_true, Y_pred, average=None)
        f1 = f1_score(Y_true, Y_pred, average="macro")
        return  uar,uf1,acc,f1

if __name__ == '__main__':
    x_meter = Meter()
    y_true = x_meter.Y_true.copy()
    y_pre =  x_meter.Y_pred.copy()
    score=x_meter.value(num_class=3)
    print(score[1].mean(),score[0].mean())
    result_txt = './evaluation/result.txt'
    print(y_true)
    output = " y_pred : %f, y_true :  %g" % (y_pre, y_true)
    with open(result_txt, "a+") as f:
        f.write(output + '\n')
        f.close()
