import numpy as np
import os
import cv2
from inspect import isclass
from sklearn.metrics import roc_auc_score, roc_curve, auc
import torch
import torch.nn.functional as F
import rpy2
from rpy2.robjects import FloatVector, pandas2ri
from rpy2.robjects.packages import importr


def list2np(outputs, targets):
    all_outputs = outputs[0]
    all_targets = targets[0]
    outputs, targets = outputs[1:], targets[1:]
    for output, target in zip(outputs, targets):
        all_outputs = np.vstack((all_outputs, output))
        all_targets = np.vstack((all_targets, target))
    return all_outputs, all_targets


def compute_icc(outputs, targets):
    outputs, targets = list2np(outputs, targets)
    
    psych = importr("psych")
    num_au = outputs.shape[1]

    total_icc = []
    for au_idx in range(num_au):
        output = outputs[:, au_idx]
        target = targets[:, au_idx]
        matrix = np.column_stack((output, target)).reshape(-1)
        values = rpy2.robjects.r.matrix(
            FloatVector(matrix),
            ncol=2, byrow=True
        )
        icc = psych.ICC(values)
        icc_df = pandas2ri.ri2py(icc[0])
        au_icc = icc_df.ICC[2]
        total_icc.append(au_icc)
    return total_icc

def compute_mae(outputs, labels):

    num_class = outputs.shape[1]

    outputs, labels = torch.tensor(outputs), torch.tensor(labels)

    total_mae = []

    for au_idx in range(num_class):
        output = outputs[:, au_idx]
        label = labels[:, au_idx]
        au_mae = F.l1_loss(output, label)
        total_mae.append(au_mae)

    return total_mae


def record_intensity_metrics(outputs, labels, loss, num_class, savepath, mode='val'):

    labels = labels.reshape(-1, num_class)
    outputs = outputs.reshape(-1, num_class)

    icc = np.array(compute_icc(outputs, labels))

    # print(outputs.shape, labels.shape)
    mae = np.array(compute_mae(outputs, labels))

    res_txt_path = os.path.join(savepath, 'log.txt')
    fp = open(res_txt_path, 'a')
    fp.write("===> loss: {}\n".format(loss))
    fp.write("===> icc: {}\n".format(icc))
    fp.write("===> mae: {}\n".format(mae))
    fp.write("===> average icc: {}\n".format(np.mean(icc)))
    fp.write("===> average mae: {}\n".format(np.mean(mae)))
    fp.close()

    print("icc {} \nmae {}".format(icc, mae))
    print("average icc {}\n".format(np.mean(icc)))
    print("average mae {}\n".format(np.mean(mae)))

    return icc, mae, np.mean(icc), np.mean(mae)


def module_to_dict(module, exclude=[]):
    return dict([(x, getattr(module, x)) for x in dir(module)
                if isclass(getattr(module, x))
                and x not in exclude
                and getattr(module, x) not in exclude])


def cal_f1score(outputs, labels, thresh=0.5, num_class=10):

    labels = labels.reshape(-1, num_class)
    outputs = outputs.reshape(-1, num_class)

    TP = np.zeros(num_class)
    TN = np.zeros(num_class)
    FN = np.zeros(num_class)
    FP = np.zeros(num_class)
    recall = np.zeros(num_class)
    precision = np.zeros(num_class)
    accuracy = np.zeros(num_class)
    F1_score = np.ones(num_class)
    F1_score = -1 * F1_score

    outputs[outputs > thresh] = 1
    outputs[outputs <= thresh] = 0
    for index, output in enumerate(outputs):
        label = labels[index]
        for i, y in enumerate(output):
            if (y == 1 and label[i] == 1):
                TP[i] += 1
            elif (y == 0 and label[i] == 1):
                FN[i] += 1
            elif (y == 1 and label[i] == 0):
                FP[i] += 1
            else: TN[i] += 1

    for i in range(num_class):
        if (TP[i] + FN[i] > 0):
            recall[i] = float(TP[i]) / float(TP[i] + FN[i])
            precision[i] = float(TP[i]) / float(TP[i] + FP[i] + 1e-6)
            F1_score[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i] + 1e-6)
            accuracy[i] = float(TP[i] + TN[i]) / (TP[i] + TN[i] + FN[i] + FP[i])

    return TP, TN, FN, FP, F1_score, recall, precision, accuracy


def record_metrics(outputs, labels, loss, num_class, savepath, mode='val'):

    labels = labels.reshape(-1, num_class)
    outputs = outputs.reshape(-1, num_class)

    TP, TN, FN, FP, F1_score, recall, precision, accuracy = cal_f1score(outputs=outputs, labels=labels, 
                                                                            num_class=num_class)

    res_txt_path = os.path.join(savepath, 'log.txt')
    fp = open(res_txt_path, 'a')
    fp.write("===> loss: {}\n".format(loss))
    fp.write("===> f1: {}\n".format(F1_score))
    fp.write("===> acc: {}\n".format(accuracy))
    fp.write("===> TP: {}\n".format(TP))
    fp.write("===> TN: {}\n".format(TN))
    fp.write("===> FN: {}\n".format(FN))
    fp.write("===> FP: {}\n".format(FP))
    fp.write("===> rec: {}\n".format(recall))
    fp.write("===> prec: {}\n".format(precision))
    fp.write("===> average f1: {}\n".format(np.mean(F1_score)))
    fp.write("===> average acc: {}\n".format(np.mean(accuracy)))
    fp.close()

    print("f1 {} \nacc {}".format(F1_score, accuracy))
    print("average f1 {}\n".format(np.mean(F1_score)))
    print("average acc {}\n".format(np.mean(accuracy)))

    return F1_score, accuracy, np.mean(F1_score), np.mean(accuracy)
