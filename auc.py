import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Calculate AUC for predictions.')
parser.add_argument('--input_path', default="results/svm_bow_lexical/baseline1/", type=str)
parser.add_argument('--input_file', type=str)
parser.add_argument('--threshold', default=0, type=float)
args = parser.parse_args()
print(args)

threshold = args.threshold
frequence = 0.05
with open(args.input_path + args.input_file) as f:
    data = f.readlines()

out_file = args.input_file.split(".")[0] + '.auc'
with open(args.input_path + out_file, 'w') as f:
    total = 0.
    negative = 0.
    positive = 0.
    predictions = []
    for line in data[1:]:
        arr = line.strip().split('\t')
        idx = int(arr[0])
        if idx == 1:
            positive += 1
            predictions.append([1, float(arr[-1])])
        else:
            negative += 1
            predictions.append([-1, float(arr[-1])])

    total = negative + positive
    f.write('Total number of instances: ' + str(int(total)) + '\n')
    f.write('P: ' + str(int(positive)) + '\n')
    f.write('N: ' + str(int(negative)) + '\n')
    f.write('-' * 30 + '\n')
    f.write('Figure\n')
    f.write('-' * 30 + '\n')
    f.write('decision_boundary\tTP\tFP\tTPR\tFPR\n')
    TP = 0
    FP = 0
    TP_0 = 0
    FP_0 = 0
    AUC = 0.

    target_TP = 1
    target_FPR = 1e-5
    table = []
    table_header = []
    predictions.sort(key=lambda x: x[-1], reverse=True)
    for i, pred in enumerate(predictions):
        if pred[0] > 0:
            TP += 1
            if pred[1] > threshold:
                TP_0 += 1
            AUC += FP
        else:
            FP += 1
            if pred[1] > threshold:
                FP_0 += 1

        if TP >= target_TP or i == int(total) - 1:
            target_TP += frequence * positive
            f.write(f"{pred[1]}\t{TP}\t{FP}\t{TP / positive}\t{FP / negative}\n")

        if FP >= target_FPR * negative or i == int(total) - 1:
            table_header.append(target_FPR)
            table.append(TP / positive)
            target_FPR *= 10

    f.write('-' * 30 + '\n')
    f.write('Table\n')
    f.write('-' * 30 + '\n')
    f.write('FPR\tTPR\n')
    for tpr, fpr in zip(table, table_header):
        f.write(f"{fpr}\t{tpr}\n")
    f.write('-' * 30 + '\n')
    f.write('AUC:\t' + str(1. - (AUC / (positive * negative))) + '\n')
    f.write('When the decision boundary is set to be 0\n')
    f.write('TP:\t' + str(int(TP_0)) + '\n')
    f.write('FN:\t' + str(int(positive - TP_0)) + '\n')
    f.write('FP:\t' + str(int(FP_0)) + '\n')
    f.write('TN:\t' + str(int(negative - FP_0)) + '\n')
